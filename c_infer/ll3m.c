/* Portable C inference for LL3M (v2 / CLM2 only).
 * Cache tiers:
 *   L1i — hot kernels (.text.hot)
 *   L1d — single-token residual/work vectors (hot_l1 pad)
 *   L2  — decode attention scores
 *   L3  — full weights + KV + prefill scratch
 */
#define _GNU_SOURCE
#include "ll3m.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* mlock ON by default; disable with -DCLM_NO_MLOCK (or -DCLM_DO_MLOCK=0). */
#if defined(CLM_NO_MLOCK)
#undef CLM_DO_MLOCK
#define CLM_DO_MLOCK 0
#elif !defined(CLM_DO_MLOCK)
#define CLM_DO_MLOCK 1
#endif

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sched.h>
#include <errno.h>
#include <unistd.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CLM_HOT  __attribute__((hot, section(".text.hot")))
#define CLM_COLD __attribute__((cold))
#define CLM_RESTRICT __restrict__
#define CLM_ALIGN(n) __attribute__((aligned(n)))
#else
#define CLM_HOT
#define CLM_COLD
#define CLM_RESTRICT
#define CLM_ALIGN(n)
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define CLM_PREFETCH_L2(p) _mm_prefetch((const char *)(p), _MM_HINT_T1)
#define CLM_PREFETCH_L1(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#define CLM_PREFETCH_L2(p) ((void)0)
#define CLM_PREFETCH_L1(p) ((void)0)
#endif

/* ---- arena + residency (from legacy path; layout-agnostic) ---- */
static void raise_memlock_limit(void) {
#if defined(__linux__)
    struct rlimit rl;
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0)
        return;
    rl.rlim_cur = rl.rlim_max;
    if (rl.rlim_max != RLIM_INFINITY) {
        /* try unlimited; ignore failure (may need CAP_IPC_LOCK / ulimit -l) */
        struct rlimit inf = {RLIM_INFINITY, RLIM_INFINITY};
        if (setrlimit(RLIMIT_MEMLOCK, &inf) != 0)
            setrlimit(RLIMIT_MEMLOCK, &rl);
    } else {
        setrlimit(RLIMIT_MEMLOCK, &rl);
    }
#endif
}

/* L3-oriented: one load per 64B line (fault + fill L3 without thrashing L1). */
static void prefault_bytes(void *p, size_t n) {
    if (!p || n == 0)
        return;
    volatile unsigned char *c = (volatile unsigned char *)p;
    for (size_t i = 0; i < n; i += 64)
        (void)c[i];
    if (n > 0)
        (void)c[n - 1];
}

/*
 * L1-oriented: dense float R+W so lines land in L1d (and stay if reused soon).
 * Used on hot_l1 pad only (must be small enough for L1d).
 */
static void touch_dense_f32(float *p, size_t n_floats) {
    if (!p || n_floats == 0)
        return;
    volatile float *v = (volatile float *)p;
    float acc = 0.f;
    for (size_t i = 0; i < n_floats; i++) {
        acc += v[i];
        v[i] = v[i]; /* write same value: allocate dirty line in L1d */
    }
    (void)acc;
}

/* L2-oriented: stride that keeps pressure moderate; good for score pads / tiles. */
static void touch_l2_stream(void *p, size_t n) {
    if (!p || n == 0)
        return;
    volatile unsigned char *c = (volatile unsigned char *)p;
    /* 64B lines, but touch two passes so the working set sticks in L2. */
    for (size_t i = 0; i < n; i += 64)
        (void)c[i];
    for (size_t i = 0; i < n; i += 64)
        (void)c[i];
}

static int madvise_hugepage(void *p, size_t n) {
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    if (!p || n == 0)
        return 0;
    return madvise(p, n, MADV_HUGEPAGE);
#else
    (void)p;
    (void)n;
    return -1;
#endif
}

static int mlock_bytes(void *p, size_t n) {
#if defined(__linux__) && CLM_DO_MLOCK
    if (!p || n == 0)
        return 0;
    if (mlock(p, n) != 0)
        return -1;
#if defined(MADV_WILLNEED)
    (void)madvise(p, n, MADV_WILLNEED);
#endif
    (void)madvise_hugepage(p, n);
    return 0;
#else
    (void)p;
    (void)n;
    return 0;
#endif
}

#ifndef MAP_HUGE_SHIFT
#define MAP_HUGE_SHIFT 26
#endif
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << MAP_HUGE_SHIFT)
#endif

/*
 * Prefer 2MiB hugetlb mmap when the admin reserved pages (nr_hugepages > 0).
 * Else posix_memalign + MADV_HUGEPAGE (THP best-effort).
 * *out_mmap = 1 if caller must munmap; *out_hugetlb = 1 if MAP_HUGETLB.
 */
static void *arena_alloc_ex(size_t bytes, size_t *out_bytes, int *out_mmap,
                            int *out_hugetlb) {
    if (out_mmap)
        *out_mmap = 0;
    if (out_hugetlb)
        *out_hugetlb = 0;
#if defined(__linux__)
    raise_memlock_limit();
    const size_t huge = 2ull * 1024ull * 1024ull;
    /*
     * MAP_HUGETLB only when payload ≥ 2MiB (avoid rounding tiny arenas up and
     * wasting L3). Needs admin-reserved nr_hugepages.
     */
    if (bytes >= huge) {
        size_t n_huge = (bytes + huge - 1) & ~(huge - 1);
        void *hp = mmap(NULL, n_huge, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (hp != MAP_FAILED) {
            memset(hp, 0, n_huge);
            prefault_bytes(hp, n_huge);
            (void)mlock_bytes(hp, n_huge);
            if (out_bytes)
                *out_bytes = n_huge;
            if (out_mmap)
                *out_mmap = 1;
            if (out_hugetlb)
                *out_hugetlb = 1;
            return hp;
        }
    }
    /* Fallback: 4KiB pages + THP hint */
    size_t page = 4096;
    size_t n = (bytes + page - 1) & ~(page - 1);
    void *p = NULL;
    if (posix_memalign(&p, page, n) != 0)
        return NULL;
    memset(p, 0, n);
    prefault_bytes(p, n);
    (void)mlock_bytes(p, n);
    if (out_bytes)
        *out_bytes = n;
    return p;
#else
    void *p = calloc(1, bytes);
    if (out_bytes)
        *out_bytes = bytes;
    return p;
#endif
}

static void *arena_alloc(size_t bytes, size_t *out_bytes) {
    return arena_alloc_ex(bytes, out_bytes, NULL, NULL);
}

static void arena_free_ex(void *p, size_t bytes, int is_mmap) {
    if (!p)
        return;
#if defined(__linux__) && CLM_DO_MLOCK
    munlock(p, bytes);
#endif
#if defined(__linux__)
    if (is_mmap)
        munmap(p, bytes);
    else
        free(p);
#else
    (void)bytes;
    (void)is_mmap;
    free(p);
#endif
}

static void arena_free(void *p, size_t bytes) {
    arena_free_ex(p, bytes, 0);
}

int clm_pin_arenas(ClmModel *m, ClmState *s) {
    int ok = 0;
    raise_memlock_limit();
    if (m && m->arena) {
        if (mlock_bytes(m->arena, m->arena_bytes) == 0)
            m->locked = 1;
        else
            ok = -1;
        prefault_bytes(m->arena, m->arena_bytes);
    }
    if (s && s->arena) {
        if (mlock_bytes(s->arena, s->arena_bytes) == 0)
            s->locked = 1;
        else
            ok = -1;
        prefault_bytes(s->arena, s->arena_bytes);
    }
    if (s && s->hot_l1) {
        if (mlock_bytes(s->hot_l1, s->hot_l1_bytes) == 0)
            s->hot_l1_locked = 1;
        else
            ok = -1;
        prefault_bytes(s->hot_l1, s->hot_l1_bytes);
    }
    if (s && s->hot_l2) {
        if (mlock_bytes(s->hot_l2, s->hot_l2_bytes) == 0)
            s->hot_l2_locked = 1;
        else
            ok = -1;
        prefault_bytes(s->hot_l2, s->hot_l2_bytes);
    }
    return ok;
}

void clm_prefault(const ClmModel *m, const ClmState *s) {
    if (m && m->arena)
        prefault_bytes(m->arena, m->arena_bytes);
    if (s && s->arena)
        prefault_bytes(s->arena, s->arena_bytes);
    if (s && s->hot_l1)
        prefault_bytes(s->hot_l1, s->hot_l1_bytes);
    if (s && s->hot_l2)
        prefault_bytes(s->hot_l2, s->hot_l2_bytes);
}

void clm_touch_hierarchy(const ClmModel *m, const ClmState *s) {
    /*
     * Order matters: install small hot sets first so they win L1/L2, then
     * stream bulk arenas into L3 without blowing out the hot sets permanently
     * (hot pads are re-touched after L3 stream).
     */
    if (s && s->hot_l1 && s->hot_l1_bytes)
        touch_dense_f32((float *)s->hot_l1, s->hot_l1_bytes / sizeof(float));
    if (s && s->hot_l2 && s->hot_l2_bytes)
        touch_l2_stream(s->hot_l2, s->hot_l2_bytes);
    if (m && m->arena)
        prefault_bytes(m->arena, m->arena_bytes); /* sequential L3 fill */
    if (s && s->arena)
        prefault_bytes(s->arena, s->arena_bytes);
    /* Re-install L1/L2 hot after L3 stream so decode starts hot. */
    if (s && s->hot_l1 && s->hot_l1_bytes)
        touch_dense_f32((float *)s->hot_l1, s->hot_l1_bytes / sizeof(float));
    if (s && s->hot_l2 && s->hot_l2_bytes)
        touch_l2_stream(s->hot_l2, s->hot_l2_bytes);
}

void clm_warm(const ClmModel *m, ClmState *s, int passes) {
    if (!m || !s || s->n_tokens < 1)
        return;
    if (passes < 1)
        passes = 1;
    clm_touch_hierarchy(m, s);
    for (int i = 0; i < passes; i++) {
        clm_forward_full(m, s);
        /* After each forward, re-touch hot pads (decode path uses them). */
        if (s->hot_l1 && s->hot_l1_bytes)
            touch_dense_f32((float *)s->hot_l1, s->hot_l1_bytes / sizeof(float));
        if (s->hot_l2 && s->hot_l2_bytes)
            touch_l2_stream(s->hot_l2, s->hot_l2_bytes);
        if ((i & 1) == 0) {
            /* Alternate full L3 refresh so weight pages stay resident. */
            prefault_bytes(m->arena, m->arena_bytes);
            prefault_bytes(s->arena, s->arena_bytes);
        }
    }
    clm_touch_hierarchy(m, s);
}

static size_t clm_sysfs_cache_size(int index) {
#if defined(__linux__)
    char path[128];
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/size", index);
    FILE *f = fopen(path, "r");
    if (!f)
        return 0;
    char buf[64];
    if (!fgets(buf, sizeof buf, f)) {
        fclose(f);
        return 0;
    }
    fclose(f);
    char *end = NULL;
    unsigned long v = strtoul(buf, &end, 10);
    if (end && (*end == 'K' || *end == 'k'))
        return (size_t)v * 1024ull;
    if (end && (*end == 'M' || *end == 'm'))
        return (size_t)v * 1024ull * 1024ull;
    return (size_t)v;
#else
    (void)index;
    return 0;
#endif
}

/* index0=L1d, index1=L1i, index2=L2, index3=L3 on typical x86. */
size_t clm_l1d_bytes(void) { return clm_sysfs_cache_size(0); }
size_t clm_l1i_bytes(void) { return clm_sysfs_cache_size(1); }
size_t clm_l2_bytes(void) { return clm_sysfs_cache_size(2); }
size_t clm_l3_ccd_bytes(void) { return clm_sysfs_cache_size(3); }

static size_t clm_self_text_bytes(void) {
#if defined(__linux__)
    /* Approximate executable text size from /proc/self/exe (full file; close enough). */
    char path[64];
    snprintf(path, sizeof path, "/proc/self/exe");
    struct stat st;
    if (stat(path, &st) == 0 && st.st_size > 0)
        return (size_t)st.st_size;
#endif
    return 0;
}

int clm_fits_l3_ccd(const ClmModel *m, const ClmState *s) {
    size_t l3 = clm_l3_ccd_bytes();
    size_t ws = clm_inference_ws_bytes(m, s);
    if (l3 == 0)
        return ws > 0; /* unknown topology: only check positive */
    return ws <= l3;
}

/* Collect unique L3 shared_cpu_list strings (one per CCD). Returns count. */
static int clm_l3_ccd_lists(char groups[][256], int max_g) {
    int ng = 0;
#if defined(__linux__)
    char path[128];
    for (int cpu = 0; cpu < 256 && ng < max_g; cpu++) {
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpu);
        FILE *f = fopen(path, "r");
        if (!f)
            continue;
        char line[256];
        if (!fgets(line, sizeof line, f)) {
            fclose(f);
            continue;
        }
        fclose(f);
        size_t L = strlen(line);
        while (L && (line[L - 1] == '\n' || line[L - 1] == '\r'))
            line[--L] = 0;
        int seen = 0;
        for (int g = 0; g < ng; g++) {
            if (strcmp(groups[g], line) == 0) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            strncpy(groups[ng], line, 255);
            groups[ng][255] = 0;
            ng++;
        }
    }
#else
    (void)groups;
    (void)max_g;
#endif
    return ng;
}

/* Parse list "0-7,16-23" into cpu_set; return quiet single-core pick.
 * Prefer preferred_cpu if in set; else highest "physical-ish" id (id < nproc/2
 * when SMT, else max id) so we avoid CPU0 which is usually busiest. */
static int clm_parse_cpu_list(const char *list, cpu_set_t *set, int preferred_cpu) {
    CPU_ZERO(set);
    if (!list || !*list)
        return -1;
    char list_copy[256];
    strncpy(list_copy, list, sizeof list_copy - 1);
    list_copy[sizeof list_copy - 1] = 0;
    char *save = NULL;
    int max_cpu = -1;
    int max_phys = -1;
    int nproc = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nproc < 1)
        nproc = 256;
    int phys_cut = nproc > 1 ? nproc / 2 : nproc;
    for (char *tok = strtok_r(list_copy, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
        int a = 0, b = 0;
        if (sscanf(tok, "%d-%d", &a, &b) == 2) {
            for (int c = a; c <= b; c++) {
                CPU_SET(c, set);
                if (c > max_cpu)
                    max_cpu = c;
                if (c < phys_cut && c > max_phys)
                    max_phys = c;
            }
        } else if (sscanf(tok, "%d", &a) == 1) {
            CPU_SET(a, set);
            if (a > max_cpu)
                max_cpu = a;
            if (a < phys_cut && a > max_phys)
                max_phys = a;
        }
    }
    if (preferred_cpu >= 0 && CPU_ISSET(preferred_cpu, set))
        return preferred_cpu;
    if (max_phys >= 0)
        return max_phys;
    return max_cpu;
}

int clm_bind_l3_ccd_ex(int ccd_index, int single_core) {
#if defined(__linux__)
    char groups[8][256];
    int ng = clm_l3_ccd_lists(groups, 8);
    if (ng < 1 || ccd_index < 0 || ccd_index >= ng)
        return -1;
    cpu_set_t set;
    int pick = clm_parse_cpu_list(groups[ccd_index], &set, -1);
    if (single_core && pick >= 0) {
        /* Private L2 is per-core; pin one quiet CPU of the CCD. */
        CPU_ZERO(&set);
        CPU_SET(pick, &set);
    }
    if (sched_setaffinity(0, sizeof set, &set) != 0)
        return -1;
    return 0;
#else
    (void)ccd_index;
    (void)single_core;
    return -1;
#endif
}

int clm_bind_l3_ccd(int ccd_index) { return clm_bind_l3_ccd_ex(ccd_index, 0); }

int clm_cpu_isolation_present(void) {
#if defined(__linux__)
    FILE *f = fopen("/sys/devices/system/cpu/isolated", "r");
    if (!f)
        return 0;
    char buf[256];
    if (!fgets(buf, sizeof buf, f)) {
        fclose(f);
        return 0;
    }
    fclose(f);
    /* non-empty and not just whitespace/newline */
    for (char *p = buf; *p; p++) {
        if (*p != ' ' && *p != '\n' && *p != '\r' && *p != '\t')
            return 1;
    }
#endif
    return 0;
}

/* Parse first CPU id from a list like "3" or "2-5,8". */
static int clm_first_cpu_in_list(const char *list) {
    int a = 0, b = 0;
    if (!list || !*list)
        return -1;
    if (sscanf(list, "%d-%d", &a, &b) >= 1)
        return a;
    return -1;
}

int clm_bind_residency_cpu(int ccd_index, int single_core, int preferred_cpu,
                           int *out_cpu) {
    if (out_cpu)
        *out_cpu = -1;
#if defined(__linux__)
    /* Env override when CLI did not pass --cpu */
    if (preferred_cpu < 0) {
        const char *env = getenv("CLM_PIN_CPU");
        if (env && *env) {
            char *end = NULL;
            long v = strtol(env, &end, 10);
            if (end != env && v >= 0 && v < CPU_SETSIZE)
                preferred_cpu = (int)v;
        }
    }

    /* Prefer kernel-isolated CPUs if the admin set isolcpus=... */
    FILE *f = fopen("/sys/devices/system/cpu/isolated", "r");
    if (f) {
        char buf[256];
        if (fgets(buf, sizeof buf, f)) {
            size_t L = strlen(buf);
            while (L && (buf[L - 1] == '\n' || buf[L - 1] == '\r'))
                buf[--L] = 0;
            if (buf[0]) {
                cpu_set_t set;
                int cpu = clm_parse_cpu_list(buf, &set, preferred_cpu);
                if (cpu < 0)
                    cpu = clm_first_cpu_in_list(buf);
                if (cpu >= 0 && single_core) {
                    CPU_ZERO(&set);
                    CPU_SET(cpu, &set);
                    fclose(f);
                    if (sched_setaffinity(0, sizeof set, &set) == 0) {
                        if (out_cpu)
                            *out_cpu = cpu;
                        return 0;
                    }
                    return -1;
                }
                if (cpu >= 0 && !single_core) {
                    fclose(f);
                    if (sched_setaffinity(0, sizeof set, &set) == 0) {
                        if (out_cpu)
                            *out_cpu = cpu;
                        return 0;
                    }
                    return -1;
                }
            }
        }
        fclose(f);
    }

    /* Keep existing single-CPU affinity if already on the requested CCD
     * (taskset -c N ./ll3m_c …) and no explicit preferred_cpu. */
    if (preferred_cpu < 0 && single_core) {
        cpu_set_t cur;
        if (sched_getaffinity(0, sizeof cur, &cur) == 0) {
            int only = -1, nset = 0;
            for (int c = 0; c < CPU_SETSIZE; c++) {
                if (CPU_ISSET(c, &cur)) {
                    nset++;
                    only = c;
                }
            }
            if (nset == 1 && only >= 0)
                preferred_cpu = only;
        }
    }

    char groups[8][256];
    int ng = clm_l3_ccd_lists(groups, 8);
    if (ng < 1 || ccd_index < 0 || ccd_index >= ng)
        return -1;
    cpu_set_t set;
    int pick = clm_parse_cpu_list(groups[ccd_index], &set, preferred_cpu);
    if (pick < 0)
        return -1;
    if (single_core) {
        CPU_ZERO(&set);
        CPU_SET(pick, &set);
    }
    if (sched_setaffinity(0, sizeof set, &set) != 0)
        return -1;
    if (out_cpu)
        *out_cpu = single_core ? pick : clm_first_cpu_in_list(groups[ccd_index]);
    return 0;
#else
    (void)ccd_index;
    (void)single_core;
    (void)preferred_cpu;
    (void)out_cpu;
    return -1;
#endif
}

int clm_residency_hard_enable(ClmModel *m, ClmState *s, int ccd_index,
                              int preferred_cpu, ClmHardResidency *out) {
    if (out)
        memset(out, 0, sizeof(*out));
    int rc = 0;
    raise_memlock_limit();

    if (out) {
        out->isol_present = clm_cpu_isolation_present();
        out->bound_cpu = -1;
        out->nice_prio = -1;
        out->hugetlb_mmap_ok = m && m->arena_hugetlb;
    }

#if defined(__linux__) && CLM_DO_MLOCK
    if (mlockall(MCL_CURRENT | MCL_FUTURE) == 0) {
        if (out)
            out->mlockall_ok = 1;
    } else {
        rc |= 1;
    }
#else
    rc |= 1;
#endif

    {
        int cpu = -1;
        if (clm_bind_residency_cpu(ccd_index, /*single_core=*/1, preferred_cpu,
                                   &cpu) == 0) {
            if (out) {
                out->bind_ok = 1;
                out->bound_cpu = cpu;
            }
        } else {
            rc |= 2;
        }
    }

    if (clm_pin_arenas(m, s) == 0) {
        if (out)
            out->pin_ok = 1;
    } else {
        rc |= 4;
    }

    int hp = 0;
    if (m && m->arena_hugetlb)
        hp++;
    if (m && m->arena && madvise_hugepage(m->arena, m->arena_bytes) == 0)
        hp++;
    if (s && s->arena_hugetlb)
        hp++;
    if (s && s->arena && madvise_hugepage(s->arena, s->arena_bytes) == 0)
        hp++;
    if (s && s->hot_l1 && madvise_hugepage(s->hot_l1, s->hot_l1_bytes) == 0)
        hp++;
    if (s && s->hot_l2 && madvise_hugepage(s->hot_l2, s->hot_l2_bytes) == 0)
        hp++;
    if (out)
        out->hugepage_ok = hp > 0 || (m && m->arena_hugetlb);
    if (hp == 0 && !(m && m->arena_hugetlb))
        rc |= 8;

#if defined(__linux__)
    {
        struct sched_param sp;
        memset(&sp, 0, sizeof sp);
        int pmax = sched_get_priority_max(SCHED_FIFO);
        int pmin = sched_get_priority_min(SCHED_FIFO);
        int prio = pmin + (pmax - pmin) / 4;
        if (prio < pmin)
            prio = pmin;
        sp.sched_priority = prio;
        if (sched_setscheduler(0, SCHED_FIFO, &sp) == 0) {
            if (out) {
                out->fifo_ok = 1;
                out->nice_prio = prio;
            }
        } else {
            if (out)
                out->nice_prio = -1;
            rc |= 16;
            (void)nice(-10);
        }
    }
#else
    rc |= 16;
#endif

    clm_touch_hierarchy(m, s);
    return rc;
}

/* ---- model IO: L3T2/L3M2 (+ legacy CLT2/CLM2) ---- */

static int CLM_COLD clm_model_load_fp(ClmModel *m, FILE *f, const char *label) {
    char magic[4];
    if (fread(magic, 1, 4, f) != 4) {
        fprintf(stderr, "short magic in %s\n", label ? label : "(mem)");
        fclose(f);
        return -1;
    }
    /* L3T2 product ternary; L3B2 research binary; L3M2 f32; legacy CLT2/CLM2 */
    if (memcmp(magic, "L3T2", 4) == 0 || memcmp(magic, "CLT2", 4) == 0) {
        extern int clm_model_load_v2t(ClmModel *m, FILE *f, const char *path);
        return clm_model_load_v2t(m, f, label);
    }
    if (memcmp(magic, "L3B2", 4) == 0) {
        extern int clm_model_load_v2b(ClmModel *m, FILE *f, const char *path);
        return clm_model_load_v2b(m, f, label);
    }
    if (memcmp(magic, "L3M2", 4) == 0 || memcmp(magic, "CLM2", 4) == 0) {
        extern int clm_model_load_v2(ClmModel *m, FILE *f, const char *path);
        return clm_model_load_v2(m, f, label);
    }
    fprintf(stderr,
            "bad magic in %s (want L3T2/L3B2/L3M2 or legacy CLT2/CLM2)\n",
            label ? label : "(mem)");
    fclose(f);
    return -1;
}

int CLM_COLD clm_model_load(ClmModel *m, const char *path) {
    memset(m, 0, sizeof(*m));
    if (!path || !path[0]) {
        fprintf(stderr, "clm_model_load: empty path\n");
        return -1;
    }
    /* Secondary build: treat special path as in-binary pack. */
    if (!strcmp(path, ":embedded:") || !strcmp(path, "@embedded")) {
        return clm_model_load_embedded(m);
    }
    FILE *f = fopen(path, "rb");
    if (!f) {
        perror(path);
        return -1;
    }
    return clm_model_load_fp(m, f, path);
}

int CLM_COLD clm_model_load_mem(ClmModel *m, const void *data, size_t nbytes,
                                const char *label) {
    memset(m, 0, sizeof(*m));
    if (!data || nbytes < 4) {
        fprintf(stderr, "clm_model_load_mem: empty blob\n");
        return -1;
    }
#if defined(__linux__) || defined(_GNU_SOURCE) || defined(__USE_GNU)
    FILE *f = fmemopen((void *)data, nbytes, "rb");
    if (!f) {
        perror("fmemopen");
        return -1;
    }
    return clm_model_load_fp(m, f, label ? label : ":mem:");
#else
    (void)label;
    fprintf(stderr, "clm_model_load_mem: fmemopen not available on this platform\n");
    return -1;
#endif
}

#if defined(CLM_EMBED_WEIGHTS)
/* objcopy -I binary symbols from clm_pack.ll3m (see Makefile embed). */
extern const unsigned char _binary_clm_pack_ll3m_start[];
extern const unsigned char _binary_clm_pack_ll3m_end[];

int clm_has_embedded_weights(void) { return 1; }

const void *clm_embedded_weights_ptr(void) {
    return (const void *)_binary_clm_pack_ll3m_start;
}

size_t clm_embedded_weights_size(void) {
    return (size_t)(_binary_clm_pack_ll3m_end - _binary_clm_pack_ll3m_start);
}

int CLM_COLD clm_model_load_embedded(ClmModel *m) {
    size_t n = clm_embedded_weights_size();
    fprintf(stderr, "loading embedded pack (%.2f MiB)\n", n / (1024.0 * 1024.0));
    return clm_model_load_mem(m, clm_embedded_weights_ptr(), n, ":embedded:");
}
#else
int clm_has_embedded_weights(void) { return 0; }
const void *clm_embedded_weights_ptr(void) { return NULL; }
size_t clm_embedded_weights_size(void) { return 0; }
int clm_model_load_embedded(ClmModel *m) {
    (void)m;
    fprintf(stderr,
            "no embedded weights (rebuild with: make -C c_infer embed "
            "EMBED_MODEL=path/to/model.ll3m)\n");
    return -1;
}
#endif

void clm_model_free(ClmModel *m) {
    if (!m)
        return;
    free(m->tok_json);
    m->tok_json = NULL;
    free(m->map_cache);
    m->map_cache = NULL;
    m->map_cache_ready = 0;
    if (m->cfg.arch == 2 && !m->arena_mmap)
        free(m->arena); /* v2 f32 path uses malloc */
    else
        arena_free_ex(m->arena, m->arena_bytes, m->arena_mmap);
    memset(m, 0, sizeof(*m));
}

/* forward decl: create may abort via free */
void clm_state_free(ClmState *s);

int clm_state_create_kv(ClmState *s, const ClmModel *m, int token_cap, int kv_cap) {
    memset(s, 0, sizeof(*s));
    if (!m)
        return -1;
    int T = m->cfg.block_size;
    int D = m->cfg.n_embd;
    int H = m->cfg.n_head;
    int L = m->cfg.n_layer;
    int V = m->cfg.vocab_size;
    int nkv = m->cfg.n_kv_head > 0 ? m->cfg.n_kv_head : H;
    int hd = m->cfg.head_dim > 0 ? m->cfg.head_dim : (D / H);
    int kv_dim = nkv * hd;
    if (kv_cap <= 0 || kv_cap > T)
        kv_cap = T;
    if (token_cap < kv_cap)
        token_cap = kv_cap;
    s->token_cap = token_cap;
    s->kv_cap = kv_cap;

    /*
     * L3 arena: decode-first layout (v2 product).
     *   KV is MQA/GQA-sized: L * kv_cap * nkv * hd
     *   No full H*T*T prefill att matrix (decode uses hot_l2 scores).
     *   Scratch rows are single-token; multi-token prefill is serial decode.
     */
    int Tkv = kv_cap;
    size_t floats = 0;
    floats += (size_t)D;                 /* x */
    floats += (size_t)D;                 /* xb */
    floats += (size_t)D;                 /* xb2 */
    floats += (size_t)(D + 2 * kv_dim);  /* qkv scratch */
    floats += (size_t)H * Tkv;           /* att fallback (if no hot_l2) */
    size_t mid = (m->cfg.arch == 2) ? (size_t)m->cfg.intermediate : (size_t)(4 * D);
    if (mid < (size_t)(4 * D))
        mid = (size_t)(4 * D);
    floats += 2 * mid;                   /* mlp / SwiGLU gate+up */
    floats += (size_t)V;                 /* logits */
    floats += (size_t)L * Tkv * kv_dim;  /* k_cache */
    floats += (size_t)L * Tkv * kv_dim;  /* v_cache */
    size_t bytes = floats * sizeof(float) + (size_t)token_cap * sizeof(int);
    s->arena = arena_alloc_ex(bytes, &s->arena_bytes, &s->arena_mmap, &s->arena_hugetlb);
    if (!s->arena)
        return -1;
    float *p = (float *)s->arena;
    s->x = p;
    p += D;
    s->xb = p;
    p += D;
    s->xb2 = p;
    p += D;
    s->qkv = p;
    p += (size_t)(D + 2 * kv_dim);
    s->att = p;
    p += (size_t)H * Tkv;
    s->mlp = p;
    p += 2 * mid;
    s->logits = p;
    p += V;
    s->k_cache = p;
    p += (size_t)L * Tkv * kv_dim;
    s->v_cache = p;
    p += (size_t)L * Tkv * kv_dim;
    s->tokens = (int *)p;
    s->n_tokens = 0;
    s->kv_len = 0;
    s->kv_pos0 = 0;
#if CLM_DO_MLOCK
    s->locked = 1; /* arena_alloc already mlock'd + prefaulted */
#endif

    /* L1d hot pad: single-token residual + work (x,xb,xb2,qkv,mlp). */
    size_t qkv_l1 = (size_t)(D + 2 * kv_dim);
    if (qkv_l1 < 3 * (size_t)D)
        qkv_l1 = 3 * (size_t)D;
    size_t l1_floats = (size_t)D + D + D + qkv_l1 + 2 * mid;
    s->hot_l1 = arena_alloc_ex(l1_floats * sizeof(float), &s->hot_l1_bytes, &s->hot_l1_mmap,
                               NULL);
    if (!s->hot_l1) {
        clm_state_free(s);
        return -1;
    }
    {
        float *h = (float *)s->hot_l1;
        s->x_l1 = h;
        h += D;
        s->xb_l1 = h;
        h += D;
        s->xb2_l1 = h;
        h += D;
        s->qkv_l1 = h;
        h += qkv_l1;
        s->mlp_l1 = h;
    }
#if CLM_DO_MLOCK
    s->hot_l1_locked = 1;
#endif

    /* L2 hot pad: per-head decode attention scores [H * Tkv]. */
    size_t l2_floats = (size_t)H * (size_t)Tkv;
    s->hot_l2 =
        arena_alloc_ex(l2_floats * sizeof(float), &s->hot_l2_bytes, &s->hot_l2_mmap, NULL);
    if (!s->hot_l2) {
        clm_state_free(s);
        return -1;
    }
    s->att_l2 = (float *)s->hot_l2;
#if CLM_DO_MLOCK
    s->hot_l2_locked = 1;
#endif
    return 0;
}

int clm_state_create(ClmState *s, const ClmModel *m, int token_cap) {
    return clm_state_create_kv(s, m, token_cap, m ? m->cfg.block_size : 0);
}

void clm_state_free(ClmState *s) {
    if (!s)
        return;
    arena_free_ex(s->hot_l1, s->hot_l1_bytes, s->hot_l1_mmap);
    arena_free_ex(s->hot_l2, s->hot_l2_bytes, s->hot_l2_mmap);
    arena_free_ex(s->arena, s->arena_bytes, s->arena_mmap);
    memset(s, 0, sizeof(*s));
}

size_t clm_weight_bytes(const ClmModel *m) {
    return m->weight_bytes + (size_t)m->cfg.vocab_size;
}
size_t clm_state_bytes(const ClmState *s) {
    return s->arena_bytes + s->hot_l1_bytes + s->hot_l2_bytes;
}
size_t clm_inference_ws_bytes(const ClmModel *m, const ClmState *s) {
    return clm_weight_bytes(m) + clm_state_bytes(s);
}

size_t clm_inference_ws_n(const ClmModel *m, const ClmState *s, int n_streams) {
    if (n_streams < 1)
        n_streams = 1;
    return clm_weight_bytes(m) + (size_t)n_streams * clm_state_bytes(s);
}

int clm_pin_model_and_states(ClmModel *m, ClmState *states, int n_streams) {
    int rc = 0;
    if (n_streams < 1 || !states)
        return -1;
    /* Pin model with first state, then remaining states only. */
    if (clm_pin_arenas(m, &states[0]) != 0)
        rc = -1;
    for (int i = 1; i < n_streams; i++) {
        if (clm_pin_arenas(NULL, &states[i]) != 0)
            rc = -1;
    }
    return rc;
}

void clm_forward_full(const ClmModel *m, ClmState *s) {
    s->kv_len = 0;
    clm_forward(m, s);
}

void clm_forward(const ClmModel *m, ClmState *s) {
    if (m->cfg.arch != 2) {
        fprintf(stderr, "clm_forward: only arch=2 (v2) supported\n");
        return;
    }
    if (s->n_tokens < 1)
        return;
    extern void clm_forward_v2(const ClmModel *m, ClmState *s);
    clm_forward_v2(m, s);
}

/* ---- sample ---- */
static uint64_t rng_next(uint64_t *s) {
    uint64_t z = (*s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static void softmax_inplace(float *x, int n) {
    float m = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > m)
            m = x[i];
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);
        sum += x[i];
    }
    float inv = 1.f / sum;
    for (int i = 0; i < n; i++)
        x[i] *= inv;
}

int clm_sample(const ClmModel *m, ClmState *s, float temperature, int top_k,
               uint64_t *rng) {
    int V = m->cfg.vocab_size;
    float *logits = s->logits;

    if (temperature <= 0.f) {
        int best = 0;
        float bv = logits[0];
        for (int i = 1; i < V; i++)
            if (logits[i] > bv) {
                bv = logits[i];
                best = i;
            }
        if (s->n_tokens < s->token_cap)
            s->tokens[s->n_tokens++] = best;
        return best;
    }

    float *tmp = s->mlp;
    for (int i = 0; i < V; i++)
        tmp[i] = logits[i] / temperature;

    if (top_k > 0 && top_k < V) {
        float thr = tmp[0];
        float *copy = s->mlp + V;
        for (int i = 0; i < V; i++)
            copy[i] = tmp[i];
        for (int i = 0; i < top_k; i++) {
            int bi = i;
            for (int j = i + 1; j < V; j++)
                if (copy[j] > copy[bi])
                    bi = j;
            float sw = copy[i];
            copy[i] = copy[bi];
            copy[bi] = sw;
        }
        thr = copy[top_k - 1];
        for (int i = 0; i < V; i++)
            if (tmp[i] < thr)
                tmp[i] = -1e10f;
    }
    softmax_inplace(tmp, V);
    double r = (rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
    double cdf = 0.0;
    int pick = V - 1;
    for (int i = 0; i < V; i++) {
        cdf += tmp[i];
        if (r < cdf) {
            pick = i;
            break;
        }
    }
    if (s->n_tokens < s->token_cap)
        s->tokens[s->n_tokens++] = pick;
    return pick;
}

int clm_set_tokens(const ClmModel *m, ClmState *s, const int *ids, int n) {
    if (!s || !ids || n < 0)
        return -1;
    s->n_tokens = 0;
    s->kv_len = 0;
    s->kv_pos0 = 0;
    for (int i = 0; i < n && s->n_tokens < s->token_cap; i++) {
        int id = ids[i];
        if (id < 0 || id >= m->cfg.vocab_size)
            id = 0;
        s->tokens[s->n_tokens++] = id;
    }
    return s->n_tokens;
}

/* Char encode/decode removed with CLM1; BPE lives in Python. */
int CLM_COLD clm_encode(const ClmModel *m, ClmState *s, const char *prompt) {
    (void)m;
    (void)s;
    (void)prompt;
    fprintf(stderr, "clm_encode removed; use clm_set_tokens (Python BPE)\n");
    return -1;
}

int clm_decode(const ClmModel *m, const int *ids, int n, char *buf, int buf_cap) {
    (void)m;
    (void)ids;
    (void)n;
    if (buf_cap > 0)
        buf[0] = 0;
    return 0;
}

/* ---- L3 residency probe (mincore) + PMU cache stats ---- */
/* ---- L3 residency probe (mincore) + PMU cache stats ---- */

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

static size_t clm_pagesize(void) {
#if defined(__linux__)
    long p = sysconf(_SC_PAGESIZE);
    return p > 0 ? (size_t)p : 4096u;
#else
    return 4096u;
#endif
}

/* Count resident bytes of [addr, addr+bytes) via mincore. */
static void mincore_resident(const void *addr, size_t bytes, size_t *out_res_bytes,
                             size_t *out_pages, size_t *out_pages_res) {
    *out_res_bytes = 0;
    *out_pages = 0;
    *out_pages_res = 0;
    if (!addr || bytes == 0)
        return;
#if defined(__linux__)
    size_t ps = clm_pagesize();
    uintptr_t a = (uintptr_t)addr;
    uintptr_t start = a & ~(uintptr_t)(ps - 1);
    size_t off = (size_t)(a - start);
    size_t span = off + bytes;
    size_t np = (span + ps - 1) / ps;
    unsigned char *vec = (unsigned char *)malloc(np);
    if (!vec)
        return;
    if (mincore((void *)start, np * ps, vec) != 0) {
        free(vec);
        return;
    }
    size_t res_pages = 0;
    for (size_t i = 0; i < np; i++) {
        if (vec[i] & 1)
            res_pages++;
    }
    free(vec);
    *out_pages = np;
    *out_pages_res = res_pages;
    /* approximate resident bytes of the logical range */
    if (res_pages >= np)
        *out_res_bytes = bytes;
    else
        *out_res_bytes = res_pages * ps > bytes ? bytes : res_pages * ps;
#else
    (void)addr;
    (void)bytes;
#endif
}

void clm_residency_probe(const ClmModel *m, const ClmState *s, ClmResidencyReport *r) {
    memset(r, 0, sizeof(*r));
    if (!m || !s)
        return;
    r->weight_payload = clm_weight_bytes(m);
    r->model_arena = m->arena_bytes;
    r->state_arena = s->arena_bytes;
    r->hot_l1_bytes = s->hot_l1_bytes;
    r->hot_l2_bytes = s->hot_l2_bytes;
    r->ws = clm_inference_ws_bytes(m, s);
    r->l1d = clm_l1d_bytes();
    r->l1i = clm_l1i_bytes();
    r->l2 = clm_l2_bytes();
    r->l3_ccd = clm_l3_ccd_bytes();
    r->text_bytes = clm_self_text_bytes();
    r->model_locked = m->locked;
    r->state_locked = s->locked;
    r->hot_l1_locked = s->hot_l1_locked;
    r->hot_l2_locked = s->hot_l2_locked;
    mincore_resident(m->arena, m->arena_bytes, &r->model_resident, &r->model_pages,
                     &r->model_pages_res);
    mincore_resident(s->arena, s->arena_bytes, &r->state_resident, &r->state_pages,
                     &r->state_pages_res);
    mincore_resident(s->hot_l1, s->hot_l1_bytes, &r->hot_l1_resident, &r->hot_l1_pages,
                     &r->hot_l1_pages_res);
    mincore_resident(s->hot_l2, s->hot_l2_bytes, &r->hot_l2_resident, &r->hot_l2_pages,
                     &r->hot_l2_pages_res);
}

void clm_print_residency_report(const ClmResidencyReport *r) {
    if (!r)
        return;
    double kib = 1024.0;
    double mib = 1024.0 * 1024.0;

    printf("--- cache hierarchy plan ---\n");
    printf("  tier     target size        what lives there\n");
    if (r->l1i)
        printf("  L1i      %7.1f KiB         hot kernels (.text.hot GEMV/decode)\n",
               r->l1i / kib);
    if (r->l1d)
        printf("  L1d      %7.1f KiB         decode residual + work vectors\n",
               r->l1d / kib);
    if (r->l2)
        printf("  L2       %7.1f KiB         attn scores + streaming weight tiles\n",
               r->l2 / kib);
    if (r->l3_ccd)
        printf("  L3 CCD   %7.1f MiB         full weights + KV + prefill scratch\n",
               r->l3_ccd / mib);

    printf("--- L1 / L2 hot pads (decode) ---\n");
    printf("  L1d pad:            %8.1f KiB  (x/xb/qkv/mlp one-token)  fill vs L1d ",
           r->hot_l1_bytes / kib);
    if (r->l1d)
        printf("%.1f%%  %s\n", 100.0 * (double)r->hot_l1_bytes / (double)r->l1d,
               r->hot_l1_bytes <= r->l1d ? "FITS" : "OVER");
    else
        printf("?\n");
    printf("  L2  pad:            %8.1f KiB  (attn scores H×T)         fill vs L2  ",
           r->hot_l2_bytes / kib);
    if (r->l2)
        printf("%.2f%%  %s\n", 100.0 * (double)r->hot_l2_bytes / (double)r->l2,
               r->hot_l2_bytes <= r->l2 ? "FITS" : "OVER");
    else
        printf("?\n");
    if (r->text_bytes && r->l1i) {
        printf("  binary image:       %8.1f KiB  (want hot path in L1i %7.1f KiB)\n",
               r->text_bytes / kib, r->l1i / kib);
        printf("  note: only .text.hot kernels are expected to stay L1i-resident\n");
    }

    printf("--- L3 occupancy (full working set) ---\n");
    printf("  weights (payload):  %8.3f MiB\n", r->weight_payload / mib);
    printf("  model arena:        %8.3f MiB  (padded alloc)\n", r->model_arena / mib);
    printf("  state+KV arena:     %8.3f MiB\n", r->state_arena / mib);
    printf("  hot L1+L2 pads:     %8.3f MiB\n",
           (r->hot_l1_bytes + r->hot_l2_bytes) / mib);
    printf("  infer WS total:     %8.3f MiB\n", r->ws / mib);
    if (r->l3_ccd) {
        double fill = 100.0 * (double)r->ws / (double)r->l3_ccd;
        double head = (r->l3_ccd > r->ws) ? (r->l3_ccd - r->ws) / mib : 0.0;
        printf("  L3 fill (WS/CCD):   %7.2f%%  headroom %.3f MiB\n", fill, head);
        printf("  model in L3 budget: %s\n", r->ws <= r->l3_ccd ? "YES (fits)" : "NO (oversize)");
    }

    printf("--- RAM page residency (mincore; mlock target = 100%%) ---\n");
    printf("  mlock: model=%s state=%s L1pad=%s L2pad=%s\n",
           r->model_locked ? "yes" : "no", r->state_locked ? "yes" : "no",
           r->hot_l1_locked ? "yes" : "no", r->hot_l2_locked ? "yes" : "no");
    if (r->hot_l1_pages)
        printf("  L1d pad pages: %zu / %zu resident (%.0f%%)\n", r->hot_l1_pages_res,
               r->hot_l1_pages,
               100.0 * (double)r->hot_l1_pages_res / (double)r->hot_l1_pages);
    if (r->hot_l2_pages)
        printf("  L2  pad pages: %zu / %zu resident (%.0f%%)\n", r->hot_l2_pages_res,
               r->hot_l2_pages,
               100.0 * (double)r->hot_l2_pages_res / (double)r->hot_l2_pages);
    if (r->model_pages)
        printf("  model pages:   %zu / %zu resident (%.1f%%)  ~%.3f MiB\n",
               r->model_pages_res, r->model_pages,
               100.0 * (double)r->model_pages_res / (double)r->model_pages,
               r->model_resident / mib);
    if (r->state_pages)
        printf("  state pages:   %zu / %zu resident (%.1f%%)  ~%.3f MiB\n",
               r->state_pages_res, r->state_pages,
               100.0 * (double)r->state_pages_res / (double)r->state_pages,
               r->state_resident / mib);
}

#if defined(__linux__)
static long clm_sys_perf_event_open(struct perf_event_attr *attr, pid_t pid, int cpu,
                                    int group_fd, unsigned long flags) {
    return syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags);
}

static int clm_open_perf(uint32_t type, uint64_t config) {
    struct perf_event_attr a;
    memset(&a, 0, sizeof a);
    a.type = type;
    a.size = sizeof a;
    a.config = config;
    a.disabled = 1;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.exclude_idle = 1;
    int fd = (int)clm_sys_perf_event_open(&a, 0, -1, -1, 0);
    return fd;
}

static int clm_open_hw(uint64_t config) {
    return clm_open_perf(PERF_TYPE_HARDWARE, config);
}

static int clm_open_hw_cache(uint64_t cache_id, uint64_t op, uint64_t result) {
    uint64_t cfg = cache_id | (op << 8) | (result << 16);
    return clm_open_perf(PERF_TYPE_HW_CACHE, cfg);
}

static int clm_open_raw(uint64_t config) {
    return clm_open_perf(PERF_TYPE_RAW, config);
}

static uint64_t clm_read_fd(int fd) {
    if (fd < 0)
        return 0;
    uint64_t v = 0;
    if (read(fd, &v, sizeof v) != (ssize_t)sizeof v)
        return 0;
    return v;
}

static void clm_close_fd(int *fd) {
    if (fd && *fd >= 0) {
        close(*fd);
        *fd = -1;
    }
}
#endif

int clm_cache_monitor_start(ClmCacheMonitor *mon) {
    memset(mon, 0, sizeof(*mon));
    mon->fd_cycles = mon->fd_instr = mon->fd_cache_ref = mon->fd_cache_miss = -1;
    mon->fd_l1d_load = mon->fd_l1d_miss = -1;
    mon->fd_l2_local_l3 = mon->fd_l2_dram_near = mon->fd_l2_dram_far = -1;
    mon->fd_l2_near_cache = mon->fd_l2_far_cache = mon->fd_l2_all = -1;
#if !defined(__linux__)
    return -1;
#else
    mon->fd_cycles = clm_open_hw(PERF_COUNT_HW_CPU_CYCLES);
    mon->fd_instr = clm_open_hw(PERF_COUNT_HW_INSTRUCTIONS);
    mon->fd_cache_ref = clm_open_hw(PERF_COUNT_HW_CACHE_REFERENCES);
    mon->fd_cache_miss = clm_open_hw(PERF_COUNT_HW_CACHE_MISSES);
    mon->fd_l1d_load = clm_open_hw_cache(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,
                                          PERF_COUNT_HW_CACHE_RESULT_ACCESS);
    mon->fd_l1d_miss = clm_open_hw_cache(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,
                                         PERF_COUNT_HW_CACHE_RESULT_MISS);
    /* AMD Zen4/5 encodings from perf (strace): l2_fill_rsp_src.* */
    mon->fd_l2_local_l3 = clm_open_raw(0x100000265ull);   /* local_ccx = this L3 */
    mon->fd_l2_dram_near = clm_open_raw(0x100000865ull);  /* dram_io_near */
    mon->fd_l2_dram_far = clm_open_raw(0x100004065ull);   /* dram_io_far */
    mon->fd_l2_near_cache = clm_open_raw(0x100000465ull); /* near_cache */
    mon->fd_l2_far_cache = clm_open_raw(0x100001065ull);  /* far_cache */
    mon->fd_l2_all = clm_open_raw(0x10000de65ull);        /* all fills */

    int *fds[] = {&mon->fd_cycles,       &mon->fd_instr,        &mon->fd_cache_ref,
                  &mon->fd_cache_miss,   &mon->fd_l1d_load,     &mon->fd_l1d_miss,
                  &mon->fd_l2_local_l3,  &mon->fd_l2_dram_near, &mon->fd_l2_dram_far,
                  &mon->fd_l2_near_cache, &mon->fd_l2_far_cache, &mon->fd_l2_all};
    int n_ok = 0;
    for (size_t i = 0; i < sizeof fds / sizeof fds[0]; i++) {
        if (*fds[i] >= 0) {
            ioctl(*fds[i], PERF_EVENT_IOC_RESET, 0);
            ioctl(*fds[i], PERF_EVENT_IOC_ENABLE, 0);
            n_ok++;
        }
    }
    mon->active = n_ok > 0;
    return mon->active ? 0 : -1;
#endif
}

int clm_cache_monitor_stop(ClmCacheMonitor *mon, ClmCacheStats *out) {
    memset(out, 0, sizeof(*out));
    if (!mon)
        return -1;
#if !defined(__linux__)
    return -1;
#else
    if (!mon->active)
        return -1;

    int *fds[] = {&mon->fd_cycles,       &mon->fd_instr,        &mon->fd_cache_ref,
                  &mon->fd_cache_miss,   &mon->fd_l1d_load,     &mon->fd_l1d_miss,
                  &mon->fd_l2_local_l3,  &mon->fd_l2_dram_near, &mon->fd_l2_dram_far,
                  &mon->fd_l2_near_cache, &mon->fd_l2_far_cache, &mon->fd_l2_all};
    for (size_t i = 0; i < sizeof fds / sizeof fds[0]; i++) {
        if (*fds[i] >= 0)
            ioctl(*fds[i], PERF_EVENT_IOC_DISABLE, 0);
    }

    out->cycles = clm_read_fd(mon->fd_cycles);
    out->instructions = clm_read_fd(mon->fd_instr);
    out->cache_refs = clm_read_fd(mon->fd_cache_ref);
    out->cache_misses = clm_read_fd(mon->fd_cache_miss);
    out->l1d_loads = clm_read_fd(mon->fd_l1d_load);
    out->l1d_misses = clm_read_fd(mon->fd_l1d_miss);
    out->l2_fill_local_l3 = clm_read_fd(mon->fd_l2_local_l3);
    out->l2_fill_dram_near = clm_read_fd(mon->fd_l2_dram_near);
    out->l2_fill_dram_far = clm_read_fd(mon->fd_l2_dram_far);
    out->l2_fill_near_cache = clm_read_fd(mon->fd_l2_near_cache);
    out->l2_fill_far_cache = clm_read_fd(mon->fd_l2_far_cache);
    out->l2_fill_all = clm_read_fd(mon->fd_l2_all);

    out->have_cache = mon->fd_cache_ref >= 0 && mon->fd_cache_miss >= 0;
    out->have_l1d = mon->fd_l1d_load >= 0 && mon->fd_l1d_miss >= 0;
    out->have_l2_fills =
        mon->fd_l2_local_l3 >= 0 || mon->fd_l2_dram_near >= 0 || mon->fd_l2_all >= 0;
    out->ok = 1;

    for (size_t i = 0; i < sizeof fds / sizeof fds[0]; i++)
        clm_close_fd(fds[i]);
    mon->active = 0;
    return 0;
#endif
}

void clm_print_cache_stats(const ClmCacheStats *s) {
    if (!s || !s->ok) {
        printf("--- cache PMU: unavailable (need Linux perf_event; check "
               "kernel.perf_event_paranoid) ---\n");
        return;
    }
    printf("--- cache PMU (decode window, user-space) ---\n");
    if (s->cycles)
        printf("  cycles:              %llu\n", (unsigned long long)s->cycles);
    if (s->instructions)
        printf("  instructions:        %llu\n", (unsigned long long)s->instructions);
    if (s->cycles && s->instructions)
        printf("  IPC:                 %.2f\n",
               (double)s->instructions / (double)s->cycles);

    if (s->have_cache) {
        double miss = s->cache_refs
                          ? 100.0 * (double)s->cache_misses / (double)s->cache_refs
                          : 0.0;
        double hit = 100.0 - miss;
        printf("  cache-references:    %llu\n", (unsigned long long)s->cache_refs);
        printf("  cache-misses:        %llu\n", (unsigned long long)s->cache_misses);
        printf("  cache hit rate:      %.2f%%   miss rate %.2f%%\n", hit, miss);
        printf("  (HW cache-refs ≈ last-level-ish; high L1 miss is normal for GEMV)\n");
    }

    if (s->have_l1d) {
        double l1m = s->l1d_loads
                         ? 100.0 * (double)s->l1d_misses / (double)s->l1d_loads
                         : 0.0;
        printf("  L1d loads:           %llu\n", (unsigned long long)s->l1d_loads);
        printf("  L1d load-misses:     %llu  (%.2f%% miss)\n",
               (unsigned long long)s->l1d_misses, l1m);
    }

    if (s->have_l2_fills) {
        uint64_t dram = s->l2_fill_dram_near + s->l2_fill_dram_far;
        uint64_t onchip = s->l2_fill_local_l3 + s->l2_fill_near_cache + s->l2_fill_far_cache;
        uint64_t parts = onchip + dram;
        uint64_t total = s->l2_fill_all;
        /* Use the larger of "all" vs sum of sources so rates stay ≤100%. */
        if (total < parts)
            total = parts;
        if (total == 0)
            total = parts;
        printf("  L2 fill sources (where L2 misses are satisfied):\n");
        if (s->l2_fill_local_l3 || total) {
            double p = total ? 100.0 * (double)s->l2_fill_local_l3 / (double)total : 0.0;
            printf("    local L3 (this CCD): %llu  (%.2f%%)  ← want high\n",
                   (unsigned long long)s->l2_fill_local_l3, p);
        }
        if (s->l2_fill_near_cache || s->l2_fill_far_cache) {
            double pn = total ? 100.0 * (double)s->l2_fill_near_cache / (double)total : 0.0;
            double pf = total ? 100.0 * (double)s->l2_fill_far_cache / (double)total : 0.0;
            printf("    near cache:          %llu  (%.2f%%)\n",
                   (unsigned long long)s->l2_fill_near_cache, pn);
            printf("    far cache:           %llu  (%.2f%%)\n",
                   (unsigned long long)s->l2_fill_far_cache, pf);
        }
        {
            double pd = total ? 100.0 * (double)dram / (double)total : 0.0;
            printf("    DRAM (near+far):     %llu  (%.2f%%)  ← want ~0 after warm\n",
                   (unsigned long long)dram, pd);
            if (pd <= 5.0 && total > 0)
                printf("  L3 residency (fill source): GOOD  (DRAM share ≤ 5%%)\n");
            else if (total > 0)
                printf("  L3 residency (fill source): WEAK  (DRAM share %.2f%%)\n", pd);
        }
    } else {
        printf("  L2 fill sources:     n/a (AMD PMU events not available)\n");
    }
}
