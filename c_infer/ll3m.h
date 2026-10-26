/* Bare-metal ll3m inference: weights+acts sized for L3 residency.
 * Pin/prefault APIs keep arenas off the page-fault / reclaim path.
 */
#ifndef CACHELM_H
#define CACHELM_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    int vocab_size;
    int block_size;
    int n_layer;
    int n_head;
    int n_embd;
    int head_dim; /* n_embd / n_head */
    int arch; /* always 2 = Qwen-dense CLM2 */
    int n_kv_head;
    int intermediate;
    float rope_theta;
    int has_ls;   /* LayerScale present */
    int has_peri; /* peri-LN present */
} ClmConfig;

/* One loaded model: weights in a single arena (mlock'd by default). */
typedef struct {
    ClmConfig cfg;
    unsigned char *vocab; /* unused (legacy field) */
    float *w;             /* packed runtime weights */
    size_t n_weight;
    size_t weight_bytes; /* post-quant payload (excludes arena page pad) */
    void *arena;
    size_t arena_bytes;
    int arena_mmap; /* 1 = munmap (hugetlb/anon), 0 = free() */
    int arena_hugetlb; /* 1 = MAP_HUGETLB 2MiB pages */
    int quantized; /* 0=f32 v2, 1=int8, 2=ternary L3T2, 3=binary L3B2 research */
    int locked;    /* arenas mlock'd */
    /* CLM2: tokenizer.json blob (Python also uses sidecar) */
    unsigned char *tok_json;
    size_t tok_json_len;
    /* Cached ternary weight map (opaque WV2t*); built once at load. */
    void *map_cache;
    int map_cache_ready;
} ClmModel;

/*
 * Scratch + KV cache with explicit cache tiers for decode:
 *
 *   L1d hot pad  — single-token residual + work vectors (x, xb, qkv, mlp)
 *   L2  hot pad  — attention scores for decode (reused every head/layer)
 *   L3  main     — full prefill buffers, logits, K/V cache, tokens
 *
 * Weights live in the model arena (L3-resident product claim). GEMV keeps the
 * activation vector in L1 and streams weight tiles through L2 (prefetch T1).
 * Kernels are marked `hot` so .text.hot stays in L1i.
 */
typedef struct {
    /* Prefill / shared (L3 arena) */
    float *x;
    float *xb;
    float *xb2;
    float *qkv;
    float *att;
    float *mlp;
    float *logits;
    float *k_cache;
    float *v_cache;
    int *tokens;
    int n_tokens;
    int token_cap;
    int kv_cap; /* KV length capacity (<= block_size); 0 → block_size */
    int kv_len;
    int kv_pos0;
    void *arena;
    size_t arena_bytes;
    int arena_mmap;
    int arena_hugetlb;
    int locked;

    /* Decode L1d hot pad (mlock'd, separate from big arena) */
    void *hot_l1;
    size_t hot_l1_bytes;
    int hot_l1_mmap;
    float *x_l1;
    float *xb_l1;
    float *xb2_l1;
    float *qkv_l1;
    float *mlp_l1;
    int hot_l1_locked;

    /* Decode L2 hot pad (scores; mlock'd) */
    void *hot_l2;
    size_t hot_l2_bytes;
    int hot_l2_mmap;
    float *att_l2; /* [n_head * block_size] scores for one decode step */
    int hot_l2_locked;
} ClmState;

/* Load L3T2/L3M2 (or legacy CLT2/CLM2) from a filesystem path. */
int clm_model_load(ClmModel *m, const char *path);
/* Load from an in-memory pack blob (same bytes as a .ll3m file). label is for logs. */
int clm_model_load_mem(ClmModel *m, const void *data, size_t nbytes, const char *label);
void clm_model_free(ClmModel *m);

/* Optional secondary build: weights linked into the binary (see Makefile embed). */
int clm_has_embedded_weights(void);
const void *clm_embedded_weights_ptr(void);
size_t clm_embedded_weights_size(void);
/* Convenience: load embedded pack if present; returns -1 if not built with embed. */
int clm_model_load_embedded(ClmModel *m);

int clm_state_create(ClmState *s, const ClmModel *m, int token_cap);
/* Same as create, but KV cache length = kv_cap (clamped to [1, block_size]). */
int clm_state_create_kv(ClmState *s, const ClmModel *m, int token_cap, int kv_cap);
void clm_state_free(ClmState *s);

void clm_forward(const ClmModel *m, ClmState *s);
void clm_forward_full(const ClmModel *m, ClmState *s);

int clm_sample(const ClmModel *m, ClmState *s, float temperature, int top_k,
               uint64_t *rng);

int clm_encode(const ClmModel *m, ClmState *s, const char *prompt);
int clm_decode(const ClmModel *m, const int *ids, int n, char *buf, int buf_cap);
/* Set prompt from pre-tokenized ids (Python BPE encodes; C runs the model). */
int clm_set_tokens(const ClmModel *m, ClmState *s, const int *ids, int n);

size_t clm_weight_bytes(const ClmModel *m);
size_t clm_state_bytes(const ClmState *s);
size_t clm_inference_ws_bytes(const ClmModel *m, const ClmState *s);
/* Shared weights + n independent states (multi-stream on one CCD). */
size_t clm_inference_ws_n(const ClmModel *m, const ClmState *s, int n_streams);
/* Pin model once + each state arena (for multi-stream). */
int clm_pin_model_and_states(ClmModel *m, ClmState *states, int n_streams);

/* ---- L3 residency helpers ---- */

/* Raise RLIMIT_MEMLOCK (best-effort) then mlock model+state arenas. */
int clm_pin_arenas(ClmModel *m, ClmState *s);

/* Touch every cache line so pages are resident before timed work. */
void clm_prefault(const ClmModel *m, const ClmState *s);

/*
 * Hierarchical touch matching desired residency:
 *   L1d — dense read/write of hot_l1 pad
 *   L2  — dense touch of hot_l2 pad + streaming weight tiles
 *   L3  — sequential 64B line stream of model + state arenas
 */
void clm_touch_hierarchy(const ClmModel *m, const ClmState *s);

/*
 * Run several full forwards so L3 holds hot working set.
 * Call after encode (and preferably after pin+prefault).
 */
void clm_warm(const ClmModel *m, ClmState *s, int passes);

/*
 * Bind process to one L3 CCD's CPUs so the working set does not bounce
 * between two 32 MiB slices. Returns 0 on success, -1 if unavailable.
 * ccd_index: 0 = first CCD, 1 = second, ...
 * single_core: if 1, pin to the first CPU of that CCD only (private L2).
 */
int clm_bind_l3_ccd(int ccd_index);
int clm_bind_l3_ccd_ex(int ccd_index, int single_core);
/*
 * Prefer kernel-isolated CPUs (/sys/devices/system/cpu/isolated) if any,
 * else preferred_cpu if it belongs to the CCD, else a quiet physical core
 * of that CCD (not the first logical CPU — often the busiest).
 * preferred_cpu < 0 means auto. single_core pins one CPU. Returns 0 on success.
 */
int clm_bind_residency_cpu(int ccd_index, int single_core, int preferred_cpu,
                           int *out_cpu);

/* 1 if kernel has isolcpus / isolated list non-empty. */
int clm_cpu_isolation_present(void);

/*
 * Hard residency mode (best-effort; prints nothing — caller reports):
 *   mlockall(CURRENT|FUTURE), pin arenas, hugepage hints, SCHED_FIFO,
 *   single-core CCD bind, hierarchical touch.
 * preferred_cpu < 0 → auto (isolcpus / quiet CCD core / CLM_PIN_CPU env).
 * Returns 0 if core steps ok; negative bitmask of soft failures.
 */
typedef struct {
    int mlockall_ok;
    int pin_ok;
    int fifo_ok;
    int bind_ok;
    int hugepage_ok;     /* madvise HUGEPAGE or MAP_HUGETLB used */
    int hugetlb_mmap_ok; /* model arena used MAP_HUGETLB 2MiB */
    int isol_present;    /* kernel isolcpus / isolated list */
    int bound_cpu;       /* cpu id we pinned to, or -1 */
    int nice_prio;       /* FIFO priority used, or -1 */
} ClmHardResidency;

int clm_residency_hard_enable(ClmModel *m, ClmState *s, int ccd_index,
                              int preferred_cpu, ClmHardResidency *out);

/* Sysfs size of one L3 instance in bytes (0 if unknown). */
size_t clm_l3_ccd_bytes(void);

/* 1 if infer WS fits one CCD L3 (or unknown size treated as fail-open only if ws=0). */
int clm_fits_l3_ccd(const ClmModel *m, const ClmState *s);

/* ---- Residency + hardware cache stats ---- */

/* Snapshot of planned L1/L2/L3 occupancy + RAM page residency (mincore). */
typedef struct {
    size_t weight_payload; /* logical weight bytes */
    size_t model_arena;    /* full model arena (padded) */
    size_t state_arena;    /* L3: activations + KV */
    size_t hot_l1_bytes;   /* L1d decode pad */
    size_t hot_l2_bytes;   /* L2 decode pad */
    size_t ws;             /* weight_payload + state_arena + hot pads */
    size_t l1d;            /* sysfs L1d (0 if unknown) */
    size_t l1i;            /* sysfs L1i */
    size_t l2;             /* sysfs L2 */
    size_t l3_ccd;         /* one CCD L3 from sysfs (0 if unknown) */
    size_t text_bytes;     /* binary .text size (kernels; want in L1i) */
    size_t model_resident; /* model arena bytes in RAM (mincore) */
    size_t state_resident; /* state arena bytes in RAM */
    size_t hot_l1_resident;
    size_t hot_l2_resident;
    size_t model_pages;
    size_t state_pages;
    size_t model_pages_res;
    size_t state_pages_res;
    size_t hot_l1_pages;
    size_t hot_l1_pages_res;
    size_t hot_l2_pages;
    size_t hot_l2_pages_res;
    int model_locked;
    int state_locked;
    int hot_l1_locked;
    int hot_l2_locked;
} ClmResidencyReport;

/* Sysfs cache sizes (bytes); 0 if unknown. */
size_t clm_l1d_bytes(void);
size_t clm_l1i_bytes(void);
size_t clm_l2_bytes(void);

void clm_residency_probe(const ClmModel *m, const ClmState *s, ClmResidencyReport *r);
void clm_print_residency_report(const ClmResidencyReport *r);

/*
 * Hardware PMU counters around a decode window (Linux perf_event_open).
 * Generic: cycles, instructions, cache-references/misses, L1D loads/misses.
 * AMD Zen: L2 fill sources (local L3 vs DRAM vs other caches) when available.
 */
typedef struct {
    int active; /* 1 if any counter opened */
    /* fds: -1 if unavailable */
    int fd_cycles;
    int fd_instr;
    int fd_cache_ref;
    int fd_cache_miss;
    int fd_l1d_load;
    int fd_l1d_miss;
    int fd_l2_local_l3;
    int fd_l2_dram_near;
    int fd_l2_dram_far;
    int fd_l2_near_cache;
    int fd_l2_far_cache;
    int fd_l2_all;
} ClmCacheMonitor;

typedef struct {
    int ok; /* 1 if at least one counter read */
    uint64_t cycles;
    uint64_t instructions;
    uint64_t cache_refs;
    uint64_t cache_misses;
    uint64_t l1d_loads;
    uint64_t l1d_misses;
    uint64_t l2_fill_local_l3;   /* fills from this CCD's L3 */
    uint64_t l2_fill_dram_near;
    uint64_t l2_fill_dram_far;
    uint64_t l2_fill_near_cache;
    uint64_t l2_fill_far_cache;
    uint64_t l2_fill_all;
    int have_cache;   /* cache-ref/miss present */
    int have_l1d;
    int have_l2_fills; /* AMD fill sources present */
} ClmCacheStats;

/* Open + enable counters (exclude kernel/HV). Returns 0 if any counter ok. */
int clm_cache_monitor_start(ClmCacheMonitor *mon);
/* Disable, read, close. Fills *out; returns 0 on success. */
int clm_cache_monitor_stop(ClmCacheMonitor *mon, ClmCacheStats *out);
void clm_print_cache_stats(const ClmCacheStats *s);

#endif
