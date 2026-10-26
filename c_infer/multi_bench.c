/*
 * Multi-stream concurrent C decode on one CCD (shared L3T2 weights).
 *
 *   ./multi_bench -m model.ll3m --token-ids ... -n 256 --streams 2 --ccd 1
 *   ./multi_bench -m model.ll3m --streams 8 --kv-cap 64 --burst 16
 *
 * Product gate: weights + N * state_bytes ≤ 0.95 × one L3 CCD.
 * Default: one pthread per stream (parallel on CCD cores).
 * --burst K>0: single-thread round-robin, K tokens per stream before switch
 *   (docs/READAHEAD_DESIGN.md KV-2 / R4).
 */
#define _GNU_SOURCE
#include "ll3m.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <sched.h>
#include <unistd.h>
#endif

typedef struct {
    const ClmModel *model;
    ClmState *st;
    int n_new;
    float temp;
    int top_k;
    uint64_t seed;
    int warm;
    int steady_warm;
    int cpu; /* -1 = no extra pin */
    pthread_barrier_t *bar_ready; /* after warm/steady */
    pthread_barrier_t *bar_go;    /* start timed window together */
    double tok_s;
    int ok;
} StreamJob;

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void *stream_main(void *arg) {
    StreamJob *j = (StreamJob *)arg;
#if defined(__linux__)
    if (j->cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(j->cpu, &set);
        (void)sched_setaffinity(0, sizeof set, &set);
    }
#endif
    for (int i = 0; i < j->warm; i++)
        clm_forward(j->model, j->st);
    uint64_t rng = j->seed;
    for (int i = 0; i < j->steady_warm; i++) {
        clm_forward(j->model, j->st);
        clm_sample(j->model, j->st, j->temp, j->top_k, &rng);
    }
    /* Synchronize so wall aggregate excludes warm/steady. */
    if (j->bar_ready)
        pthread_barrier_wait(j->bar_ready);
    if (j->bar_go)
        pthread_barrier_wait(j->bar_go);
    double t0 = now_s();
    for (int i = 0; i < j->n_new; i++) {
        clm_forward(j->model, j->st);
        clm_sample(j->model, j->st, j->temp, j->top_k, &rng);
    }
    double dt = now_s() - t0;
    j->tok_s = j->n_new / (dt > 1e-9 ? dt : 1e-9);
    j->ok = 1;
    return NULL;
}

static int parse_ids(const char *token_ids, int *tmp, int maxn) {
    int nt = 0;
    const char *p = token_ids;
    while (*p && nt < maxn) {
        while (*p == ',' || *p == ' ' || *p == '\t')
            p++;
        if (!*p)
            break;
        char *end = NULL;
        long v = strtol(p, &end, 10);
        if (end == p)
            break;
        tmp[nt++] = (int)v;
        p = end;
    }
    return nt;
}

/* First physical CPUs of a CCD from sysfs (same idea as clm_bind_l3_ccd). */
static int ccd_cpu_list(int ccd_index, int *cpus, int maxn) {
#if defined(__linux__)
    char path[128];
    char groups[8][256];
    int ng = 0;
    for (int cpu = 0; cpu < 256 && ng < 8; cpu++) {
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
            strncpy(groups[ng], line, sizeof groups[0] - 1);
            groups[ng][sizeof groups[0] - 1] = 0;
            ng++;
        }
    }
    if (ccd_index < 0 || ccd_index >= ng)
        return 0;
    int n = 0;
    char list_copy[256];
    strncpy(list_copy, groups[ccd_index], sizeof list_copy - 1);
    list_copy[sizeof list_copy - 1] = 0;
    char *save = NULL;
    for (char *tok = strtok_r(list_copy, ",", &save); tok && n < maxn;
         tok = strtok_r(NULL, ",", &save)) {
        int a = 0, b = 0;
        if (sscanf(tok, "%d-%d", &a, &b) == 2) {
            /* Prefer first half of range as physical cores when SMT listed later. */
            for (int c = a; c <= b && n < maxn; c++)
                cpus[n++] = c;
        } else if (sscanf(tok, "%d", &a) == 1) {
            cpus[n++] = a;
        }
    }
    return n;
#else
    (void)ccd_index;
    (void)cpus;
    (void)maxn;
    return 0;
#endif
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *token_ids = "12,34,56,78,90,100,110,120,130,140,150,160";
    int n_new = 256;
    int streams = 2;
    int kv_cap = 0; /* 0 = full block_size */
    int ccd = 1;
    int warm = 16;
    int steady = 32;
    int pin_threads = 1;
    int burst = 0; /* 0 = pthread parallel; >0 = single-thread RR burst */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)
            model_path = argv[++i];
        else if (!strcmp(argv[i], "--token-ids") && i + 1 < argc)
            token_ids = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc)
            n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--streams") && i + 1 < argc)
            streams = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv-cap") && i + 1 < argc)
            kv_cap = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ccd") && i + 1 < argc)
            ccd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc)
            warm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steady-warm") && i + 1 < argc)
            steady = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-pin-threads"))
            pin_threads = 0;
        else if (!strcmp(argv[i], "--burst") && i + 1 < argc)
            burst = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            fprintf(stderr,
                    "usage: %s -m model.ll3m [--streams N] [--kv-cap K] [--ccd N]\n"
                    "          [-n tokens] [--token-ids ...] [--burst K]\n"
                    "  --burst K  single-thread RR: K decode steps per stream\n"
                    "             before switch (0 = pthread parallel, default)\n",
                    argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg %s\n", argv[i]);
            return 1;
        }
    }
    if (!model_path || streams < 1 || streams > 32) {
        fprintf(stderr, "need -m and --streams 1..32\n");
        return 1;
    }

    /* Bind process to whole CCD (not single core) so threads can fan out. */
    if (clm_bind_l3_ccd_ex(ccd, /*single_core=*/0) != 0)
        fprintf(stderr, "warn: CCD bind failed (ccd=%d)\n", ccd);

    ClmModel model;
    if (clm_model_load(&model, model_path) != 0)
        return 1;

    int cap = model.cfg.block_size + n_new + 8;
    ClmState *sts = (ClmState *)calloc((size_t)streams, sizeof(ClmState));
    if (!sts) {
        clm_model_free(&model);
        return 1;
    }

    int ids[8192];
    int nt = parse_ids(token_ids, ids, 8192);
    if (nt < 1) {
        fprintf(stderr, "empty token-ids\n");
        free(sts);
        clm_model_free(&model);
        return 1;
    }

    if (kv_cap <= 0)
        kv_cap = model.cfg.block_size;
    if (kv_cap > model.cfg.block_size)
        kv_cap = model.cfg.block_size;
    printf("kv_cap=%d (block_size=%d)\n", kv_cap, model.cfg.block_size);

    for (int i = 0; i < streams; i++) {
        if (clm_state_create_kv(&sts[i], &model, cap, kv_cap) != 0) {
            fprintf(stderr, "state create failed for stream %d\n", i);
            for (int j = 0; j < i; j++)
                clm_state_free(&sts[j]);
            free(sts);
            clm_model_free(&model);
            return 1;
        }
        if (clm_set_tokens(&model, &sts[i], ids, nt) < 1) {
            fprintf(stderr, "set_tokens failed stream %d\n", i);
            return 1;
        }
    }

    size_t ws1 = clm_inference_ws_bytes(&model, &sts[0]);
    size_t wsn = clm_inference_ws_n(&model, &sts[0], streams);
    size_t l3 = clm_l3_ccd_bytes();
    size_t gate = (size_t)(l3 * 0.95);
    printf("=== multi-stream bench ===\n");
    printf("streams=%d  ccd=%d  n_new=%d  burst=%d  mode=%s\n", streams, ccd, n_new,
           burst, burst > 0 ? "rr-burst" : "pthread");
    printf("WS 1-stream: %.3f MiB\n", ws1 / (1024.0 * 1024.0));
    printf("WS %d-stream: %.3f MiB  (weights+N*state)\n", streams,
           wsn / (1024.0 * 1024.0));
    if (l3)
        printf("CCD L3: %.1f MiB  gate 95%%: %.1f MiB  fit: %s\n", l3 / (1024.0 * 1024.0),
               gate / (1024.0 * 1024.0), wsn <= gate ? "YES" : "NO");
    if (l3 && wsn > gate) {
        fprintf(stderr, "refusing: multi-stream WS exceeds CCD gate\n");
        for (int i = 0; i < streams; i++)
            clm_state_free(&sts[i]);
        free(sts);
        clm_model_free(&model);
        return 2;
    }

    if (clm_pin_model_and_states(&model, sts, streams) != 0)
        fprintf(stderr, "warn: pin partial fail\n");
    for (int i = 0; i < streams; i++)
        clm_touch_hierarchy(&model, &sts[i]);

    int cpus[32];
    int ncpu = ccd_cpu_list(ccd, cpus, 32);
    /* Prefer distinct physical cores: on 9950X list is 8-15,24-31 — take first N. */
    StreamJob *jobs = (StreamJob *)calloc((size_t)streams, sizeof(StreamJob));
    if (!jobs) {
        for (int i = 0; i < streams; i++)
            clm_state_free(&sts[i]);
        free(sts);
        clm_model_free(&model);
        return 1;
    }
    for (int i = 0; i < streams; i++) {
        jobs[i].model = &model;
        jobs[i].st = &sts[i];
        jobs[i].n_new = n_new;
        jobs[i].temp = 0.f; /* greedy */
        jobs[i].top_k = 40;
        jobs[i].seed = 0xC0FFEEULL + (uint64_t)i * 0x9E3779B97F4A7C15ULL;
        jobs[i].warm = warm;
        jobs[i].steady_warm = steady;
        jobs[i].cpu = (pin_threads && ncpu > 0) ? cpus[i % ncpu] : -1;
        jobs[i].bar_ready = NULL;
        jobs[i].bar_go = NULL;
        jobs[i].tok_s = 0.0;
        jobs[i].ok = 0;
    }

    double wall = 0.0;
    double sum = 0.0;
    double agg = 0.0;

    if (burst > 0) {
        /*
         * R4: single-thread round-robin with burst K tokens per stream.
         * One core on the CCD (first listed CPU when pin_threads).
         */
        fprintf(stderr,
                "note: --burst is single-thread RR (research/fairness); "
                "product multi-agg is pthread mode (burst=0)\n");
        int pin_cpu = (pin_threads && ncpu > 0) ? cpus[0] : -1;
#if defined(__linux__)
        if (pin_cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(pin_cpu, &set);
            (void)sched_setaffinity(0, sizeof set, &set);
        }
#endif
        /* Warm / steady per stream (serial). */
        for (int i = 0; i < streams; i++) {
            for (int w = 0; w < warm; w++)
                clm_forward(&model, &sts[i]);
            uint64_t rng = jobs[i].seed;
            for (int w = 0; w < steady; w++) {
                clm_forward(&model, &sts[i]);
                clm_sample(&model, &sts[i], 0.f, 40, &rng);
            }
            jobs[i].seed = rng;
            jobs[i].cpu = pin_cpu;
        }
        int *done = (int *)calloc((size_t)streams, sizeof(int));
        if (!done) {
            free(jobs);
            for (int i = 0; i < streams; i++)
                clm_state_free(&sts[i]);
            free(sts);
            clm_model_free(&model);
            return 1;
        }
        int remaining = streams * n_new;
        double t0 = now_s();
        while (remaining > 0) {
            for (int i = 0; i < streams && remaining > 0; i++) {
                int left = n_new - done[i];
                if (left <= 0)
                    continue;
                int step = burst < left ? burst : left;
                uint64_t rng = jobs[i].seed;
                for (int k = 0; k < step; k++) {
                    clm_forward(&model, &sts[i]);
                    clm_sample(&model, &sts[i], 0.f, 40, &rng);
                }
                jobs[i].seed = rng;
                done[i] += step;
                remaining -= step;
            }
        }
        wall = now_s() - t0;
        for (int i = 0; i < streams; i++) {
            jobs[i].tok_s = n_new / (wall > 1e-9 ? wall : 1e-9);
            jobs[i].ok = (done[i] == n_new);
            sum += jobs[i].tok_s;
            printf("stream %d: %.1f tok/s  cpu=%d  ok=%d  (rr share of wall)\n", i,
                   jobs[i].tok_s, jobs[i].cpu, jobs[i].ok);
        }
        free(done);
        agg = (streams * n_new) / (wall > 1e-9 ? wall : 1e-9);
        /* In RR mode per-stream tok/s above is wall-shared (same number each).
         * Also report fair share = agg / streams. */
        printf("per-stream fair share: %.1f tok/s  (agg/N)\n", agg / streams);
    } else {
        pthread_t *ths = (pthread_t *)calloc((size_t)streams, sizeof(pthread_t));
        if (!ths) {
            free(jobs);
            for (int i = 0; i < streams; i++)
                clm_state_free(&sts[i]);
            free(sts);
            clm_model_free(&model);
            return 1;
        }
        pthread_barrier_t bar_ready, bar_go;
        pthread_barrier_init(&bar_ready, NULL, (unsigned)streams + 1); /* +main */
        pthread_barrier_init(&bar_go, NULL, (unsigned)streams + 1);
        for (int i = 0; i < streams; i++) {
            jobs[i].bar_ready = &bar_ready;
            jobs[i].bar_go = &bar_go;
        }
        for (int i = 0; i < streams; i++) {
            if (pthread_create(&ths[i], NULL, stream_main, &jobs[i]) != 0) {
                fprintf(stderr, "pthread_create failed\n");
                free(ths);
                free(jobs);
                for (int j = 0; j < streams; j++)
                    clm_state_free(&sts[j]);
                free(sts);
                clm_model_free(&model);
                return 1;
            }
        }
        /* Wait until all streams finished warm/steady, then start timed window. */
        pthread_barrier_wait(&bar_ready);
        double t0 = now_s();
        pthread_barrier_wait(&bar_go);
        for (int i = 0; i < streams; i++)
            pthread_join(ths[i], NULL);
        wall = now_s() - t0;
        pthread_barrier_destroy(&bar_ready);
        pthread_barrier_destroy(&bar_go);
        for (int i = 0; i < streams; i++) {
            printf("stream %d: %.1f tok/s  cpu=%d  ok=%d\n", i, jobs[i].tok_s, jobs[i].cpu,
                   jobs[i].ok);
            sum += jobs[i].tok_s;
        }
        agg = (streams * n_new) / (wall > 1e-9 ? wall : 1e-9);
        free(ths);
        printf("per-stream mean: %.1f tok/s\n", sum / streams);
    }

    printf("aggregate (wall): %.1f tok/s  (%.3fs for %d×%d tokens)\n", agg, wall, streams,
           n_new);
    printf("MULTI_STREAM_OK  streams=%d  fit=%s  mode=%s\n", streams,
           (!l3 || wsn <= gate) ? "YES" : "NO", burst > 0 ? "rr-burst" : "pthread");

    for (int i = 0; i < streams; i++)
        clm_state_free(&sts[i]);
    free(sts);
    free(jobs);
    clm_model_free(&model);
    return 0;
}
