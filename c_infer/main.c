/*
 * LL3M C inference CLI
 *
 *   ./ll3m_c -m model.ll3m --token-ids 1,2,3 -n 200 --l3-resident --greedy
 *   ./ll3m_c -m model.ll3m --token-ids ... -n 200 --residency-hard --greedy
 *
 * --l3-resident: CCD bind, mlock arenas, prefault, warm, refuse if WS > CCD.
 * --residency-hard: implies l3-resident + mlockall + single-core pin (private L2)
 *   + SCHED_FIFO + hugepage hints + hierarchical L1→L2→L3 touch patterns.
 *
 * Access pattern intent:
 *   L1i  hot kernels (.text.hot)
 *   L1d  one-token residual / work pad (dense R+W)
 *   L2   scores + prefetched weight tiles (T1 prefetch)
 *   L3   full weights + KV (sequential 64B line stream)
 */
#define _POSIX_C_SOURCE 200809L
#include "ll3m.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(__linux__)
#include <unistd.h>
#endif

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void usage(const char *argv0) {
    fprintf(stderr,
            "usage: %s -m model.ll3m [-p prompt] [-n tokens] [-t temp] [-k topk]\n"
            "          [--seed N] [--bench] [--greedy]\n"
            "          [--l3-resident] [--residency-hard] [--ccd N] [--cpu N]\n"
            "          [--warm N] [--steady-warm N] [--no-cache-stats]\n"
            "          [--stream] [--quiet] [--token-ids ID,ID,...]\n"
            "\n"
            "Secondary builds can embed the pack in the binary:\n"
            "  make -C c_infer embed EMBED_MODEL=path/to/model.ll3m\n"
            "  make -C c_infer static-embed EMBED_MODEL=...\n"
            "Then omit -m, or pass -m :embedded:\n",
            argv0);
}

int main(int argc, char **argv) {
    const char *model_path = NULL;
    const char *prompt = "ROMEO:\n";
    int n_new = 200;
    float temp = 0.8f;
    int top_k = 40;
    uint64_t seed = 42;
    int bench = 0;
    int greedy = 0;
    int l3_resident = 0;
    int residency_hard = 0;
    int ccd = 0;
    int pin_cpu = -1;       /* --cpu N or CLM_PIN_CPU; -1 = auto quiet core */
    int warm_passes = -1;   /* full-forward warm */
    int steady_warm = -1;   /* decode-only steps before PMU window */
    int cache_stats = 1;
    int stream = 0;
    int quiet = 0;
    const char *token_ids = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)
            model_path = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc)
            prompt = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc)
            n_new = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc)
            temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-k") && i + 1 < argc)
            top_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = (uint64_t)strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--bench"))
            bench = 1;
        else if (!strcmp(argv[i], "--greedy"))
            greedy = 1;
        else if (!strcmp(argv[i], "--l3-resident"))
            l3_resident = 1;
        else if (!strcmp(argv[i], "--residency-hard") || !strcmp(argv[i], "--hard"))
            residency_hard = 1;
        else if (!strcmp(argv[i], "--ccd") && i + 1 < argc)
            ccd = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu") && i + 1 < argc)
            pin_cpu = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warm") && i + 1 < argc)
            warm_passes = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--steady-warm") && i + 1 < argc)
            steady_warm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-cache-stats"))
            cache_stats = 0;
        else if (!strcmp(argv[i], "--cache-stats"))
            cache_stats = 1;
        else if (!strcmp(argv[i], "--stream"))
            stream = 1;
        else if (!strcmp(argv[i], "--quiet"))
            quiet = 1;
        else if (!strcmp(argv[i], "--token-ids") && i + 1 < argc)
            token_ids = argv[++i];
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "unknown arg: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (!model_path) {
        if (clm_has_embedded_weights()) {
            model_path = ":embedded:";
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (greedy)
        temp = 0.f;
    if (residency_hard)
        l3_resident = 1;
    if (quiet) {
        cache_stats = 0;
        bench = 0;
    }
    if (warm_passes < 0)
        warm_passes = residency_hard ? 24 : (l3_resident ? 12 : 3);
    if (steady_warm < 0)
        steady_warm = residency_hard ? 64 : (l3_resident ? 16 : 0);

    /* Bind early so allocations + run stay on isolated core or one CCD core. */
    int ccd_ok = -1;
    int early_cpu = -1;
    if (l3_resident || residency_hard) {
        ccd_ok = clm_bind_residency_cpu(ccd, /*single_core=*/residency_hard ? 1 : 0,
                                        pin_cpu, &early_cpu);
    }

    ClmModel model;
    if (clm_model_load(&model, model_path) != 0)
        return 1;

    int cap = model.cfg.block_size + n_new + 8;
    ClmState st;
    if (clm_state_create(&st, &model, cap) != 0) {
        clm_model_free(&model);
        return 1;
    }

    ClmHardResidency hard;
    memset(&hard, 0, sizeof hard);
    int hard_rc = 0;
    int pin_rc = 0;
    if (residency_hard) {
        hard_rc = clm_residency_hard_enable(&model, &st, ccd, pin_cpu, &hard);
        pin_rc = hard.pin_ok ? 0 : -1;
        ccd_ok = hard.bind_ok ? 0 : -1;
    } else {
        pin_rc = clm_pin_arenas(&model, &st);
        clm_prefault(&model, &st);
        clm_touch_hierarchy(&model, &st);
    }

    size_t ws = clm_inference_ws_bytes(&model, &st);
    size_t l3 = clm_l3_ccd_bytes();
    int fits = clm_fits_l3_ccd(&model, &st);

    if (!quiet) {
    printf("=== LL3M C inference ===\n");
    printf("config: L%d d%d h%d T%d vocab=%d\n", model.cfg.n_layer, model.cfg.n_embd,
           model.cfg.n_head, model.cfg.block_size, model.cfg.vocab_size);
    printf("weights:     %8.3f MiB\n", clm_weight_bytes(&model) / (1024.0 * 1024.0));
    printf("activations: %8.3f MiB\n", clm_state_bytes(&st) / (1024.0 * 1024.0));
    printf("infer WS:    %8.3f MiB  (target: fit in L3, no DRAM on hot path)\n",
           ws / (1024.0 * 1024.0));
    if (l3)
        printf("sysfs L3 (one CCD instance): %.1f MiB\n", l3 / (1024.0 * 1024.0));
    printf("fits one CCD L3: %s\n", fits ? "YES" : "NO");
    printf("mlock arenas: model=%s state=%s pin_rc=%d\n",
           model.locked ? "locked" : "not-locked", st.locked ? "locked" : "not-locked",
           pin_rc);
    if (l3_resident) {
        printf("L3-resident mode: CCD bind=%s (ccd=%d)  warm=%d  hard=%s\n",
               ccd_ok == 0 ? "OK" : "FAILED/skip", ccd, warm_passes,
               residency_hard ? "YES" : "no");
    }
    if (residency_hard) {
        printf("residency-hard:\n");
        printf("  mlockall:      %s\n",
               hard.mlockall_ok ? "OK" : "fail (need ulimit -l unlimited?)");
        printf("  pin arenas:    %s\n", hard.pin_ok ? "OK" : "fail");
        printf("  CPU bind:      %s", hard.bind_ok ? "OK" : "fail");
        if (hard.bound_cpu >= 0)
            printf("  cpu=%d", hard.bound_cpu);
        printf("  isolcpus=%s\n", hard.isol_present ? "present" : "absent");
        printf("  MAP_HUGETLB:   %s  (model arena 2MiB pages)\n",
               hard.hugetlb_mmap_ok || model.arena_hugetlb ? "OK" : "fallback 4K+THP");
        printf("  hugepage madv: %s\n", hard.hugepage_ok ? "OK" : "skip");
        printf("  SCHED_FIFO:    %s", hard.fifo_ok ? "OK" : "fail (CAP_SYS_NICE / chrt)");
        if (hard.fifo_ok)
            printf(" prio=%d", hard.nice_prio);
        printf("\n");
        printf("  warm:          full=%d  steady_decode=%d  (PMU only on timed window)\n",
               warm_passes, steady_warm);
        if (hard_rc)
            printf("  soft_fail_mask: 0x%x  (non-fatal; continuing)\n", hard_rc);
        printf("  access: L1 dense pad → L2 tiled GEMV/scores → L3 sequential stream\n");
        if (!hard.isol_present)
            printf("  tip: boot with isolcpus=N nohz_full=N for cleaner DRAM%% (see "
                   "scripts/residency_host_setup.sh)\n");
        if (!model.arena_hugetlb)
            printf("  tip: reserve 2MiB pages: scripts/residency_host_setup.sh hugepages 64\n");
    } else if (model.arena_hugetlb) {
        printf("model arena: MAP_HUGETLB 2MiB pages\n");
    }
    if (model.quantized == 2)
        printf("GEMV: ternary W (2-bit packed) × f32 x  (x in L1, W tiles→L2)\n");
    else if (model.quantized == 1)
        printf("GEMV: int8 path\n");
    else
        printf("GEMV: f32 (v2 bring-up; ternary pack next)\n");
#if defined(CLM_EMBED_QUANT) && CLM_EMBED_QUANT == 1
    printf("embeds: int8 + per-row scale (compact L3)\n");
#else
    printf("embeds: f32\n");
#endif
    printf("decode: KV in L3; residual/work in L1d pad; scores in L2 pad\n");
    printf("\n");

    {
        ClmResidencyReport rep;
        clm_residency_probe(&model, &st, &rep);
        if (!quiet) { clm_print_residency_report(&rep); printf("\n"); }
    }

    } /* !quiet banner */

    if (l3_resident && !fits) {
        fprintf(stderr,
                "REFUSE: infer WS %.3f MiB exceeds one L3 CCD (%.1f MiB). "
                "Shrink model or quant harder.\n",
                ws / (1024.0 * 1024.0), l3 / (1024.0 * 1024.0));
        clm_state_free(&st);
        clm_model_free(&model);
        return 2;
    }

    if (!(token_ids && token_ids[0])) {
        fprintf(stderr, "ll3m pack requires --token-ids (BPE in Python wrapper)\n");
        (void)prompt;
        return 1;
    }
    /* comma/space-separated token ids from Python BPE */
    {
        int tmp[8192];
        int nt = 0;
        const char *p = token_ids;
        while (*p && nt < 8192) {
            while (*p == ',' || *p == ' ' || *p == '\t') p++;
            if (!*p) break;
            char *end = NULL;
            long v = strtol(p, &end, 10);
            if (end == p) break;
            tmp[nt++] = (int)v;
            p = end;
        }
        if (clm_set_tokens(&model, &st, tmp, nt) < 1) {
            fprintf(stderr, "empty token-ids\n");
            return 1;
        }
    }

    /* Hierarchical install then full-forward warm (fills L3 via real access). */
    clm_touch_hierarchy(&model, &st);
    clm_warm(&model, &st, warm_passes);
    clm_touch_hierarchy(&model, &st);

    if (!quiet) {
        ClmResidencyReport rep;
        clm_residency_probe(&model, &st, &rep);
        printf("--- after hierarchical warm ---\n");
        size_t tot_p =
            rep.model_pages + rep.state_pages + rep.hot_l1_pages + rep.hot_l2_pages;
        size_t tot_r = rep.model_pages_res + rep.state_pages_res + rep.hot_l1_pages_res +
                       rep.hot_l2_pages_res;
        if (tot_p)
            printf("  all arenas in RAM: %.1f%%  (%zu / %zu pages)\n",
                   100.0 * (double)tot_r / (double)tot_p, tot_r, tot_p);
        if (rep.hot_l1_pages)
            printf("  L1d pad resident:  %.0f%%\n",
                   100.0 * (double)rep.hot_l1_pages_res / (double)rep.hot_l1_pages);
        if (rep.hot_l2_pages)
            printf("  L2  pad resident:  %.0f%%\n",
                   100.0 * (double)rep.hot_l2_pages_res / (double)rep.hot_l2_pages);
        if (rep.l3_ccd)
            printf("  model claims %.2f%% of CCD L3 (by size)\n",
                   100.0 * (double)rep.ws / (double)rep.l3_ccd);
        printf("\n");
    }

    if (bench) {
        const int N = 50;
        double t0 = now_s();
        for (int i = 0; i < N; i++)
            clm_forward(&model, &st);
        double dt = now_s() - t0;
        printf("bench: %d forwards in %.3fs  (%.2f ms/fwd, window T=%d)\n", N, dt,
               1000.0 * dt / N,
               st.n_tokens > model.cfg.block_size ? model.cfg.block_size : st.n_tokens);
    }

    uint64_t rng = seed ? seed : 0xC0FFEEULL;

    /*
     * Steady-state decode warm: run real single-token steps WITHOUT PMU so cold
     * fill noise is not counted. Then re-touch L1/L2 and open PMU for the timed
     * window only.
     */
    if (steady_warm > 0) {
        for (int i = 0; i < steady_warm; i++) {
            clm_forward(&model, &st);
            clm_sample(&model, &st, temp, top_k, &rng);
        }
        if (!quiet)
            printf("steady-warm: %d decode steps (not in PMU window)\n\n", steady_warm);
    }

    clm_touch_hierarchy(&model, &st);

    ClmCacheMonitor mon;
    ClmCacheStats cstats;
    memset(&cstats, 0, sizeof cstats);
    int mon_ok = -1;
    if (cache_stats)
        mon_ok = clm_cache_monitor_start(&mon);

    double t0 = now_s();
    int prompt_n = st.n_tokens;
    for (int i = 0; i < n_new; i++) {
        clm_forward(&model, &st);
        int tid = clm_sample(&model, &st, temp, top_k, &rng);
        if (stream) {
            /* stream protocol: one token id per line for Python wrapper */
            printf("%d\n", tid);
            fflush(stdout);
        }
    }
    double dt = now_s() - t0;

    if (cache_stats && mon_ok == 0)
        clm_cache_monitor_stop(&mon, &cstats);

    if (!stream) {
        /* CLM2: print generated token ids (Python decodes BPE). */
        for (int i = prompt_n; i < st.n_tokens; i++)
            printf("%d%s", st.tokens[i], i + 1 < st.n_tokens ? " " : "\n");
    }

    if (!quiet) printf("\n--- %d new tokens in %.3fs (%.1f tok/s) ---\n", n_new, dt,
           n_new / (dt > 1e-9 ? dt : 1e-9));

    if (cache_stats) {
        printf("\n");
        if (mon_ok == 0)
            clm_print_cache_stats(&cstats);
        else
            clm_print_cache_stats(NULL);
    }

    if (residency_hard)
        printf("\nRESIDENCY_HARD_RUN  mask=0x%x\n", hard_rc);
    else if (l3_resident)
        printf("\nL3_RESIDENT_RUN_OK  (try --residency-hard for mlockall+FIFO+1core)\n");

    clm_state_free(&st);
    clm_model_free(&model);
    return 0;
}
