/* L3M2 (f32) + L3T2 (ternary Linears + int8 embeds) v2 runtime.
 * Legacy magic CLM2/CLT2 still accepted by the loader.
 * RoPE, MQA/GQA, SwiGLU (fused gate+up), RMSNorm, optional LayerScale.
 */
#define _GNU_SOURCE
#include "ll3m.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CLM_HOT __attribute__((hot, section(".text.hot")))
#define CLM_RESTRICT __restrict__
#else
#define CLM_HOT
#define CLM_RESTRICT
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
/*
 * Weight-tile software prefetch knobs (R1 / docs/READAHEAD_DESIGN.md):
 *   -DCLM_W_PREFETCH_AHEAD=N   packed-byte offset in row loops (default 32)
 *   -DCLM_W_PREFETCH_HINT=0|1|2  0=NTA (default), 1=T1/L2, 2=T0/L1
 *   -DCLM_W_ROW_PREFETCH=N     bytes into current/next rows (default 64)
 *   -DCLM_NEXT_LAYER_PREFETCH=0  disable end-of-layer next-matrix kick
 */
#ifndef CLM_W_PREFETCH_AHEAD
#define CLM_W_PREFETCH_AHEAD 32
#endif
#ifndef CLM_W_ROW_PREFETCH
#define CLM_W_ROW_PREFETCH 64
#endif
#ifndef CLM_W_PREFETCH_HINT
#define CLM_W_PREFETCH_HINT 0
#endif
#ifndef CLM_NEXT_LAYER_PREFETCH
#define CLM_NEXT_LAYER_PREFETCH 1
#endif
#if CLM_W_PREFETCH_HINT == 2
#define CLM_W_HINT _MM_HINT_T0
#elif CLM_W_PREFETCH_HINT == 1
#define CLM_W_HINT _MM_HINT_T1
#else
#define CLM_W_HINT _MM_HINT_NTA
#endif
#define CLM_PREFETCH_W(p) _mm_prefetch((const char *)(p), CLM_W_HINT)
#define CLM_PREFETCH_X(p) _mm_prefetch((const char *)(p), _MM_HINT_T0)
#else
#ifndef CLM_W_PREFETCH_AHEAD
#define CLM_W_PREFETCH_AHEAD 32
#endif
#ifndef CLM_W_ROW_PREFETCH
#define CLM_W_ROW_PREFETCH 64
#endif
#ifndef CLM_NEXT_LAYER_PREFETCH
#define CLM_NEXT_LAYER_PREFETCH 1
#endif
#define CLM_PREFETCH_W(p) ((void)0)
#define CLM_PREFETCH_X(p) ((void)0)
#endif

/* ---- ternary Linear: 2-bit pack, codes 0→-1, 1→0, 2→+1 ---- */
typedef struct {
    const uint8_t *wp; /* out * stride */
    const float *scale; /* out */
    int out;
    int in;
    int stride; /* ceil(in/4) */
} TMat;

static inline int tmat_stride(int in) { return (in + 3) / 4; }
/* Binary 1-bit pack: 8 weights per byte (L3B2 research). */
static inline int bmat_stride(int in) { return (in + 7) / 8; }

#if defined(__AVX512F__)
static inline float hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }

/* L1-hot LUT: byte → 4 ternary floats {-1,0,+1}. Built once. */
static float g_ternary_lut[256][4] __attribute__((aligned(64)));
static int g_ternary_lut_ready = 0;

static void ternary_lut_init(void) {
    if (g_ternary_lut_ready)
        return;
    for (int b = 0; b < 256; b++) {
        g_ternary_lut[b][0] = (float)((b & 3) - 1);
        g_ternary_lut[b][1] = (float)(((b >> 2) & 3) - 1);
        g_ternary_lut[b][2] = (float)(((b >> 4) & 3) - 1);
        g_ternary_lut[b][3] = (float)(((b >> 6) & 3) - 1);
    }
    g_ternary_lut_ready = 1;
}

/* Unpack 4 packed bytes → 16 ternary floats via LUT (no stack bounce). */
static inline __m512 unpack_ternary16(const uint8_t *CLM_RESTRICT p) {
    __m128 a = _mm_load_ps(g_ternary_lut[p[0]]);
    __m128 b = _mm_load_ps(g_ternary_lut[p[1]]);
    __m128 c = _mm_load_ps(g_ternary_lut[p[2]]);
    __m128 d = _mm_load_ps(g_ternary_lut[p[3]]);
    __m512 r = _mm512_castps128_ps512(a);
    r = _mm512_insertf32x4(r, b, 1);
    r = _mm512_insertf32x4(r, c, 2);
    r = _mm512_insertf32x4(r, d, 3);
    return r;
}

static inline float dot_ternary_row(const float *CLM_RESTRICT x, const uint8_t *CLM_RESTRICT row,
                                    int in) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    int i = 0;
    /* 32 weights / step: two 16-wide FMAs, hide latency */
    for (; i + 32 <= in; i += 32) {
        CLM_PREFETCH_W(row + (i >> 2) + CLM_W_PREFETCH_AHEAD);
        __m512 q0 = unpack_ternary16(row + (i >> 2));
        __m512 q1 = unpack_ternary16(row + (i >> 2) + 4);
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), q0, acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i + 16), q1, acc1);
    }
    for (; i + 16 <= in; i += 16) {
        __m512 q0 = unpack_ternary16(row + (i >> 2));
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), q0, acc0);
    }
    float s = hsum512(_mm512_add_ps(acc0, acc1));
    for (; i < in; i++) {
        uint8_t b = row[i >> 2];
        int c = (b >> (2 * (i & 3))) & 3;
        if (c != 1)
            s += (c == 2) ? x[i] : -x[i];
    }
    return s;
}

static inline void dot_ternary_row_pair(const float *CLM_RESTRICT x,
                                        const uint8_t *CLM_RESTRICT rg,
                                        const uint8_t *CLM_RESTRICT ru, int in, float *sg,
                                        float *su) {
    __m512 ag0 = _mm512_setzero_ps(), ag1 = _mm512_setzero_ps();
    __m512 au0 = _mm512_setzero_ps(), au1 = _mm512_setzero_ps();
    int i = 0;
    for (; i + 32 <= in; i += 32) {
        CLM_PREFETCH_W(rg + (i >> 2) + CLM_W_PREFETCH_AHEAD);
        CLM_PREFETCH_W(ru + (i >> 2) + CLM_W_PREFETCH_AHEAD);
        __m512 x0 = _mm512_loadu_ps(x + i);
        __m512 x1 = _mm512_loadu_ps(x + i + 16);
        __m512 qg0 = unpack_ternary16(rg + (i >> 2));
        __m512 qg1 = unpack_ternary16(rg + (i >> 2) + 4);
        __m512 qu0 = unpack_ternary16(ru + (i >> 2));
        __m512 qu1 = unpack_ternary16(ru + (i >> 2) + 4);
        ag0 = _mm512_fmadd_ps(x0, qg0, ag0);
        au0 = _mm512_fmadd_ps(x0, qu0, au0);
        ag1 = _mm512_fmadd_ps(x1, qg1, ag1);
        au1 = _mm512_fmadd_ps(x1, qu1, au1);
    }
    for (; i + 16 <= in; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        ag0 = _mm512_fmadd_ps(xv, unpack_ternary16(rg + (i >> 2)), ag0);
        au0 = _mm512_fmadd_ps(xv, unpack_ternary16(ru + (i >> 2)), au0);
    }
    float g = hsum512(_mm512_add_ps(ag0, ag1));
    float u = hsum512(_mm512_add_ps(au0, au1));
    for (; i < in; i++) {
        int sh = 2 * (i & 3);
        int cg = (rg[i >> 2] >> sh) & 3;
        int cu = (ru[i >> 2] >> sh) & 3;
        float xv = x[i];
        if (cg != 1)
            g += (cg == 2) ? xv : -xv;
        if (cu != 1)
            u += (cu == 2) ? xv : -xv;
    }
    *sg = g;
    *su = u;
}

#endif /* __AVX512F__ */

static CLM_HOT void gemv_f32(float *CLM_RESTRICT y, const float *CLM_RESTRICT x,
                             const float *CLM_RESTRICT W, int out, int in) {
    for (int o = 0; o < out; o++) {
        const float *row = W + (size_t)o * in;
        float s = 0.f;
#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= in; i += 16)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(x + i), _mm512_loadu_ps(row + i), acc);
        s = hsum512(acc);
        for (; i < in; i++)
            s += x[i] * row[i];
#else
        for (int i = 0; i < in; i++)
            s += x[i] * row[i];
#endif
        y[o] = s;
    }
}

/* y[o] = scale[o] * sum_i q[o,i] * x[i]
 * AVX path: two output rows per inner pass so x[] loads stay hot. */
static CLM_HOT void gemv_ternary(float *CLM_RESTRICT y, const float *CLM_RESTRICT x,
                                 const TMat *m) {
    int out = m->out, in = m->in, stride = m->stride;
    const uint8_t *wp = m->wp;
    const float *sc = m->scale;
    int o = 0;
#if defined(__AVX512F__)
    for (; o + 4 <= out; o += 4) {
        const uint8_t *r0 = wp + (size_t)o * stride;
        const uint8_t *r1 = r0 + stride;
        const uint8_t *r2 = r1 + stride;
        const uint8_t *r3 = r2 + stride;
        CLM_PREFETCH_W(r0 + CLM_W_ROW_PREFETCH);
        CLM_PREFETCH_W(r1 + CLM_W_ROW_PREFETCH);
        CLM_PREFETCH_W(r2 + CLM_W_ROW_PREFETCH);
        CLM_PREFETCH_W(r3 + CLM_W_ROW_PREFETCH);
        if (o + 8 <= out)
            CLM_PREFETCH_W(wp + (size_t)(o + 4) * stride);
        float s0, s1, s2, s3;
        dot_ternary_row_pair(x, r0, r1, in, &s0, &s1);
        dot_ternary_row_pair(x, r2, r3, in, &s2, &s3);
        y[o] = s0 * sc[o];
        y[o + 1] = s1 * sc[o + 1];
        y[o + 2] = s2 * sc[o + 2];
        y[o + 3] = s3 * sc[o + 3];
    }
    for (; o + 2 <= out; o += 2) {
        const uint8_t *r0 = wp + (size_t)o * stride;
        const uint8_t *r1 = r0 + stride;
        float s0, s1;
        dot_ternary_row_pair(x, r0, r1, in, &s0, &s1);
        y[o] = s0 * sc[o];
        y[o + 1] = s1 * sc[o + 1];
    }
    for (; o < out; o++) {
        const uint8_t *row = wp + (size_t)o * stride;
        y[o] = sc[o] * dot_ternary_row(x, row, in);
    }
#else
    for (; o < out; o++) {
        const uint8_t *row = wp + (size_t)o * stride;
        CLM_PREFETCH_W(row + CLM_W_ROW_PREFETCH);
        float s = 0.f;
        int i = 0;
        for (; i + 4 <= in; i += 4) {
            uint8_t b = row[i >> 2];
            int c0 = b & 3, c1 = (b >> 2) & 3, c2 = (b >> 4) & 3, c3 = (b >> 6) & 3;
            if (c0 != 1)
                s += (c0 == 2) ? x[i] : -x[i];
            if (c1 != 1)
                s += (c1 == 2) ? x[i + 1] : -x[i + 1];
            if (c2 != 1)
                s += (c2 == 2) ? x[i + 2] : -x[i + 2];
            if (c3 != 1)
                s += (c3 == 2) ? x[i + 3] : -x[i + 3];
        }
        for (; i < in; i++) {
            uint8_t b = row[i >> 2];
            int c = (b >> (2 * (i & 3))) & 3;
            if (c != 1)
                s += (c == 2) ? x[i] : -x[i];
        }
        y[o] = s * sc[o];
    }
#endif
}

/* ---- binary Linear: 1-bit pack, bit0→-1 / bit1→+1 (LSB = first col) ---- */

static inline float dot_binary_row(const float *CLM_RESTRICT x, const uint8_t *CLM_RESTRICT row,
                                   int in) {
    float s = 0.f;
    int i = 0;
    for (; i + 8 <= in; i += 8) {
        uint8_t b = row[i >> 3];
        s += ((b & 1) ? x[i] : -x[i]);
        s += ((b & 2) ? x[i + 1] : -x[i + 1]);
        s += ((b & 4) ? x[i + 2] : -x[i + 2]);
        s += ((b & 8) ? x[i + 3] : -x[i + 3]);
        s += ((b & 16) ? x[i + 4] : -x[i + 4]);
        s += ((b & 32) ? x[i + 5] : -x[i + 5]);
        s += ((b & 64) ? x[i + 6] : -x[i + 6]);
        s += ((b & 128) ? x[i + 7] : -x[i + 7]);
    }
    if (i < in) {
        uint8_t b = row[i >> 3];
        for (int k = 0; i + k < in; k++)
            s += ((b & (1u << k)) ? x[i + k] : -x[i + k]);
    }
    return s;
}

/* Portable scalar GEMV for L3B2 (research stub; AVX later). */
static CLM_HOT void gemv_binary(float *CLM_RESTRICT y, const float *CLM_RESTRICT x,
                                const TMat *m) {
    int out = m->out, in = m->in, stride = m->stride;
    const uint8_t *wp = m->wp;
    const float *sc = m->scale;
    for (int o = 0; o < out; o++) {
        const uint8_t *row = wp + (size_t)o * stride;
        CLM_PREFETCH_W(row + CLM_W_ROW_PREFETCH);
        y[o] = sc[o] * dot_binary_row(x, row, in);
    }
}

static CLM_HOT void gemv_binary_pair(float *CLM_RESTRICT gate, float *CLM_RESTRICT up,
                                     const float *CLM_RESTRICT x, const TMat *mg,
                                     const TMat *mu) {
    int mid = mg->out, in = mg->in, stride = mg->stride;
    for (int o = 0; o < mid; o++) {
        const uint8_t *rg = mg->wp + (size_t)o * stride;
        const uint8_t *ru = mu->wp + (size_t)o * stride;
        CLM_PREFETCH_W(rg + CLM_W_ROW_PREFETCH);
        CLM_PREFETCH_W(ru + CLM_W_ROW_PREFETCH);
        gate[o] = mg->scale[o] * dot_binary_row(x, rg, in);
        up[o] = mu->scale[o] * dot_binary_row(x, ru, in);
    }
}

/* Fused SwiGLU projections: stream x once per output row for gate and up. */
static CLM_HOT void gemv_ternary_pair(float *CLM_RESTRICT gate, float *CLM_RESTRICT up,
                                      const float *CLM_RESTRICT x, const TMat *mg,
                                      const TMat *mu) {
    int mid = mg->out, in = mg->in, stride = mg->stride;
    for (int o = 0; o < mid; o++) {
        const uint8_t *rg = mg->wp + (size_t)o * stride;
        const uint8_t *ru = mu->wp + (size_t)o * stride;
        CLM_PREFETCH_W(rg + CLM_W_ROW_PREFETCH);
        CLM_PREFETCH_W(ru + CLM_W_ROW_PREFETCH);
#if defined(__AVX512F__)
        float sg, su;
        dot_ternary_row_pair(x, rg, ru, in, &sg, &su);
        gate[o] = sg * mg->scale[o];
        up[o] = su * mu->scale[o];
#else
        float sg = 0.f, su = 0.f;
        int i = 0;
        for (; i + 4 <= in; i += 4) {
            uint8_t bg = rg[i >> 2];
            uint8_t bu = ru[i >> 2];
            float x0 = x[i], x1 = x[i + 1], x2 = x[i + 2], x3 = x[i + 3];
#define ACC(s, b, xv)                                                              \
    do {                                                                           \
        int c = (b)&3;                                                             \
        if (c != 1)                                                                \
            (s) += (c == 2) ? (xv) : -(xv);                                        \
    } while (0)
            ACC(sg, bg, x0);
            ACC(su, bu, x0);
            ACC(sg, bg >> 2, x1);
            ACC(su, bu >> 2, x1);
            ACC(sg, bg >> 4, x2);
            ACC(su, bu >> 4, x2);
            ACC(sg, bg >> 6, x3);
            ACC(su, bu >> 6, x3);
#undef ACC
        }
        for (; i < in; i++) {
            int sh = 2 * (i & 3);
            int cg = (rg[i >> 2] >> sh) & 3;
            int cu = (ru[i >> 2] >> sh) & 3;
            float xv = x[i];
            if (cg != 1)
                sg += (cg == 2) ? xv : -xv;
            if (cu != 1)
                su += (cu == 2) ? xv : -xv;
        }
        gate[o] = sg * mg->scale[o];
        up[o] = su * mu->scale[o];
#endif
    }
}

static void rms_norm(float *out, const float *x, const float *w, int D) {
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= D; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        acc = _mm512_fmadd_ps(xv, xv, acc);
    }
    float ms = hsum512(acc);
    for (; i < D; i++)
        ms += x[i] * x[i];
    ms = 1.f / sqrtf(ms / (float)D + 1e-6f);
    __m512 msv = _mm512_set1_ps(ms);
    for (i = 0; i + 16 <= D; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 wv = _mm512_loadu_ps(w + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(_mm512_mul_ps(xv, msv), wv));
    }
    for (; i < D; i++)
        out[i] = x[i] * ms * w[i];
#else
    float ms = 0.f;
    for (int i = 0; i < D; i++)
        ms += x[i] * x[i];
    ms = 1.f / sqrtf(ms / (float)D + 1e-6f);
    for (int i = 0; i < D; i++)
        out[i] = x[i] * ms * w[i];
#endif
}

/* logits[v] = sc[v] * sum_i xb[i] * wte_q[v,i]  (int8 embeds)
 * AVX: four vocab rows share xb loads (V=2048 product). */
static CLM_HOT void lm_head_int8(float *CLM_RESTRICT logits, const float *CLM_RESTRICT xb,
                                 const int8_t *CLM_RESTRICT wte_q,
                                 const float *CLM_RESTRICT wte_sc, int V, int D) {
    int v = 0;
#if defined(__AVX512F__)
    for (; v + 4 <= V; v += 4) {
        const int8_t *r0 = wte_q + (size_t)v * D;
        const int8_t *r1 = r0 + D;
        const int8_t *r2 = r1 + D;
        const int8_t *r3 = r2 + D;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            __m512 xv = _mm512_loadu_ps(xb + i);
            __m128i b0 = _mm_loadu_si128((const __m128i *)(r0 + i));
            __m128i b1 = _mm_loadu_si128((const __m128i *)(r1 + i));
            __m128i b2 = _mm_loadu_si128((const __m128i *)(r2 + i));
            __m128i b3 = _mm_loadu_si128((const __m128i *)(r3 + i));
            a0 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0)), a0);
            a1 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1)), a1);
            a2 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b2)), a2);
            a3 = _mm512_fmadd_ps(xv, _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b3)), a3);
        }
        float s0 = hsum512(a0), s1 = hsum512(a1), s2 = hsum512(a2), s3 = hsum512(a3);
        for (; i < D; i++) {
            float xv = xb[i];
            s0 += xv * (float)r0[i];
            s1 += xv * (float)r1[i];
            s2 += xv * (float)r2[i];
            s3 += xv * (float)r3[i];
        }
        logits[v] = s0 * wte_sc[v];
        logits[v + 1] = s1 * wte_sc[v + 1];
        logits[v + 2] = s2 * wte_sc[v + 2];
        logits[v + 3] = s3 * wte_sc[v + 3];
    }
    for (; v + 2 <= V; v += 2) {
        const int8_t *r0 = wte_q + (size_t)v * D;
        const int8_t *r1 = r0 + D;
        __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            __m512 xv = _mm512_loadu_ps(xb + i);
            __m128i b0 = _mm_loadu_si128((const __m128i *)(r0 + i));
            __m128i b1 = _mm_loadu_si128((const __m128i *)(r1 + i));
            __m512 x0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b0));
            __m512 x1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b1));
            a0 = _mm512_fmadd_ps(xv, x0, a0);
            a1 = _mm512_fmadd_ps(xv, x1, a1);
        }
        float s0 = hsum512(a0), s1 = hsum512(a1);
        for (; i < D; i++) {
            float xv = xb[i];
            s0 += xv * (float)r0[i];
            s1 += xv * (float)r1[i];
        }
        logits[v] = s0 * wte_sc[v];
        logits[v + 1] = s1 * wte_sc[v + 1];
    }
#endif
    for (; v < V; v++) {
        const int8_t *row = wte_q + (size_t)v * D;
        float sc = wte_sc[v];
#if defined(__AVX512F__)
        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            __m128i b = _mm_loadu_si128((const __m128i *)(row + i));
            __m512i bi = _mm512_cvtepi8_epi32(b);
            __m512 xf = _mm512_cvtepi32_ps(bi);
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(xb + i), xf, acc);
        }
        float s = hsum512(acc);
        for (; i < D; i++)
            s += xb[i] * (float)row[i];
        logits[v] = s * sc;
#else
        float s = 0.f;
        for (int i = 0; i < D; i++)
            s += xb[i] * (float)row[i];
        logits[v] = s * sc;
#endif
    }
}

static void silu_mul(float *y, const float *gate, const float *up, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        float sig = 1.f / (1.f + expf(-g));
        y[i] = (g * sig) * up[i];
    }
}

/* f32 helpers used by attention / residual (hd often 48 = 3×16). */
static inline float dot_f32(const float *CLM_RESTRICT a, const float *CLM_RESTRICT b, int n) {
#if defined(__AVX512F__)
    /* Product head_dim=48 = 3×16: fully unrolled, no tail. */
    if (n == 48) {
        __m512 a0 = _mm512_loadu_ps(a);
        __m512 b0 = _mm512_loadu_ps(b);
        __m512 a1 = _mm512_loadu_ps(a + 16);
        __m512 b1 = _mm512_loadu_ps(b + 16);
        __m512 a2 = _mm512_loadu_ps(a + 32);
        __m512 b2 = _mm512_loadu_ps(b + 32);
        __m512 acc = _mm512_mul_ps(a0, b0);
        acc = _mm512_fmadd_ps(a1, b1, acc);
        acc = _mm512_fmadd_ps(a2, b2, acc);
        return hsum512(acc);
    }
    __m512 acc = _mm512_setzero_ps();
    int i = 0;
    for (; i + 16 <= n; i += 16)
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
    float s = hsum512(acc);
    for (; i < n; i++)
        s += a[i] * b[i];
    return s;
#else
    float s = 0.f;
    for (int i = 0; i < n; i++)
        s += a[i] * b[i];
    return s;
#endif
}

static inline void axpy_f32(float *CLM_RESTRICT y, const float *CLM_RESTRICT x, float a, int n) {
#if defined(__AVX512F__)
    __m512 av = _mm512_set1_ps(a);
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 yv = _mm512_loadu_ps(y + i);
        __m512 xv = _mm512_loadu_ps(x + i);
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(av, xv, yv));
    }
    for (; i < n; i++)
        y[i] += a * x[i];
#else
    for (int i = 0; i < n; i++)
        y[i] += a * x[i];
#endif
}

static inline void add_f32(float *CLM_RESTRICT y, const float *CLM_RESTRICT x, int n) {
#if defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= n; i += 16)
        _mm512_storeu_ps(y + i, _mm512_add_ps(_mm512_loadu_ps(y + i), _mm512_loadu_ps(x + i)));
    for (; i < n; i++)
        y[i] += x[i];
#else
    for (int i = 0; i < n; i++)
        y[i] += x[i];
#endif
}

static inline void add_scaled_f32(float *CLM_RESTRICT y, const float *CLM_RESTRICT x,
                                  const float *CLM_RESTRICT s, int n) {
#if defined(__AVX512F__)
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 sv = _mm512_loadu_ps(s + i);
        __m512 yv = _mm512_loadu_ps(y + i);
        _mm512_storeu_ps(y + i, _mm512_fmadd_ps(xv, sv, yv));
    }
    for (; i < n; i++)
        y[i] += x[i] * s[i];
#else
    for (int i = 0; i < n; i++)
        y[i] += x[i] * s[i];
#endif
}

/* QK-Norm: x = rms(x) * scale_vec (in-place). */
static inline void qk_norm_inplace(float *CLM_RESTRICT x, const float *CLM_RESTRICT scale, int hd) {
    float ms = dot_f32(x, x, hd);
    ms = 1.f / sqrtf(ms / (float)hd + 1e-6f);
#if defined(__AVX512F__)
    __m512 msv = _mm512_set1_ps(ms);
    int i = 0;
    for (; i + 16 <= hd; i += 16) {
        __m512 xv = _mm512_loadu_ps(x + i);
        __m512 sv = _mm512_loadu_ps(scale + i);
        _mm512_storeu_ps(x + i, _mm512_mul_ps(_mm512_mul_ps(xv, msv), sv));
    }
    for (; i < hd; i++)
        x[i] = x[i] * ms * scale[i];
#else
    for (int i = 0; i < hd; i++)
        x[i] = x[i] * ms * scale[i];
#endif
}

/*
 * LLaMA / torch rotate-half RoPE (matches ll3m.model_torch.apply_rope).
 * inv[j] = theta ^ (-2j / hd), j = 0 .. half-1
 * out[i]       =  q[i]*c - q[i+half]*s
 * out[i+half]  =  q[i+half]*c + q[i]*s
 *
 * Product path (hd=48, T<=256, theta=10000): cache cos/sin per (pos,j).
 * Falls back to powf/cosf/sinf for exotic shapes.
 */
enum { ROPE_CACHE_T = 256, ROPE_CACHE_HALF = 32 }; /* hd=48 → half=24; pad to 32 */
static float g_rope_cos[ROPE_CACHE_T][ROPE_CACHE_HALF];
static float g_rope_sin[ROPE_CACHE_T][ROPE_CACHE_HALF];
static int g_rope_hd = 0;
static float g_rope_theta = 0.f;
static int g_rope_ready = 0;

static void rope_cache_init(int hd, float theta) {
    int half = hd / 2;
    if (half > ROPE_CACHE_HALF || half < 1)
        return;
    for (int pos = 0; pos < ROPE_CACHE_T; pos++) {
        for (int j = 0; j < half; j++) {
            float freq = 1.f / powf(theta, (float)(2 * j) / (float)hd);
            float ang = (float)pos * freq;
            g_rope_cos[pos][j] = cosf(ang);
            g_rope_sin[pos][j] = sinf(ang);
        }
    }
    g_rope_hd = hd;
    g_rope_theta = theta;
    g_rope_ready = 1;
}

static void rope_half_inplace(float *x, int hd, int pos, float theta) {
    int half = hd / 2;
    if (g_rope_ready && g_rope_hd == hd && g_rope_theta == theta && pos >= 0 &&
        pos < ROPE_CACHE_T && half <= ROPE_CACHE_HALF) {
        const float *c = g_rope_cos[pos];
        const float *s = g_rope_sin[pos];
        for (int j = 0; j < half; j++) {
            float x0 = x[j], x1 = x[j + half];
            float cj = c[j], sj = s[j];
            x[j] = x0 * cj - x1 * sj;
            x[j + half] = x1 * cj + x0 * sj;
        }
        return;
    }
    for (int j = 0; j < half; j++) {
        float freq = 1.f / powf(theta, (float)(2 * j) / (float)hd);
        float ang = (float)pos * freq;
        float c = cosf(ang), s = sinf(ang);
        float x0 = x[j], x1 = x[j + half];
        x[j] = x0 * c - x1 * s;
        x[j + half] = x1 * c + x0 * s;
    }
}

/* ---- f32 weight map (CLM2) ---- */
typedef struct {
    const float *wte;
    struct {
        const float *rms1;
        const float *wq, *wk, *wv, *wo;
        const float *q_norm, *k_norm;
        const float *rms2;
        const float *wgate, *wup, *wdown;
        const float *ls1, *ls2;
        const float *post_attn, *post_mlp;
    } layer[48];
    const float *ln_f;
} WV2f;

/* ---- ternary weight map (CLT2) ---- */
typedef struct {
    const int8_t *wte_q; /* V*D */
    const float *wte_scale; /* V */
    struct {
        const float *rms1;
        TMat wq, wk, wv, wo;
        const float *q_norm, *k_norm;
        const float *rms2;
        TMat wgate, wup, wdown;
        const float *ls1, *ls2;
        const float *post_attn, *post_mlp;
    } layer[48];
    const float *ln_f;
} WV2t;

static void map_v2_ternary_fixed(WV2t *v, const unsigned char *base, const ClmConfig *c);
static void map_v2_binary_fixed(WV2t *v, const unsigned char *base, const ClmConfig *c);

static const float *take_f(const float **p, size_t n) {
    const float *r = *p;
    *p += n;
    return r;
}

static void map_v2_f32(WV2f *v, const float *base, const ClmConfig *c) {
    int V = c->vocab_size, L = c->n_layer, D = c->n_embd;
    int nkv = c->n_kv_head, hd = c->head_dim, mid = c->intermediate;
    const float *p = base;
    memset(v, 0, sizeof(*v));
    v->wte = take_f(&p, (size_t)V * D);
    for (int i = 0; i < L; i++) {
        v->layer[i].rms1 = take_f(&p, D);
        v->layer[i].wq = take_f(&p, (size_t)D * D);
        v->layer[i].wk = take_f(&p, (size_t)nkv * hd * D);
        v->layer[i].wv = take_f(&p, (size_t)nkv * hd * D);
        v->layer[i].wo = take_f(&p, (size_t)D * D);
        v->layer[i].q_norm = take_f(&p, hd);
        v->layer[i].k_norm = take_f(&p, hd);
        v->layer[i].rms2 = take_f(&p, D);
        v->layer[i].wgate = take_f(&p, (size_t)mid * D);
        v->layer[i].wup = take_f(&p, (size_t)mid * D);
        v->layer[i].wdown = take_f(&p, (size_t)D * mid);
        if (c->has_ls) {
            v->layer[i].ls1 = take_f(&p, D);
            v->layer[i].ls2 = take_f(&p, D);
        }
        if (c->has_peri) {
            v->layer[i].post_attn = take_f(&p, D);
            v->layer[i].post_mlp = take_f(&p, D);
        }
    }
    v->ln_f = take_f(&p, D);
}

static size_t v2_f32_count(const ClmConfig *c) {
    int V = c->vocab_size, L = c->n_layer, D = c->n_embd;
    int nkv = c->n_kv_head, hd = c->head_dim, mid = c->intermediate;
    size_t n = (size_t)V * D;
    size_t per = (size_t)D + (size_t)D * D + (size_t)nkv * hd * D + (size_t)nkv * hd * D +
                 (size_t)D * D + (size_t)hd + hd + D + (size_t)mid * D + (size_t)mid * D +
                 (size_t)D * mid;
    if (c->has_ls)
        per += 2 * (size_t)D;
    if (c->has_peri)
        per += 2 * (size_t)D;
    return n + (size_t)L * per + (size_t)D;
}

static size_t norms_f32_count(const ClmConfig *c) {
    int L = c->n_layer, D = c->n_embd, hd = c->head_dim;
    size_t per = (size_t)D + hd + hd + D; /* rms1, qn, kn, rms2 */
    if (c->has_ls)
        per += 2 * (size_t)D;
    if (c->has_peri)
        per += 2 * (size_t)D;
    return (size_t)L * per + (size_t)D;
}

/* ---- load ---- */

static int read_cfg_header(ClmModel *m, FILE *f, const char *path, uint32_t *tok_len_out) {
    uint32_t V, T, L, H, nkv, D, mid, flags, tok_len;
    float theta;
    if (fread(&V, 4, 1, f) != 1 || fread(&T, 4, 1, f) != 1 || fread(&L, 4, 1, f) != 1 ||
        fread(&H, 4, 1, f) != 1 || fread(&nkv, 4, 1, f) != 1 || fread(&D, 4, 1, f) != 1 ||
        fread(&mid, 4, 1, f) != 1 || fread(&theta, 4, 1, f) != 1 ||
        fread(&flags, 4, 1, f) != 1 || fread(&tok_len, 4, 1, f) != 1) {
        fprintf(stderr, "short header in %s\n", path);
        return -1;
    }
    m->cfg.vocab_size = (int)V;
    m->cfg.block_size = (int)T;
    m->cfg.n_layer = (int)L;
    m->cfg.n_head = (int)H;
    m->cfg.n_kv_head = (int)nkv;
    m->cfg.n_embd = (int)D;
    m->cfg.intermediate = (int)mid;
    m->cfg.rope_theta = theta;
    m->cfg.arch = 2;
    m->cfg.has_ls = (flags & 1) != 0;
    m->cfg.has_peri = (flags & 2) != 0;
    m->cfg.head_dim = (int)D / (int)H;
    if (m->cfg.head_dim * m->cfg.n_head != m->cfg.n_embd || L < 1 || L > 48) {
        fprintf(stderr, "invalid config in %s\n", path);
        return -1;
    }
    *tok_len_out = tok_len;
    return 0;
}

static int read_tok(ClmModel *m, FILE *f, uint32_t tok_len) {
    if (tok_len == 0)
        return 0;
    m->tok_json = (unsigned char *)malloc(tok_len);
    if (!m->tok_json || fread(m->tok_json, 1, tok_len, f) != tok_len) {
        free(m->tok_json);
        m->tok_json = NULL;
        return -1;
    }
    m->tok_json_len = tok_len;
    return 0;
}

int clm_model_load_v2(ClmModel *m, FILE *f, const char *path) {
    /* Called after magic CLM2 already consumed. */
    uint32_t tok_len = 0;
    if (read_cfg_header(m, f, path, &tok_len) != 0) {
        fclose(f);
        return -1;
    }
    if (read_tok(m, f, tok_len) != 0) {
        fclose(f);
        return -1;
    }
    size_t nf = v2_f32_count(&m->cfg);
    size_t bytes = nf * sizeof(float);
    float *src = (float *)malloc(bytes);
    if (!src || fread(src, 1, bytes, f) != bytes) {
        free(src);
        free(m->tok_json);
        m->tok_json = NULL;
        fclose(f);
        fprintf(stderr, "L3M2 weight read failed (%zu bytes expected)\n", bytes);
        return -1;
    }
    fclose(f);
    m->arena = src;
    m->arena_bytes = bytes;
    m->arena_mmap = 0;
    m->w = src;
    m->n_weight = nf;
    m->weight_bytes = bytes;
    m->quantized = 0;
    m->vocab = NULL;
    m->locked = 0;
#if defined(__linux__)
    if (mlock(src, bytes) == 0)
        m->locked = 1;
#endif
    rope_cache_init(m->cfg.head_dim, m->cfg.rope_theta);
    fprintf(stderr,
            "L3M2 loaded: L%d d%d h%d/kv%d mid%d  weights=%.1f MiB (f32)\n",
            m->cfg.n_layer, m->cfg.n_embd, m->cfg.n_head, m->cfg.n_kv_head,
            m->cfg.intermediate, bytes / (1024.0 * 1024.0));
    return 0;
}

/* CLT2: norms f32 + int8 embeds + ternary linears. Stored as opaque arena; w points
 * to start. map rebuilt each forward from arena base (cheap pointer walk). */
int clm_model_load_v2t(ClmModel *m, FILE *f, const char *path) {
#if defined(__AVX512F__)
    ternary_lut_init();
#endif
    uint32_t tok_len = 0;
    if (read_cfg_header(m, f, path, &tok_len) != 0) {
        fclose(f);
        return -1;
    }
    if (read_tok(m, f, tok_len) != 0) {
        fclose(f);
        return -1;
    }
    const ClmConfig *c = &m->cfg;
    int V = c->vocab_size, L = c->n_layer, D = c->n_embd;
    int nkv = c->n_kv_head, hd = c->head_dim, mid = c->intermediate;

    size_t nf_norms = norms_f32_count(c);
    size_t embed_q = (size_t)V * D;
    size_t embed_sc = (size_t)V;
    /* linear bytes: for each mat scale[out]*4 + out*ceil(in/4) */
    size_t lin = 0;
    size_t mats[][2] = {
        {(size_t)D, (size_t)D},
        {(size_t)nkv * hd, (size_t)D},
        {(size_t)nkv * hd, (size_t)D},
        {(size_t)D, (size_t)D},
        {(size_t)mid, (size_t)D},
        {(size_t)mid, (size_t)D},
        {(size_t)D, (size_t)mid},
    };
    for (int i = 0; i < L; i++) {
        for (int k = 0; k < 7; k++) {
            size_t o = mats[k][0], inn = mats[k][1];
            lin += o * sizeof(float) + o * (size_t)tmat_stride((int)inn);
        }
    }
    size_t bytes = nf_norms * sizeof(float) + embed_q + embed_sc * sizeof(float) + lin;
    unsigned char *src = (unsigned char *)malloc(bytes);
    if (!src || fread(src, 1, bytes, f) != bytes) {
        free(src);
        free(m->tok_json);
        m->tok_json = NULL;
        fclose(f);
        fprintf(stderr, "L3T2 weight read failed (%zu bytes expected)\n", bytes);
        return -1;
    }
    fclose(f);
    m->arena = src;
    m->arena_bytes = bytes;
    m->arena_mmap = 0;
    m->w = (float *)src; /* opaque base */
    m->n_weight = bytes / sizeof(float);
    m->weight_bytes = bytes;
    m->quantized = 2; /* ternary linears */
    m->vocab = NULL;
    m->locked = 0;
#if defined(__linux__)
    if (mlock(src, bytes) == 0)
        m->locked = 1;
#endif
    rope_cache_init(m->cfg.head_dim, m->cfg.rope_theta);
    {
        WV2t *map = (WV2t *)malloc(sizeof(WV2t));
        if (map) {
            map_v2_ternary_fixed(map, (const unsigned char *)m->arena, &m->cfg);
            m->map_cache = map;
            m->map_cache_ready = 1;
        } else {
            m->map_cache = NULL;
            m->map_cache_ready = 0;
        }
    }
    fprintf(stderr,
            "L3T2 loaded: L%d d%d h%d/kv%d mid%d  weights=%.2f MiB (ternary+int8 embed)\n",
            m->cfg.n_layer, m->cfg.n_embd, m->cfg.n_head, m->cfg.n_kv_head,
            m->cfg.intermediate, bytes / (1024.0 * 1024.0));
    return 0;
}

/* L3B2: same layout as L3T2 but 1-bit packed Linears (research denser pack). */
int clm_model_load_v2b(ClmModel *m, FILE *f, const char *path) {
    uint32_t tok_len = 0;
    if (read_cfg_header(m, f, path, &tok_len) != 0) {
        fclose(f);
        return -1;
    }
    if (read_tok(m, f, tok_len) != 0) {
        fclose(f);
        return -1;
    }
    const ClmConfig *c = &m->cfg;
    int V = c->vocab_size, L = c->n_layer, D = c->n_embd;
    int nkv = c->n_kv_head, hd = c->head_dim, mid = c->intermediate;

    size_t nf_norms = norms_f32_count(c);
    size_t embed_q = (size_t)V * D;
    size_t embed_sc = (size_t)V;
    size_t lin = 0;
    size_t mats[][2] = {
        {(size_t)D, (size_t)D},
        {(size_t)nkv * hd, (size_t)D},
        {(size_t)nkv * hd, (size_t)D},
        {(size_t)D, (size_t)D},
        {(size_t)mid, (size_t)D},
        {(size_t)mid, (size_t)D},
        {(size_t)D, (size_t)mid},
    };
    for (int i = 0; i < L; i++) {
        for (int k = 0; k < 7; k++) {
            size_t o = mats[k][0], inn = mats[k][1];
            lin += o * sizeof(float) + o * (size_t)bmat_stride((int)inn);
        }
    }
    size_t bytes = nf_norms * sizeof(float) + embed_q + embed_sc * sizeof(float) + lin;
    unsigned char *src = (unsigned char *)malloc(bytes);
    if (!src || fread(src, 1, bytes, f) != bytes) {
        free(src);
        free(m->tok_json);
        m->tok_json = NULL;
        fclose(f);
        fprintf(stderr, "L3B2 weight read failed (%zu bytes expected)\n", bytes);
        return -1;
    }
    fclose(f);
    m->arena = src;
    m->arena_bytes = bytes;
    m->arena_mmap = 0;
    m->w = (float *)src;
    m->n_weight = bytes / sizeof(float);
    m->weight_bytes = bytes;
    m->quantized = 3; /* binary linears */
    m->vocab = NULL;
    m->locked = 0;
#if defined(__linux__)
    if (mlock(src, bytes) == 0)
        m->locked = 1;
#endif
    rope_cache_init(m->cfg.head_dim, m->cfg.rope_theta);
    {
        WV2t *map = (WV2t *)malloc(sizeof(WV2t));
        if (map) {
            map_v2_binary_fixed(map, (const unsigned char *)m->arena, &m->cfg);
            m->map_cache = map;
            m->map_cache_ready = 1;
        } else {
            m->map_cache = NULL;
            m->map_cache_ready = 0;
        }
    }
    fprintf(stderr,
            "L3B2 loaded: L%d d%d h%d/kv%d mid%d  weights=%.2f MiB (binary+int8 embed)\n",
            m->cfg.n_layer, m->cfg.n_embd, m->cfg.n_head, m->cfg.n_kv_head,
            m->cfg.intermediate, bytes / (1024.0 * 1024.0));
    return 0;
}

static void take_tmat_bytes(TMat *m, const unsigned char **p, int out, int in) {
    m->out = out;
    m->in = in;
    m->stride = tmat_stride(in);
    m->scale = (const float *)*p;
    *p += (size_t)out * sizeof(float);
    m->wp = *p;
    *p += (size_t)out * m->stride;
}

static void take_bmat_bytes(TMat *m, const unsigned char **p, int out, int in) {
    m->out = out;
    m->in = in;
    m->stride = bmat_stride(in);
    m->scale = (const float *)*p;
    *p += (size_t)out * sizeof(float);
    m->wp = *p;
    *p += (size_t)out * m->stride;
}

static void map_v2_binary_fixed(WV2t *v, const unsigned char *base, const ClmConfig *c) {
    int V = c->vocab_size, L = c->n_layer, D = c->n_embd;
    int nkv = c->n_kv_head, hd = c->head_dim, mid = c->intermediate;
    memset(v, 0, sizeof(*v));
    const float *fp = (const float *)base;
    for (int i = 0; i < L; i++) {
        v->layer[i].rms1 = take_f(&fp, D);
        v->layer[i].q_norm = take_f(&fp, hd);
        v->layer[i].k_norm = take_f(&fp, hd);
        v->layer[i].rms2 = take_f(&fp, D);
        if (c->has_ls) {
            v->layer[i].ls1 = take_f(&fp, D);
            v->layer[i].ls2 = take_f(&fp, D);
        }
        if (c->has_peri) {
            v->layer[i].post_attn = take_f(&fp, D);
            v->layer[i].post_mlp = take_f(&fp, D);
        }
    }
    v->ln_f = take_f(&fp, D);
    const unsigned char *p = (const unsigned char *)fp;
    v->wte_q = (const int8_t *)p;
    p += (size_t)V * D;
    v->wte_scale = (const float *)p;
    p += (size_t)V * sizeof(float);
    for (int i = 0; i < L; i++) {
        take_bmat_bytes(&v->layer[i].wq, &p, D, D);
        take_bmat_bytes(&v->layer[i].wk, &p, nkv * hd, D);
        take_bmat_bytes(&v->layer[i].wv, &p, nkv * hd, D);
        take_bmat_bytes(&v->layer[i].wo, &p, D, D);
        take_bmat_bytes(&v->layer[i].wgate, &p, mid, D);
        take_bmat_bytes(&v->layer[i].wup, &p, mid, D);
        take_bmat_bytes(&v->layer[i].wdown, &p, D, mid);
    }
}

static void map_v2_ternary_fixed(WV2t *v, const unsigned char *base, const ClmConfig *c) {
    int V = c->vocab_size, L = c->n_layer, D = c->n_embd;
    int nkv = c->n_kv_head, hd = c->head_dim, mid = c->intermediate;
    memset(v, 0, sizeof(*v));
    const float *fp = (const float *)base;
    for (int i = 0; i < L; i++) {
        v->layer[i].rms1 = take_f(&fp, D);
        v->layer[i].q_norm = take_f(&fp, hd);
        v->layer[i].k_norm = take_f(&fp, hd);
        v->layer[i].rms2 = take_f(&fp, D);
        if (c->has_ls) {
            v->layer[i].ls1 = take_f(&fp, D);
            v->layer[i].ls2 = take_f(&fp, D);
        }
        if (c->has_peri) {
            v->layer[i].post_attn = take_f(&fp, D);
            v->layer[i].post_mlp = take_f(&fp, D);
        }
    }
    v->ln_f = take_f(&fp, D);
    const unsigned char *p = (const unsigned char *)fp;
    v->wte_q = (const int8_t *)p;
    p += (size_t)V * D;
    v->wte_scale = (const float *)p;
    p += (size_t)V * sizeof(float);
    for (int i = 0; i < L; i++) {
        take_tmat_bytes(&v->layer[i].wq, &p, D, D);
        take_tmat_bytes(&v->layer[i].wk, &p, nkv * hd, D);
        take_tmat_bytes(&v->layer[i].wv, &p, nkv * hd, D);
        take_tmat_bytes(&v->layer[i].wo, &p, D, D);
        take_tmat_bytes(&v->layer[i].wgate, &p, mid, D);
        take_tmat_bytes(&v->layer[i].wup, &p, mid, D);
        take_tmat_bytes(&v->layer[i].wdown, &p, D, mid);
    }
}

/* ---- decode ---- */

static void v2_decode_one_f32(const ClmModel *m, ClmState *s, int t) {
    const ClmConfig *c = &m->cfg;
    int D = c->n_embd, H = c->n_head, nkv = c->n_kv_head, hd = c->head_dim;
    int mid = c->intermediate, V = c->vocab_size;
    int B = (s->kv_cap > 0) ? s->kv_cap : c->block_size;
    if (B > c->block_size)
        B = c->block_size;
    WV2f wv;
    map_v2_f32(&wv, m->w, c);

    float *x = s->x_l1 ? s->x_l1 : s->x;
    float *xb = s->xb_l1 ? s->xb_l1 : s->xb;
    float *xb2 = s->xb2_l1 ? s->xb2_l1 : s->xb2;
    float *qkv = s->qkv_l1 ? s->qkv_l1 : s->qkv;
    float *scratch = s->mlp_l1 ? s->mlp_l1 : s->mlp;

    int id = s->tokens[t];
    if (id < 0 || id >= V)
        id = 0;
    memcpy(x, wv.wte + (size_t)id * D, (size_t)D * sizeof(float));

    int kv_dim = nkv * hd;
    for (int li = 0; li < c->n_layer; li++) {
        float *kc = s->k_cache + (size_t)li * B * kv_dim;
        float *vc = s->v_cache + (size_t)li * B * kv_dim;
        rms_norm(xb, x, wv.layer[li].rms1, D);
        float *q = qkv;
        float *k = qkv + D;
        float *v = qkv + D + kv_dim;
        gemv_f32(q, xb, wv.layer[li].wq, D, D);
        gemv_f32(k, xb, wv.layer[li].wk, kv_dim, D);
        gemv_f32(v, xb, wv.layer[li].wv, kv_dim, D);
        for (int h = 0; h < H; h++) {
            float *qh = q + h * hd;
            float ms = 0.f;
            for (int i = 0; i < hd; i++)
                ms += qh[i] * qh[i];
            ms = 1.f / sqrtf(ms / (float)hd + 1e-6f);
            for (int i = 0; i < hd; i++)
                qh[i] = qh[i] * ms * wv.layer[li].q_norm[i];
        }
        for (int h = 0; h < nkv; h++) {
            float *kh = k + h * hd;
            float ms = 0.f;
            for (int i = 0; i < hd; i++)
                ms += kh[i] * kh[i];
            ms = 1.f / sqrtf(ms / (float)hd + 1e-6f);
            for (int i = 0; i < hd; i++)
                kh[i] = kh[i] * ms * wv.layer[li].k_norm[i];
        }
        for (int h = 0; h < nkv; h++)
            rope_half_inplace(k + h * hd, hd, t, c->rope_theta);
        for (int h = 0; h < H; h++)
            rope_half_inplace(q + h * hd, hd, t, c->rope_theta);
        memcpy(kc + (size_t)t * kv_dim, k, (size_t)kv_dim * sizeof(float));
        memcpy(vc + (size_t)t * kv_dim, v, (size_t)kv_dim * sizeof(float));

        memset(xb2, 0, (size_t)D * sizeof(float));
        float scale = 1.f / sqrtf((float)hd);
        float *scores = s->att_l2 ? s->att_l2 : s->att;
        for (int h = 0; h < H; h++) {
            int hkv = h / (H / nkv);
            const float *qh = q + h * hd;
            float *sc = scores + (size_t)h * B;
            float mscore = -1e30f;
            for (int u = 0; u <= t; u++) {
                const float *kh = kc + (size_t)u * kv_dim + hkv * hd;
                float dot = 0.f;
                for (int i = 0; i < hd; i++)
                    dot += qh[i] * kh[i];
                sc[u] = dot * scale;
                if (sc[u] > mscore)
                    mscore = sc[u];
            }
            float sum = 0.f;
            for (int u = 0; u <= t; u++) {
                sc[u] = expf(sc[u] - mscore);
                sum += sc[u];
            }
            float inv = 1.f / sum;
            for (int u = 0; u <= t; u++)
                sc[u] *= inv;
            float *out_h = xb2 + h * hd;
            memset(out_h, 0, (size_t)hd * sizeof(float));
            for (int u = 0; u <= t; u++) {
                const float *vh = vc + (size_t)u * kv_dim + hkv * hd;
                float a = sc[u];
                for (int i = 0; i < hd; i++)
                    out_h[i] += a * vh[i];
            }
        }
        gemv_f32(xb, xb2, wv.layer[li].wo, D, D);
        if (wv.layer[li].ls1) {
            for (int i = 0; i < D; i++)
                x[i] += xb[i] * wv.layer[li].ls1[i];
        } else {
            for (int i = 0; i < D; i++)
                x[i] += xb[i];
        }
        if (wv.layer[li].post_attn)
            rms_norm(x, x, wv.layer[li].post_attn, D);

        rms_norm(xb, x, wv.layer[li].rms2, D);
        float *gate = scratch;
        float *up = scratch + mid;
        /* fused: two gemvs still; x stays in L1 */
        gemv_f32(gate, xb, wv.layer[li].wgate, mid, D);
        gemv_f32(up, xb, wv.layer[li].wup, mid, D);
        silu_mul(gate, gate, up, mid);
        gemv_f32(xb2, gate, wv.layer[li].wdown, D, mid);
        if (wv.layer[li].ls2) {
            for (int i = 0; i < D; i++)
                x[i] += xb2[i] * wv.layer[li].ls2[i];
        } else {
            for (int i = 0; i < D; i++)
                x[i] += xb2[i];
        }
        if (wv.layer[li].post_mlp)
            rms_norm(x, x, wv.layer[li].post_mlp, D);
    }
    rms_norm(xb, x, wv.ln_f, D);
    for (int v = 0; v < V; v++) {
        const float *row = wv.wte + (size_t)v * D;
        float ssum = 0.f;
        for (int i = 0; i < D; i++)
            ssum += xb[i] * row[i];
        s->logits[v] = ssum;
    }
    if (x != s->x)
        memcpy(s->x, x, (size_t)D * sizeof(float));
}

/* Packed Linear decode: ternary (quant=2) or binary (quant=3) share int8 embeds. */
static void v2_decode_one_packed(const ClmModel *m, ClmState *s, int t, const WV2t *wv) {
    const ClmConfig *c = &m->cfg;
    const int binary = (m->quantized == 3);
    int D = c->n_embd, H = c->n_head, nkv = c->n_kv_head, hd = c->head_dim;
    int mid = c->intermediate, V = c->vocab_size;
    int B = (s->kv_cap > 0) ? s->kv_cap : c->block_size;
    if (B > c->block_size)
        B = c->block_size;

    float *x = s->x_l1 ? s->x_l1 : s->x;
    float *xb = s->xb_l1 ? s->xb_l1 : s->xb;
    float *xb2 = s->xb2_l1 ? s->xb2_l1 : s->xb2;
    float *qkv = s->qkv_l1 ? s->qkv_l1 : s->qkv;
    float *scratch = s->mlp_l1 ? s->mlp_l1 : s->mlp;

    int id = s->tokens[t];
    if (id < 0 || id >= V)
        id = 0;
    {
        const int8_t *row = wv->wte_q + (size_t)id * D;
        float sc = wv->wte_scale[id];
#if defined(__AVX512F__)
        __m512 scv = _mm512_set1_ps(sc);
        int i = 0;
        for (; i + 16 <= D; i += 16) {
            __m128i b = _mm_loadu_si128((const __m128i *)(row + i));
            __m512 xf = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(b));
            _mm512_storeu_ps(x + i, _mm512_mul_ps(xf, scv));
        }
        for (; i < D; i++)
            x[i] = (float)row[i] * sc;
#else
        for (int i = 0; i < D; i++)
            x[i] = (float)row[i] * sc;
#endif
    }

    int kv_dim = nkv * hd;
    for (int li = 0; li < c->n_layer; li++) {
        float *kc = s->k_cache + (size_t)li * B * kv_dim;
        float *vc = s->v_cache + (size_t)li * B * kv_dim;
        rms_norm(xb, x, wv->layer[li].rms1, D);
        float *q = qkv;
        float *k = qkv + D;
        float *v = qkv + D + kv_dim;
        if (binary) {
            gemv_binary(q, xb, &wv->layer[li].wq);
            gemv_binary_pair(k, v, xb, &wv->layer[li].wk, &wv->layer[li].wv);
        } else {
            gemv_ternary(q, xb, &wv->layer[li].wq);
            /* MQA: wk/wv share in-dim and out=kv_dim — stream xb once. */
            gemv_ternary_pair(k, v, xb, &wv->layer[li].wk, &wv->layer[li].wv);
        }
        for (int h = 0; h < H; h++)
            qk_norm_inplace(q + h * hd, wv->layer[li].q_norm, hd);
        for (int h = 0; h < nkv; h++)
            qk_norm_inplace(k + h * hd, wv->layer[li].k_norm, hd);
        for (int h = 0; h < nkv; h++)
            rope_half_inplace(k + h * hd, hd, t, c->rope_theta);
        for (int h = 0; h < H; h++)
            rope_half_inplace(q + h * hd, hd, t, c->rope_theta);
        memcpy(kc + (size_t)t * kv_dim, k, (size_t)kv_dim * sizeof(float));
        memcpy(vc + (size_t)t * kv_dim, v, (size_t)kv_dim * sizeof(float));

        memset(xb2, 0, (size_t)D * sizeof(float));
        float scale = 1.f / sqrtf((float)hd);
        float *scores = s->att_l2 ? s->att_l2 : s->att;
        /* MQA (nkv==1): stream each K/V position once across all heads. */
        if (nkv == 1) {
            float mscore[64]; /* H<=48 product */
            if (H > 64)
                goto attn_generic;
            for (int h = 0; h < H; h++)
                mscore[h] = -1e30f;
            for (int u = 0; u <= t; u++) {
                const float *kh = kc + (size_t)u * kv_dim;
                if (u + 1 <= t)
                    CLM_PREFETCH_X(kc + (size_t)(u + 1) * kv_dim);
                for (int h = 0; h < H; h++) {
                    float d = dot_f32(q + h * hd, kh, hd) * scale;
                    scores[(size_t)h * B + u] = d;
                    if (d > mscore[h])
                        mscore[h] = d;
                }
            }
            float inv[64];
            for (int h = 0; h < H; h++) {
                float *sc = scores + (size_t)h * B;
                float sum = 0.f;
                for (int u = 0; u <= t; u++) {
                    sc[u] = expf(sc[u] - mscore[h]);
                    sum += sc[u];
                }
                inv[h] = 1.f / sum;
            }
            /* u=0: scale-store V (skip zero); u>=1: axpy. */
            {
                const float *vh0 = vc;
                for (int h = 0; h < H; h++) {
                    float a = scores[(size_t)h * B] * inv[h];
                    float *out_h = xb2 + h * hd;
#if defined(__AVX512F__)
                    if (hd == 48) {
                        __m512 av = _mm512_set1_ps(a);
                        _mm512_storeu_ps(out_h, _mm512_mul_ps(av, _mm512_loadu_ps(vh0)));
                        _mm512_storeu_ps(out_h + 16, _mm512_mul_ps(av, _mm512_loadu_ps(vh0 + 16)));
                        _mm512_storeu_ps(out_h + 32, _mm512_mul_ps(av, _mm512_loadu_ps(vh0 + 32)));
                    } else
#endif
                    {
                        for (int i = 0; i < hd; i++)
                            out_h[i] = a * vh0[i];
                    }
                }
            }
            for (int u = 1; u <= t; u++) {
                const float *vh = vc + (size_t)u * kv_dim;
                if (u + 1 <= t)
                    CLM_PREFETCH_X(vc + (size_t)(u + 1) * kv_dim);
                for (int h = 0; h < H; h++)
                    axpy_f32(xb2 + h * hd, vh, scores[(size_t)h * B + u] * inv[h], hd);
            }
        } else {
        attn_generic:
            for (int h = 0; h < H; h++) {
                int hkv = h / (H / nkv);
                const float *qh = q + h * hd;
                float *sc = scores + (size_t)h * B;
                float mscore = -1e30f;
                for (int u = 0; u <= t; u++) {
                    const float *kh = kc + (size_t)u * kv_dim + hkv * hd;
                    if (u + 1 <= t)
                        CLM_PREFETCH_X(kc + (size_t)(u + 1) * kv_dim + hkv * hd);
                    float dot = dot_f32(qh, kh, hd);
                    sc[u] = dot * scale;
                    if (sc[u] > mscore)
                        mscore = sc[u];
                }
                float sum = 0.f;
                for (int u = 0; u <= t; u++) {
                    sc[u] = expf(sc[u] - mscore);
                    sum += sc[u];
                }
                float invs = 1.f / sum;
                float *out_h = xb2 + h * hd;
                memset(out_h, 0, (size_t)hd * sizeof(float));
                for (int u = 0; u <= t; u++) {
                    const float *vh = vc + (size_t)u * kv_dim + hkv * hd;
                    if (u + 1 <= t)
                        CLM_PREFETCH_X(vc + (size_t)(u + 1) * kv_dim + hkv * hd);
                    axpy_f32(out_h, vh, sc[u] * invs, hd);
                }
            }
        }
        if (binary)
            gemv_binary(xb, xb2, &wv->layer[li].wo);
        else
            gemv_ternary(xb, xb2, &wv->layer[li].wo);
        if (wv->layer[li].ls1)
            add_scaled_f32(x, xb, wv->layer[li].ls1, D);
        else
            add_f32(x, xb, D);
        if (wv->layer[li].post_attn)
            rms_norm(x, x, wv->layer[li].post_attn, D);

        rms_norm(xb, x, wv->layer[li].rms2, D);
        float *gate = scratch;
        float *up = scratch + mid;
        if (binary) {
            gemv_binary_pair(gate, up, xb, &wv->layer[li].wgate, &wv->layer[li].wup);
            silu_mul(gate, gate, up, mid);
            gemv_binary(xb2, gate, &wv->layer[li].wdown);
        } else {
            gemv_ternary_pair(gate, up, xb, &wv->layer[li].wgate, &wv->layer[li].wup);
            silu_mul(gate, gate, up, mid);
            gemv_ternary(xb2, gate, &wv->layer[li].wdown);
        }
        if (wv->layer[li].ls2)
            add_scaled_f32(x, xb2, wv->layer[li].ls2, D);
        else
            add_f32(x, xb2, D);
        if (wv->layer[li].post_mlp)
            rms_norm(x, x, wv->layer[li].post_mlp, D);
        /* Kick next layer weight tiles into L2 while finishing residuals. */
#if CLM_NEXT_LAYER_PREFETCH
        if (li + 1 < c->n_layer) {
            const uint8_t *nwq = wv->layer[li + 1].wq.wp;
            const uint8_t *nwk = wv->layer[li + 1].wk.wp;
            const uint8_t *nwg = wv->layer[li + 1].wgate.wp;
            CLM_PREFETCH_W(nwq);
            CLM_PREFETCH_W(nwq + CLM_W_ROW_PREFETCH);
            CLM_PREFETCH_W(nwq + 2 * CLM_W_ROW_PREFETCH);
            CLM_PREFETCH_W(nwk);
            CLM_PREFETCH_W(nwk + CLM_W_ROW_PREFETCH);
            CLM_PREFETCH_W(nwg);
            CLM_PREFETCH_W(nwg + CLM_W_ROW_PREFETCH);
            CLM_PREFETCH_W(wv->layer[li + 1].wup.wp);
        }
#endif
    }
    rms_norm(xb, x, wv->ln_f, D);
    lm_head_int8(s->logits, xb, wv->wte_q, wv->wte_scale, V, D);
    if (x != s->x)
        memcpy(s->x, x, (size_t)D * sizeof(float));
}

void clm_forward_v2(const ClmModel *m, ClmState *s) {
    int n = s->n_tokens;
    if (n < 1)
        return;
    int B = (s->kv_cap > 0) ? s->kv_cap : m->cfg.block_size;
    if (B > m->cfg.block_size)
        B = m->cfg.block_size;
    if (n > B) {
        int drop = n - B;
        memmove(s->tokens, s->tokens + drop, (size_t)B * sizeof(int));
        s->n_tokens = B;
        n = B;
        int kv_dim = m->cfg.n_kv_head * m->cfg.head_dim;
        for (int li = 0; li < m->cfg.n_layer; li++) {
            float *kc = s->k_cache + (size_t)li * B * kv_dim;
            float *vc = s->v_cache + (size_t)li * B * kv_dim;
            memmove(kc, kc + (size_t)drop * kv_dim,
                    (size_t)(B - drop) * kv_dim * sizeof(float));
            memmove(vc, vc + (size_t)drop * kv_dim,
                    (size_t)(B - drop) * kv_dim * sizeof(float));
        }
        if (s->kv_len > drop)
            s->kv_len -= drop;
        else
            s->kv_len = 0;
        s->kv_pos0 = 0;
    }
    int t0 = s->kv_len;
    if (t0 < 0 || t0 > n)
        t0 = 0;
    if (m->quantized == 2 || m->quantized == 3) {
        const WV2t *wvp;
        WV2t wv_local;
        if (m->map_cache_ready && m->map_cache) {
            wvp = (const WV2t *)m->map_cache;
        } else {
            if (m->quantized == 3)
                map_v2_binary_fixed(&wv_local, (const unsigned char *)m->arena, &m->cfg);
            else
                map_v2_ternary_fixed(&wv_local, (const unsigned char *)m->arena, &m->cfg);
            wvp = &wv_local;
        }
        for (int t = t0; t < n; t++)
            v2_decode_one_packed(m, s, t, wvp);
    } else {
        for (int t = t0; t < n; t++)
            v2_decode_one_f32(m, s, t);
    }
    s->kv_len = n;
    s->kv_pos0 = 0;
}
