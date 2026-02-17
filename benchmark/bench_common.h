/**
 * @file bench_common.h
 * @brief Shared utilities for semicolon-lapack benchmarks.
 *
 * Provides: high-resolution timer, cache flushing, statistics (min/median),
 * matrix generation, adaptive iteration counts, and sweep shape definitions.
 */

#ifndef BENCH_COMMON_H
#define BENCH_COMMON_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <float.h>

/* ---------- High-resolution wall-clock timer ---------- */

static inline double get_wtime(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------- Cache flushing ---------- */

#define FLUSH_CACHE_BYTES (32 * 1024 * 1024)  /* 32 MB */

static volatile double bench_flush_sink_;

static inline void flush_cache(void)
{
    static double* buf = NULL;
    size_t n = FLUSH_CACHE_BYTES / sizeof(double);
    if (!buf) {
        buf = (double*)malloc(FLUSH_CACHE_BYTES);
        if (!buf) return;
    }
    for (size_t i = 0; i < n; i += 8)
        buf[i] = 1.0;
    bench_flush_sink_ = buf[0];
}

/* ---------- Statistics ---------- */

static int bench_cmp_double_(const void* a, const void* b)
{
    double da = *(const double*)a;
    double db = *(const double*)b;
    return (da > db) - (da < db);
}

/**
 * Compute min and median of a times array (sorts in-place).
 */
static inline void compute_stats(double* times, int n,
                                 double* t_min, double* t_med)
{
    qsort(times, (size_t)n, sizeof(double), bench_cmp_double_);
    *t_min = times[0];
    *t_med = (n % 2 == 1)
        ? times[n / 2]
        : 0.5 * (times[n / 2 - 1] + times[n / 2]);
}

/* ---------- Matrix utilities ---------- */

/**
 * Fill an m-by-n column-major matrix with random values in [-0.5, 0.5]
 * using rand(), then boost the diagonal by max(m,n) for conditioning.
 */
static inline void fill_random(double* A, int m, int n, int lda,
                               unsigned int seed)
{
    srand(seed);
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            A[i + (size_t)j * lda] = (double)rand() / RAND_MAX - 0.5;

    /* Diagonal boost for conditioning */
    int mindim = m < n ? m : n;
    int maxdim = m > n ? m : n;
    for (int i = 0; i < mindim; i++)
        A[i + (size_t)i * lda] += maxdim;
}

/**
 * Column-major matrix copy respecting leading dimension.
 */
static inline void copy_matrix(const double* src, double* dst,
                               int m, int n, int lda)
{
    for (int j = 0; j < n; j++)
        memcpy(dst + (size_t)j * lda, src + (size_t)j * lda,
               (size_t)m * sizeof(double));
}

/* ---------- Adaptive iteration counts ---------- */

static inline int iters_for_size(int m, int n)
{
    int maxdim = m > n ? m : n;
    if (maxdim <= 32)   return 5000;
    if (maxdim <= 64)   return 3000;
    if (maxdim <= 128)  return 2000;
    if (maxdim <= 256)  return 1000;
    if (maxdim <= 512)  return 500;
    if (maxdim <= 1024) return 200;
    if (maxdim <= 2048) return 50;
    return 10;
}

/* ---------- Sweep shapes ---------- */

typedef struct { int m; int n; } bench_shape_t;

static const bench_shape_t sweep_shapes[] = {
    /* Square matrices (dense coverage for smooth performance curves) */
    {4, 4}, {8, 8}, {12, 12}, {16, 16}, {24, 24}, {32, 32},
    {48, 48}, {64, 64}, {80, 80}, {100, 100}, {112, 112}, {128, 128},
    {150, 150}, {180, 180}, {200, 200}, {240, 240}, {256, 256},
    {300, 300}, {350, 350}, {384, 384}, {400, 400}, {450, 450},
    {500, 500}, {550, 550}, {600, 600}, {700, 700}, {800, 800},
    {900, 900}, {1000, 1000}, {1200, 1200}, {1500, 1500},
    {2000, 2000}, {2500, 2500}, {3000, 3000}, {4000, 4000},
    /* Tall-skinny */
    {1000, 100}, {2000, 200}, {4000, 400}, {4000, 100},
    /* Short-wide */
    {100, 1000}, {200, 2000}, {400, 4000},
};
#define N_SWEEP (int)(sizeof(sweep_shapes) / sizeof(sweep_shapes[0]))

/* ---------- Common defaults ---------- */

#define DEFAULT_M    1000
#define DEFAULT_N    1000
#define DEFAULT_ITERS 100
#define WARMUP_ITERS  3

#endif /* BENCH_COMMON_H */
