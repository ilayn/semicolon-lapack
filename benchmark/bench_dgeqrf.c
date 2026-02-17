/**
 * @file bench_dgeqrf.c
 * @brief Benchmark for dgeqrf (QR factorization).
 *
 * Usage:
 *   ./bench_dgeqrf [n] [iterations]    Single size benchmark
 *   ./bench_dgeqrf --sweep             Sweep n=32..4096, print table
 *   ./bench_dgeqrf --sweep --csv       Same but CSV output
 *
 * Profiling:
 *   samply record ./bench_dgeqrf 1000 50
 */

#define _POSIX_C_SOURCE 199309L
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "semicolon_lapack_double.h"

#define DEFAULT_N 1000
#define DEFAULT_ITERS 100
#define WARMUP_ITERS 3

static const int sweep_sizes[] = {
    4, 8, 12, 16, 24, 32, 48, 64, 80, 100, 112, 128, 150, 180, 200, 240,
    256, 300, 350, 384, 400, 450, 500, 550, 600, 700, 800, 900, 1000,
    1200, 1500, 2000, 2500, 3000, 4000
};
#define N_SWEEP (int)(sizeof(sweep_sizes) / sizeof(sweep_sizes[0]))

static double bench_one(int n, int iters)
{
    int lda = n;
    size_t matrix_size = (size_t)n * n * sizeof(double);
    int info;

    double* A = malloc(matrix_size);
    double* A_orig = malloc(matrix_size);
    double* tau = malloc(n * sizeof(double));

    if (!A || !A_orig || !tau) {
        fprintf(stderr, "Memory allocation failed for n=%d\n", n);
        free(A); free(A_orig); free(tau);
        return -1.0;
    }

    /* Workspace query */
    double work_query;
    dgeqrf(n, n, A, lda, tau, &work_query, -1, &info);
    int lwork = (int)work_query;
    double* work = malloc(lwork * sizeof(double));
    if (!work) {
        fprintf(stderr, "Workspace allocation failed for n=%d\n", n);
        free(A); free(A_orig); free(tau);
        return -1.0;
    }

    srand(42);
    for (int i = 0; i < n * n; i++) {
        A_orig[i] = (double)rand() / RAND_MAX - 0.5;
    }
    for (int i = 0; i < n; i++) {
        A_orig[i + i * lda] += n;
    }

    /* Warmup */
    for (int w = 0; w < WARMUP_ITERS; w++) {
        memcpy(A, A_orig, matrix_size);
        dgeqrf(n, n, A, lda, tau, work, lwork, &info);
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int iter = 0; iter < iters; iter++) {
        memcpy(A, A_orig, matrix_size);
        dgeqrf(n, n, A, lda, tau, work, lwork, &info);
        if (info != 0 && iter == 0) {
            fprintf(stderr, "dgeqrf failed with info=%d\n", info);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;

    free(A);
    free(A_orig);
    free(tau);
    free(work);

    return elapsed;
}

static int iters_for_size(int n)
{
    if (n <= 32) return 5000;
    if (n <= 64) return 3000;
    if (n <= 128) return 2000;
    if (n <= 256) return 1000;
    if (n <= 512) return 500;
    if (n <= 1024) return 200;
    if (n <= 2048) return 50;
    return 10;
}

int main(int argc, char* argv[])
{
    int do_sweep = 0;
    int do_csv = 0;
    int n = DEFAULT_N;
    int iters = DEFAULT_ITERS;

    int positional = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sweep") == 0) {
            do_sweep = 1;
        } else if (strcmp(argv[i], "--csv") == 0) {
            do_csv = 1;
        } else {
            int val = atoi(argv[i]);
            if (val > 0) {
                if (positional == 0) { n = val; positional++; }
                else if (positional == 1) { iters = val; positional++; }
            }
        }
    }

    if (do_sweep) {
        if (do_csv) {
            printf("n,iters,time_s,time_per_iter_ms,gflops\n");
        } else {
            printf("%-8s %6s %10s %12s %10s\n",
                   "n", "iters", "total(s)", "per_iter(ms)", "GFLOP/s");
            printf("%-8s %6s %10s %12s %10s\n",
                   "---", "-----", "--------", "-----------", "-------");
        }

        for (int s = 0; s < N_SWEEP; s++) {
            int sn = sweep_sizes[s];
            int si = iters_for_size(sn);
            double elapsed = bench_one(sn, si);
            if (elapsed < 0.0) continue;

            double per_iter = elapsed / si * 1000.0;
            double gflops = (4.0 / 3.0 * sn * sn * sn) * si / elapsed / 1e9;

            if (do_csv) {
                printf("%d,%d,%.6f,%.3f,%.2f\n", sn, si, elapsed, per_iter, gflops);
            } else {
                printf("%-8d %6d %10.3f %12.3f %10.2f\n",
                       sn, si, elapsed, per_iter, gflops);
            }
            fflush(stdout);
        }
    } else {
        double elapsed = bench_one(n, iters);
        if (elapsed < 0.0) return 1;

        double per_iter = elapsed / iters * 1000.0;
        double gflops = (4.0 / 3.0 * n * n * n) * iters / elapsed / 1e9;

        if (do_csv) {
            printf("n,iters,time_s,time_per_iter_ms,gflops\n");
            printf("%d,%d,%.6f,%.3f,%.2f\n", n, iters, elapsed, per_iter, gflops);
        } else {
            printf("Benchmarking dgeqrf: n=%d, iterations=%d (warmup=%d)\n",
                   n, iters, WARMUP_ITERS);
            printf("Total time: %.3f s\n", elapsed);
            printf("Time per iteration: %.3f ms\n", per_iter);
            printf("Performance: %.2f GFLOP/s\n", gflops);
        }
    }

    return 0;
}
