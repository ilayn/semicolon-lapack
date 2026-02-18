/**
 * @file bench_dgetrf_fortran.c
 * @brief Benchmark for vendor dgetrf via Fortran LAPACK interface.
 *
 * Usage:
 *   ./bench_dgetrf_fortran [m] [n] [iters]        Single size benchmark
 *   ./bench_dgetrf_fortran --sweep [iters]         Sweep sizes, print table
 *   ./bench_dgetrf_fortran --sweep [iters] --csv   Same but CSV output
 *
 * Profiling:
 *   samply record ./bench_dgetrf_fortran 1000 1000 50
 */

#define _POSIX_C_SOURCE 199309L
#include "bench_common.h"
#include "bench_flops.h"

extern void dgetrf_(int* m, int* n, double* A, int* lda, int* ipiv, int* info);

/* ---------- Bench one configuration ---------- */

static int bench_one(int m, int n, int iters,
                     double* out_min, double* out_med)
{
    int lda = m;
    size_t matrix_bytes = (size_t)m * n * sizeof(double);
    int mindim = m < n ? m : n;
    int info;

    double* A      = (double*)malloc(matrix_bytes);
    double* A_orig = (double*)malloc(matrix_bytes);
    int*    ipiv   = (int*)malloc((size_t)mindim * sizeof(int));
    double* times  = (double*)malloc((size_t)iters * sizeof(double));

    if (!A || !A_orig || !ipiv || !times) {
        fprintf(stderr, "Memory allocation failed for m=%d, n=%d\n", m, n);
        free(A); free(A_orig); free(ipiv); free(times);
        return -1;
    }

    fill_random(A_orig, m, n, lda, 42);

    /* Warmup */
    for (int w = 0; w < WARMUP_ITERS; w++) {
        copy_matrix(A_orig, A, m, n, lda);
        dgetrf_(&m, &n, A, &lda, ipiv, &info);
    }

    /* Timed iterations */
    for (int iter = 0; iter < iters; iter++) {
        copy_matrix(A_orig, A, m, n, lda);
        flush_cache();
        double t0 = get_wtime();
        dgetrf_(&m, &n, A, &lda, ipiv, &info);
        double t1 = get_wtime();
        times[iter] = t1 - t0;
        if (info != 0 && iter == 0)
            fprintf(stderr, "dgetrf_ failed with info=%d\n", info);
    }

    compute_stats(times, iters, out_min, out_med);

    free(A);
    free(A_orig);
    free(ipiv);
    free(times);
    return 0;
}

/* ---------- Main ---------- */

int main(int argc, char* argv[])
{
    int do_sweep = 0, do_csv = 0;
    int m = DEFAULT_M, n = DEFAULT_N, iters = DEFAULT_ITERS;

    int pos[3] = {0, 0, 0};
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sweep") == 0)       { do_sweep = 1; }
        else if (strcmp(argv[i], "--csv") == 0)     { do_csv = 1; }
        else {
            int val = atoi(argv[i]);
            if (val > 0 && npos < 3) { pos[npos++] = val; }
        }
    }
    /* 1 arg: n.  2 args: n iters.  3 args: m n iters. */
    if (npos == 1)      { m = pos[0]; n = pos[0]; }
    else if (npos == 2) { m = pos[0]; n = pos[0]; iters = pos[1]; }
    else if (npos == 3) { m = pos[0]; n = pos[1]; iters = pos[2]; }

    if (do_sweep) {
        if (npos > 1) {
            fprintf(stderr, "Sweep does not take extra parameters."
                    " Usage: %s --sweep [iters]\n", argv[0]);
            return 1;
        }
        int sweep_iters_override = (npos == 1) ? pos[0] : 0;

        if (do_csv) {
            printf("m,n,iters,min_ms,med_ms,min_gflops,med_gflops\n");
        } else {
            printf("%-6s %-6s %5s %9s %9s %10s %10s\n",
                   "m", "n", "iters", "min(ms)", "med(ms)",
                   "min(GF/s)", "med(GF/s)");
            printf("%-6s %-6s %5s %9s %9s %10s %10s\n",
                   "---", "---", "-----", "-------", "-------",
                   "---------", "---------");
        }

        for (int s = 0; s < N_SWEEP; s++) {
            int sm = sweep_shapes[s].m;
            int sn = sweep_shapes[s].n;
            int si = sweep_iters_override > 0
                         ? sweep_iters_override
                         : iters_for_size(sm, sn);
            double t_min, t_med;
            if (bench_one(sm, sn, si, &t_min, &t_med) < 0)
                continue;

            double flops = getrf_flops(sm, sn);
            double gf_min = flops / t_min / 1e9;
            double gf_med = flops / t_med / 1e9;

            if (do_csv) {
                printf("%d,%d,%d,%.4f,%.4f,%.2f,%.2f\n",
                       sm, sn, si, t_min * 1e3, t_med * 1e3, gf_min, gf_med);
            } else {
                printf("%-6d %-6d %5d %9.3f %9.3f %10.2f %10.2f\n",
                       sm, sn, si, t_min * 1e3, t_med * 1e3, gf_min, gf_med);
            }
            fflush(stdout);
        }
    } else {
        double t_min, t_med;
        if (bench_one(m, n, iters, &t_min, &t_med) != 0)
            return 1;

        double flops = getrf_flops(m, n);
        double gf_min = flops / t_min / 1e9;
        double gf_med = flops / t_med / 1e9;

        if (do_csv) {
            printf("m,n,iters,min_ms,med_ms,min_gflops,med_gflops\n");
            printf("%d,%d,%d,%.4f,%.4f,%.2f,%.2f\n",
                   m, n, iters, t_min * 1e3, t_med * 1e3, gf_min, gf_med);
        } else {
            printf("Benchmarking vendor dgetrf_: m=%d, n=%d, iters=%d (warmup=%d)\n",
                   m, n, iters, WARMUP_ITERS);
            printf("Min time:    %.3f ms  (%.2f GFLOP/s)\n",
                   t_min * 1e3, gf_min);
            printf("Median time: %.3f ms  (%.2f GFLOP/s)\n",
                   t_med * 1e3, gf_med);
        }
    }

    return 0;
}
