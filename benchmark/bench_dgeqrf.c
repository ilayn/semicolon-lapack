/**
 * @file bench_dgeqrf.c
 * @brief Benchmark for dgeqrf (QR factorization) — semicolon-lapack.
 *
 * Usage:
 *   ./bench_dgeqrf [m] [n] [iters]        Single size benchmark
 *   ./bench_dgeqrf --sweep [iters]         Sweep sizes, print table
 *   ./bench_dgeqrf --sweep [iters] --csv   Same but CSV output
 *   ./bench_dgeqrf [m] [n] [iters] --check Run correctness check
 *
 * Profiling:
 *   samply record ./bench_dgeqrf 1000 1000 50
 */

#define _POSIX_C_SOURCE 199309L
#include "bench_common.h"
#include "bench_flops.h"
#include "semicolon_lapack_double.h"

/* ---------- Correctness check ---------- */

/**
 * Check ||A - Q*R|| / (max(m,n) * ||A|| * eps).
 *
 * On entry, A_qr holds the dgeqrf output (R in upper triangle,
 * Householder vectors below).  A_orig holds the original matrix.
 */
static double check_geqrf(int m, int n, int lda,
                           const double* A_qr, const double* tau,
                           const double* A_orig)
{
    int mindim = m < n ? m : n;
    int maxdim = m > n ? m : n;
    int info;

    /* Workspace query for dormqr */
    double wq;
    int lwork = -1;
    dormqr("L", "N", m, n, mindim,
           A_qr, lda, tau, NULL, lda, &wq, lwork, &info);
    lwork = (int)wq;
    double* work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) return -1.0;

    /* Extract R (upper triangle of A_qr) into C, zero below diagonal */
    double* C = (double*)malloc((size_t)lda * n * sizeof(double));
    if (!C) { free(work); return -1.0; }
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            if (i <= j)
                C[i + (size_t)j * lda] = A_qr[i + (size_t)j * lda];
            else
                C[i + (size_t)j * lda] = 0.0;
        }
    }

    /* C = Q * R  via  dormqr("L", "N", ...) */
    dormqr("L", "N", m, n, mindim,
           A_qr, lda, tau, C, lda, work, lwork, &info);

    /* C = C - A_orig */
    for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
            C[i + (size_t)j * lda] -= A_orig[i + (size_t)j * lda];

    /* ||C||_F */
    double* work_norm = (double*)malloc((size_t)m * sizeof(double));
    double res_norm = dlange("F", m, n, C, lda, work_norm);

    /* ||A_orig||_F */
    double a_norm = dlange("F", m, n, A_orig, lda, work_norm);

    free(work_norm);
    free(C);
    free(work);

    return res_norm / (maxdim * a_norm * DBL_EPSILON);
}

/* ---------- Bench one configuration ---------- */

static int bench_one(int m, int n, int iters, int do_check,
                     double* out_min, double* out_med)
{
    int lda = m;
    size_t matrix_bytes = (size_t)m * n * sizeof(double);
    int mindim = m < n ? m : n;
    int info;

    double* A      = (double*)malloc(matrix_bytes);
    double* A_orig = (double*)malloc(matrix_bytes);
    double* tau    = (double*)malloc((size_t)mindim * sizeof(double));
    double* times  = (double*)malloc((size_t)iters * sizeof(double));

    if (!A || !A_orig || !tau || !times) {
        fprintf(stderr, "Memory allocation failed for m=%d, n=%d\n", m, n);
        free(A); free(A_orig); free(tau); free(times);
        return -1;
    }

    /* Workspace query */
    double work_query;
    dgeqrf(m, n, A, lda, tau, &work_query, -1, &info);
    int lwork = (int)work_query;
    double* work = (double*)malloc((size_t)lwork * sizeof(double));
    if (!work) {
        fprintf(stderr, "Workspace allocation failed for m=%d, n=%d\n", m, n);
        free(A); free(A_orig); free(tau); free(times);
        return -1;
    }

    fill_random(A_orig, m, n, lda, 42);

    /* Warmup */
    for (int w = 0; w < WARMUP_ITERS; w++) {
        copy_matrix(A_orig, A, m, n, lda);
        dgeqrf(m, n, A, lda, tau, work, lwork, &info);
    }

    /* Timed iterations */
    for (int iter = 0; iter < iters; iter++) {
        copy_matrix(A_orig, A, m, n, lda);
        flush_cache();
        double t0 = get_wtime();
        dgeqrf(m, n, A, lda, tau, work, lwork, &info);
        double t1 = get_wtime();
        times[iter] = t1 - t0;
        if (info != 0 && iter == 0)
            fprintf(stderr, "dgeqrf failed with info=%d\n", info);
    }

    compute_stats(times, iters, out_min, out_med);

    /* Correctness check (on last factorization result) */
    if (do_check) {
        double resid = check_geqrf(m, n, lda, A, tau, A_orig);
        fprintf(stderr, "  check: ||A-QR||/(max(m,n)*||A||*eps) = %.2e %s\n",
                resid, resid < 10.0 ? "PASS" : "FAIL");
    }

    free(A);
    free(A_orig);
    free(tau);
    free(work);
    free(times);
    return 0;
}

/* ---------- Main ---------- */

int main(int argc, char* argv[])
{
    int do_sweep = 0, do_csv = 0, do_check = 0;
    int m = DEFAULT_M, n = DEFAULT_N, iters = DEFAULT_ITERS;

    int pos[3] = {0, 0, 0};
    int npos = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--sweep") == 0)       { do_sweep = 1; }
        else if (strcmp(argv[i], "--csv") == 0)     { do_csv = 1; }
        else if (strcmp(argv[i], "--check") == 0)   { do_check = 1; }
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

        /* In sweep mode, a positional argument overrides default iters */
        int sweep_iters_override = (npos >= 1) ? pos[0] : 0;

        for (int s = 0; s < N_SWEEP; s++) {
            int sm = sweep_shapes[s].m;
            int sn = sweep_shapes[s].n;
            int si = sweep_iters_override > 0
                         ? sweep_iters_override
                         : iters_for_size(sm, sn);
            double t_min, t_med;
            if (bench_one(sm, sn, si, do_check, &t_min, &t_med) < 0)
                continue;

            double flops = geqrf_flops(sm, sn);
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
        if (bench_one(m, n, iters, do_check, &t_min, &t_med) != 0)
            return 1;

        double flops = geqrf_flops(m, n);
        double gf_min = flops / t_min / 1e9;
        double gf_med = flops / t_med / 1e9;

        if (do_csv) {
            printf("m,n,iters,min_ms,med_ms,min_gflops,med_gflops\n");
            printf("%d,%d,%d,%.4f,%.4f,%.2f,%.2f\n",
                   m, n, iters, t_min * 1e3, t_med * 1e3, gf_min, gf_med);
        } else {
            printf("Benchmarking dgeqrf: m=%d, n=%d, iters=%d (warmup=%d)\n",
                   m, n, iters, WARMUP_ITERS);
            printf("Min time:    %.3f ms  (%.2f GFLOP/s)\n",
                   t_min * 1e3, gf_min);
            printf("Median time: %.3f ms  (%.2f GFLOP/s)\n",
                   t_med * 1e3, gf_med);
        }
    }

    return 0;
}
