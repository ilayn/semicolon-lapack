/**
 * @file zhpevx_debug.c
 * @brief Side-by-side comparison of our zhpevx vs OpenBLAS Fortran zhpevx_
 *        for the pathological jtype=10 case (eigenvalues {1, ulp, ulp}).
 *
 * Phase 1: Sweeps 1000 random unitary matrices Q, constructs A = Q*D*Q^H with
 *          D = diag(ulp, ulp, 1), calls both our and OpenBLAS zhpevx, and
 *          tracks the worst-case orthogonality residual for each.
 *
 * Phase 2: For the worst-case matrix, reruns stage-by-stage comparison:
 *          zhptrd -> dstebz -> zstein -> zupmtr
 *
 * Build (CI, both Linux and macOS):
 *   ${CC:-cc} -O2 -g tests/ci/zhpevx_debug.c \
 *     -I src/include -I include \
 *     $(pkg-config --cflags --libs openblas) \
 *     -Lbuilddir/src -lsemilapack \
 *     -lm -Wl,-rpath,$CONDA_PREFIX/lib -Wl,-rpath,$(pwd)/builddir/src \
 *     -o zhpevx_debug
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <float.h>

#include "semicolon_lapack_complex_double.h"

/* OpenBLAS Fortran LAPACK symbols */
extern void zhptrd_(const char* uplo, const int* n,
                    double _Complex* AP, double* D, double* E,
                    double _Complex* tau, int* info);

extern void dstebz_(const char* range, const char* order, const int* n,
                    const double* vl, const double* vu,
                    const int* il, const int* iu, const double* abstol,
                    const double* D, const double* E,
                    int* m, int* nsplit, double* W,
                    int* iblock, int* isplit,
                    double* work, int* iwork, int* info);

extern void zstein_(const int* n, const double* D, const double* E,
                    const int* m, const double* W,
                    const int* iblock, const int* isplit,
                    double _Complex* Z, const int* ldz,
                    double* work, int* iwork, int* ifail, int* info);

extern void zupmtr_(const char* side, const char* uplo, const char* trans,
                    const int* m, const int* n,
                    const double _Complex* AP, const double _Complex* tau,
                    double _Complex* C, const int* ldc,
                    double _Complex* work, int* info);

extern void zhpevx_(const char* jobz, const char* range, const char* uplo,
                    const int* n, double _Complex* AP,
                    const double* vl, const double* vu,
                    const int* il, const int* iu,
                    const double* abstol, int* m, double* W,
                    double _Complex* Z, const int* ldz,
                    double _Complex* work, double* rwork,
                    int* iwork, int* ifail, int* info);

#define N 3
#define LDZ N
#define NPACK (N*(N+1)/2)
#define NTRIALS 1000

static void print_vec_d(const char* label, const double* v, int len)
{
    printf("  %s:", label);
    for (int i = 0; i < len; i++)
        printf(" %22.15e", v[i]);
    printf("\n");
}

static void print_vec_z(const char* label, const double _Complex* v, int len)
{
    printf("  %s:\n", label);
    for (int i = 0; i < len; i++)
        printf("    [%d] (%22.15e, %22.15e)\n", i, creal(v[i]), cimag(v[i]));
}

static void print_mat_z(const char* label, const double _Complex* Z, int ld,
                        int rows, int cols)
{
    printf("  %s (%dx%d):\n", label, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("    row %d:", i);
        for (int j = 0; j < cols; j++)
            printf(" (%12.9f,%12.9f)", creal(Z[i + j * ld]), cimag(Z[i + j * ld]));
        printf("\n");
    }
}

static void print_vec_i(const char* label, const int* v, int len)
{
    printf("  %s:", label);
    for (int i = 0; i < len; i++)
        printf(" %d", v[i]);
    printf("\n");
}

static double orthogonality_residual(const double _Complex* Z, int n, int m, int ldz)
{
    double ulp = DBL_EPSILON;
    double maxerr = 0.0;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < m; j++) {
            double _Complex dot = 0.0;
            for (int k = 0; k < n; k++)
                dot += conj(Z[k + i * ldz]) * Z[k + j * ldz];
            double expected = (i == j) ? 1.0 : 0.0;
            double err = cabs(dot - expected);
            if (err > maxerr) maxerr = err;
        }
    }
    return maxerr / ((double)n * ulp);
}

/* Simple xoshiro256+ for deterministic random numbers */
static uint64_t rng_state[4];

static void rng_init(uint64_t seed)
{
    /* SplitMix64 seeding */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = z ^ (z >> 31);
    }
}

static uint64_t rng_next(void)
{
    uint64_t result = rng_state[0] + rng_state[3];
    uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = (rng_state[3] << 45) | (rng_state[3] >> 19);
    return result;
}

static double rng_uniform(void)
{
    return (double)(rng_next() >> 11) * 0x1.0p-53;
}

static void make_hermitian_matrix(double _Complex* AP_save,
                                  double angles[6])
{
    double evals[N] = {DBL_EPSILON, DBL_EPSILON, 1.0};

    double _Complex Q[N * N];
    memset(Q, 0, sizeof(Q));
    Q[0] = 1.0; Q[4] = 1.0; Q[8] = 1.0;

    /* Rotation in (0,1) plane */
    double c = cos(angles[0]), s = sin(angles[0]);
    for (int j = 0; j < N; j++) {
        double _Complex t0 = Q[0 + j * N], t1 = Q[1 + j * N];
        Q[0 + j * N] =  c * t0 + s * t1;
        Q[1 + j * N] = -s * t0 + c * t1;
    }
    /* Rotation in (1,2) plane */
    c = cos(angles[1]); s = sin(angles[1]);
    for (int j = 0; j < N; j++) {
        double _Complex t1 = Q[1 + j * N], t2 = Q[2 + j * N];
        Q[1 + j * N] =  c * t1 + s * t2;
        Q[2 + j * N] = -s * t1 + c * t2;
    }
    /* Complex rotation in (0,2) plane */
    {
        double _Complex phase = CMPLX(cos(angles[2]), sin(angles[2]));
        c = cos(angles[3]); s = sin(angles[3]);
        for (int j = 0; j < N; j++) {
            double _Complex t0 = Q[0 + j * N], t2 = Q[2 + j * N];
            Q[0 + j * N] =  c * t0 + s * phase * t2;
            Q[2 + j * N] = -s * conj(phase) * t0 + c * t2;
        }
    }
    /* Complex rotation in (0,1) plane */
    {
        double _Complex phase = CMPLX(cos(angles[4]), sin(angles[4]));
        c = cos(angles[5]); s = sin(angles[5]);
        for (int j = 0; j < N; j++) {
            double _Complex t0 = Q[0 + j * N], t1 = Q[1 + j * N];
            Q[0 + j * N] =  c * t0 + s * phase * t1;
            Q[1 + j * N] = -s * conj(phase) * t0 + c * t1;
        }
    }

    /* A = Q * diag(evals) * Q^H, packed lower triangle */
    int idx = 0;
    for (int j = 0; j < N; j++) {
        for (int i = j; i < N; i++) {
            double _Complex sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += Q[i + k * N] * evals[k] * conj(Q[j + k * N]);
            if (i == j) sum = creal(sum);
            AP_save[idx++] = sum;
        }
    }
}

static double run_zhpevx_ours(const double _Complex* AP_save,
                              double _Complex* Z_out, double* W_out, int* m_out)
{
    double abstol = DBL_MIN + DBL_MIN;
    double _Complex AP[NPACK];
    double _Complex work[2 * N];
    double rwork[7 * N];
    int iwork[5 * N], ifail[N], info;

    memcpy(AP, AP_save, sizeof(AP));
    zhpevx("V", "A", "L", N, AP, 0.0, 0.0, 0, N - 1,
           abstol, m_out, W_out, Z_out, LDZ,
           work, rwork, iwork, ifail, &info);
    return orthogonality_residual(Z_out, N, *m_out, LDZ);
}

static double run_zhpevx_obl(const double _Complex* AP_save,
                             double _Complex* Z_out, double* W_out, int* m_out)
{
    double abstol = DBL_MIN + DBL_MIN;
    double _Complex AP[NPACK];
    double _Complex work[2 * N];
    double rwork[7 * N];
    int iwork[5 * N], ifail[N], info;
    int fn = N, fldz = LDZ, fil = 1, fiu = N;
    double fvl = 0.0, fvu = 0.0;

    memcpy(AP, AP_save, sizeof(AP));
    zhpevx_("V", "A", "L", &fn, AP,
            &fvl, &fvu, &fil, &fiu,
            &abstol, m_out, W_out, Z_out, &fldz,
            work, rwork, iwork, ifail, &info);
    return orthogonality_residual(Z_out, N, *m_out, LDZ);
}

static void detailed_comparison(const double _Complex* AP_save)
{
    double abstol = DBL_MIN + DBL_MIN;
    double _Complex AP_ours[NPACK], AP_obl[NPACK];
    double d_ours[N], e_ours[N], d_obl[N], e_obl[N];
    double _Complex tau_ours[N], tau_obl[N];
    int iinfo, fn = N;

    printf("\n========================================\n");
    printf("=== Stage 1: zhptrd ===\n");
    printf("========================================\n");
    memcpy(AP_ours, AP_save, NPACK * sizeof(double _Complex));
    zhptrd("L", N, AP_ours, d_ours, e_ours, tau_ours, &iinfo);
    printf("[ours]  info=%d\n", iinfo);
    print_vec_d("d", d_ours, N);
    print_vec_d("e", e_ours, N - 1);
    print_vec_z("tau", tau_ours, N - 1);

    memcpy(AP_obl, AP_save, NPACK * sizeof(double _Complex));
    zhptrd_("L", &fn, AP_obl, d_obl, e_obl, tau_obl, &iinfo);
    printf("[oblas] info=%d\n", iinfo);
    print_vec_d("d", d_obl, N);
    print_vec_d("e", e_obl, N - 1);
    print_vec_z("tau", tau_obl, N - 1);

    printf("  d diff:");
    for (int i = 0; i < N; i++) printf(" %.3e", fabs(d_ours[i] - d_obl[i]));
    printf("\n  e diff:");
    for (int i = 0; i < N - 1; i++) printf(" %.3e", fabs(e_ours[i] - e_obl[i]));
    printf("\n");

    printf("\n========================================\n");
    printf("=== Stage 2: dstebz ===\n");
    printf("========================================\n");
    double w_ours[N], w_obl[N];
    int iblock_ours[N], isplit_ours[N], iblock_obl[N], isplit_obl[N];
    int m_ours, m_obl, nsplit_ours, nsplit_obl;
    double rwork_s[4 * N]; int iwork_s[3 * N];

    dstebz("A", "B", N, 0.0, 0.0, 0, N - 1, abstol,
           d_ours, e_ours, &m_ours, &nsplit_ours, w_ours,
           iblock_ours, isplit_ours, rwork_s, iwork_s, &iinfo);
    printf("[ours]  m=%d, nsplit=%d\n", m_ours, nsplit_ours);
    print_vec_d("w", w_ours, m_ours);
    print_vec_i("iblock", iblock_ours, m_ours);
    print_vec_i("isplit", isplit_ours, nsplit_ours);

    int fil = 1, fiu = N;
    dstebz_("A", "B", &fn, &(double){0.0}, &(double){0.0},
            &fil, &fiu, &abstol,
            d_obl, e_obl, &m_obl, &nsplit_obl, w_obl,
            iblock_obl, isplit_obl, rwork_s, iwork_s, &iinfo);
    printf("[oblas] m=%d, nsplit=%d\n", m_obl, nsplit_obl);
    print_vec_d("w", w_obl, m_obl);
    print_vec_i("iblock", iblock_obl, m_obl);
    print_vec_i("isplit", isplit_obl, nsplit_obl);

    printf("  w diff:");
    int mmin = m_ours < m_obl ? m_ours : m_obl;
    for (int i = 0; i < mmin; i++) printf(" %.3e", fabs(w_ours[i] - w_obl[i]));
    printf("\n");

    printf("\n========================================\n");
    printf("=== Stage 3: zstein ===\n");
    printf("========================================\n");
    double _Complex Z_ours[N * N], Z_obl[N * N];
    double rwork_st[5 * N]; int iwork_st[N], ifail_ours[N], ifail_obl[N];
    memset(Z_ours, 0, sizeof(Z_ours));
    memset(Z_obl, 0, sizeof(Z_obl));

    zstein(N, d_ours, e_ours, m_ours, w_ours,
           iblock_ours, isplit_ours, Z_ours, LDZ,
           rwork_st, iwork_st, ifail_ours, &iinfo);
    printf("[ours]  info=%d\n", iinfo);
    print_mat_z("Z_pre", Z_ours, LDZ, N, m_ours);
    printf("  ortho: %.6e\n", orthogonality_residual(Z_ours, N, m_ours, LDZ));

    int fldz = LDZ;
    zstein_(&fn, d_obl, e_obl, &m_obl, w_obl,
            iblock_obl, isplit_obl, Z_obl, &fldz,
            rwork_st, iwork_st, ifail_obl, &iinfo);
    printf("[oblas] info=%d\n", iinfo);
    print_mat_z("Z_pre", Z_obl, LDZ, N, m_obl);
    printf("  ortho: %.6e\n", orthogonality_residual(Z_obl, N, m_obl, LDZ));

    printf("\n========================================\n");
    printf("=== Stage 4: zupmtr ===\n");
    printf("========================================\n");
    double _Complex work_u[N];
    zupmtr("L", "L", "N", N, m_ours, AP_ours, tau_ours, Z_ours, LDZ,
           work_u, &iinfo);
    printf("[ours]  info=%d\n", iinfo);
    print_mat_z("Z_post", Z_ours, LDZ, N, m_ours);
    printf("  ortho: %.6e\n", orthogonality_residual(Z_ours, N, m_ours, LDZ));

    int fm = N, fmm = m_obl;
    zupmtr_("L", "L", "N", &fm, &fmm, AP_obl, tau_obl, Z_obl, &fldz,
            work_u, &iinfo);
    printf("[oblas] info=%d\n", iinfo);
    print_mat_z("Z_post", Z_obl, LDZ, N, m_obl);
    printf("  ortho: %.6e\n", orthogonality_residual(Z_obl, N, m_obl, LDZ));
}

int main(void)
{
    printf("=== zhpevx_debug: sweep %d random matrices ===\n", NTRIALS);
    printf("N = %d, eigenvalue distribution: {ulp, ulp, 1.0}\n", N);
    printf("abstol = 2*DBL_MIN (forces dstebz+zstein path)\n\n");

    rng_init(0xA37B1C924E68F05DULL);

    double worst_ours = 0.0, worst_obl = 0.0;
    double _Complex worst_AP_ours[NPACK], worst_AP_obl[NPACK];
    int worst_trial_ours = -1, worst_trial_obl = -1;
    double max_diff = 0.0;

    for (int trial = 0; trial < NTRIALS; trial++) {
        double angles[6];
        for (int i = 0; i < 6; i++)
            angles[i] = rng_uniform() * 2.0 * M_PI;

        double _Complex AP_save[NPACK];
        make_hermitian_matrix(AP_save, angles);

        double _Complex Z_ours[N * N], Z_obl[N * N];
        double W_ours[N], W_obl[N];
        int m_ours, m_obl;

        double res_ours = run_zhpevx_ours(AP_save, Z_ours, W_ours, &m_ours);
        double res_obl  = run_zhpevx_obl(AP_save, Z_obl, W_obl, &m_obl);

        double diff = fabs(res_ours - res_obl);
        if (diff > max_diff) max_diff = diff;

        if (res_ours > worst_ours) {
            worst_ours = res_ours;
            worst_trial_ours = trial;
            memcpy(worst_AP_ours, AP_save, sizeof(AP_save));
        }
        if (res_obl > worst_obl) {
            worst_obl = res_obl;
            worst_trial_obl = trial;
            memcpy(worst_AP_obl, AP_save, sizeof(AP_save));
        }

        if (res_ours > 30.0 || res_obl > 30.0 || diff > 1.0) {
            printf("  trial %4d: ours=%.3e  oblas=%.3e  diff=%.3e%s\n",
                   trial, res_ours, res_obl, diff,
                   (res_ours > 50.0 || res_obl > 50.0) ? "  *** FAIL ***" : "");
        }
    }

    printf("\n========================================\n");
    printf("=== SWEEP SUMMARY ===\n");
    printf("========================================\n");
    printf("  Trials: %d\n", NTRIALS);
    printf("  Worst [ours]:  %.6e (trial %d)%s\n",
           worst_ours, worst_trial_ours, worst_ours > 50.0 ? " FAIL" : " PASS");
    printf("  Worst [oblas]: %.6e (trial %d)%s\n",
           worst_obl, worst_trial_obl, worst_obl > 50.0 ? " FAIL" : " PASS");
    printf("  Max |ours-oblas| diff: %.6e\n", max_diff);
    printf("\n");

    if (worst_ours > 20.0 || worst_obl > 20.0) {
        printf("========================================\n");
        printf("=== DETAILED STAGE-BY-STAGE for worst [ours] matrix ===\n");
        printf("========================================\n");
        printf("  (trial %d, residual %.6e)\n", worst_trial_ours, worst_ours);
        print_vec_z("AP_save", worst_AP_ours, NPACK);
        detailed_comparison(worst_AP_ours);

        if (worst_trial_obl != worst_trial_ours) {
            printf("\n========================================\n");
            printf("=== DETAILED STAGE-BY-STAGE for worst [oblas] matrix ===\n");
            printf("========================================\n");
            printf("  (trial %d, residual %.6e)\n", worst_trial_obl, worst_obl);
            print_vec_z("AP_save", worst_AP_obl, NPACK);
            detailed_comparison(worst_AP_obl);
        }
    } else {
        printf("All residuals below 20 — no detailed comparison needed.\n");
    }

    return 0;
}
