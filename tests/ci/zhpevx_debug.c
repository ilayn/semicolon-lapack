/**
 * @file zhpevx_debug.c
 * @brief Stage-by-stage comparison of our zhpevx pipeline vs OpenBLAS Fortran
 *        for the exact failing matrix from zdrvst n=3 jtype=10 on macOS ARM64.
 *
 * Runs: zhptrd -> dstebz -> zstein -> zupmtr, comparing our C implementation
 * against OpenBLAS's Fortran symbols at each stage.
 *
 * Build:
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

/* OpenBLAS Fortran LAPACK symbols (all args by pointer, 1-based) */
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

/* Exact packed lower-triangle matrix from macOS ARM64 CI dump */
static const double _Complex AP_MACOS[NPACK] = {
    4.91800740771642708e-01 + 0.0 * I,
    4.26710053799474409e-01 - 4.33152203490123830e-02 * I,
    2.54303979114093304e-01 - 3.61189704644434770e-02 * I,
    3.74049209520915527e-01 + 0.0 * I,
    2.23827572119812329e-01 - 8.94080584792857243e-03 * I,
    1.34150049707441515e-01 + 0.0 * I
};

int main(void)
{
    double abstol = DBL_MIN + DBL_MIN;
    int fn = N, fldz = LDZ;

    printf("=== zhpevx_debug: stage-by-stage comparison ===\n");
    printf("N = %d, using exact matrix from macOS CI zdrvst n=3 jtype=10\n", N);
    printf("abstol = 2*DBL_MIN = %.6e\n\n", abstol);

    printf("Input AP (packed lower):\n");
    print_vec_z("AP", AP_MACOS, NPACK);

    /* ================================================================ */
    /* Stage 1: ZHPTRD                                                  */
    /* ================================================================ */
    printf("\n========================================\n");
    printf("=== Stage 1: ZHPTRD ===\n");
    printf("========================================\n");

    double _Complex AP_ours[NPACK], AP_obl[NPACK];
    double d_ours[N], e_ours[N], d_obl[N], e_obl[N];
    double _Complex tau_ours[N], tau_obl[N];
    int iinfo;

    memcpy(AP_ours, AP_MACOS, sizeof(AP_ours));
    zhptrd("L", N, AP_ours, d_ours, e_ours, tau_ours, &iinfo);
    printf("[ours]  info=%d\n", iinfo);
    print_vec_d("d", d_ours, N);
    print_vec_d("e", e_ours, N - 1);
    print_vec_z("tau", tau_ours, N - 1);

    memcpy(AP_obl, AP_MACOS, sizeof(AP_obl));
    zhptrd_("L", &fn, AP_obl, d_obl, e_obl, tau_obl, &iinfo);
    printf("[oblas] info=%d\n", iinfo);
    print_vec_d("d", d_obl, N);
    print_vec_d("e", e_obl, N - 1);
    print_vec_z("tau", tau_obl, N - 1);

    printf("\n  d diff:");
    for (int i = 0; i < N; i++) printf(" %.3e", fabs(d_ours[i] - d_obl[i]));
    printf("\n  e diff:");
    for (int i = 0; i < N - 1; i++) printf(" %.3e", fabs(e_ours[i] - e_obl[i]));
    printf("\n  tau diff:");
    for (int i = 0; i < N - 1; i++) printf(" %.3e", cabs(tau_ours[i] - tau_obl[i]));
    printf("\n  AP diff:");
    for (int i = 0; i < NPACK; i++) printf(" %.3e", cabs(AP_ours[i] - AP_obl[i]));
    printf("\n");

    /* ================================================================ */
    /* Stage 2: DSTEBZ                                                  */
    /* ================================================================ */
    printf("\n========================================\n");
    printf("=== Stage 2: DSTEBZ ===\n");
    printf("========================================\n");

    double w_ours[N], w_obl[N];
    int iblock_ours[N], isplit_ours[N], iblock_obl[N], isplit_obl[N];
    int m_ours, m_obl, nsplit_ours, nsplit_obl;
    double stebz_work[4 * N];
    int stebz_iwork[3 * N];

    /* Use ours d/e for ours, obl d/e for obl */
    dstebz("A", "B", N, 0.0, 0.0, 0, N - 1, abstol,
           d_ours, e_ours, &m_ours, &nsplit_ours, w_ours,
           iblock_ours, isplit_ours, stebz_work, stebz_iwork, &iinfo);
    printf("[ours]  info=%d, m=%d, nsplit=%d\n", iinfo, m_ours, nsplit_ours);
    print_vec_d("w", w_ours, m_ours);
    print_vec_i("iblock", iblock_ours, m_ours);
    print_vec_i("isplit", isplit_ours, nsplit_ours);

    {
        int fil = 1, fiu = N;
        double fvl = 0.0, fvu = 0.0;
        dstebz_("A", "B", &fn, &fvl, &fvu, &fil, &fiu, &abstol,
                d_obl, e_obl, &m_obl, &nsplit_obl, w_obl,
                iblock_obl, isplit_obl, stebz_work, stebz_iwork, &iinfo);
    }
    printf("[oblas] info=%d, m=%d, nsplit=%d\n", iinfo, m_obl, nsplit_obl);
    print_vec_d("w", w_obl, m_obl);
    print_vec_i("iblock", iblock_obl, m_obl);
    print_vec_i("isplit", isplit_obl, nsplit_obl);

    printf("\n  w diff:");
    int mmin = m_ours < m_obl ? m_ours : m_obl;
    for (int i = 0; i < mmin; i++) printf(" %.3e", fabs(w_ours[i] - w_obl[i]));
    printf("\n  iblock diff:");
    for (int i = 0; i < mmin; i++) printf(" %d", iblock_ours[i] - iblock_obl[i]);
    printf("\n");

    /* ================================================================ */
    /* Stage 3: ZSTEIN                                                  */
    /* ================================================================ */
    printf("\n========================================\n");
    printf("=== Stage 3: ZSTEIN ===\n");
    printf("========================================\n");

    double _Complex Z_ours[N * N], Z_obl[N * N];
    double stein_rwork[5 * N];
    int stein_iwork[N], ifail_ours[N], ifail_obl[N];

    memset(Z_ours, 0, sizeof(Z_ours));
    zstein(N, d_ours, e_ours, m_ours, w_ours,
           iblock_ours, isplit_ours, Z_ours, LDZ,
           stein_rwork, stein_iwork, ifail_ours, &iinfo);
    printf("[ours]  info=%d\n", iinfo);
    print_vec_i("ifail", ifail_ours, m_ours);
    print_mat_z("Z_pre", Z_ours, LDZ, N, m_ours);
    printf("  ortho residual: %.6e\n", orthogonality_residual(Z_ours, N, m_ours, LDZ));

    memset(Z_obl, 0, sizeof(Z_obl));
    zstein_(&fn, d_obl, e_obl, &m_obl, w_obl,
            iblock_obl, isplit_obl, Z_obl, &fldz,
            stein_rwork, stein_iwork, ifail_obl, &iinfo);
    printf("[oblas] info=%d\n", iinfo);
    print_vec_i("ifail", ifail_obl, m_obl);
    print_mat_z("Z_pre", Z_obl, LDZ, N, m_obl);
    printf("  ortho residual: %.6e\n", orthogonality_residual(Z_obl, N, m_obl, LDZ));

    /* ================================================================ */
    /* Stage 4: ZUPMTR                                                  */
    /* ================================================================ */
    printf("\n========================================\n");
    printf("=== Stage 4: ZUPMTR ===\n");
    printf("========================================\n");

    double _Complex upmtr_work[N];
    int fm_ours = N, fm2_ours = m_ours;
    int fm_obl = N, fm2_obl = m_obl;

    zupmtr("L", "L", "N", N, m_ours, AP_ours, tau_ours, Z_ours, LDZ,
           upmtr_work, &iinfo);
    printf("[ours]  info=%d\n", iinfo);
    print_mat_z("Z_post", Z_ours, LDZ, N, m_ours);
    printf("  ortho residual: %.6e\n", orthogonality_residual(Z_ours, N, m_ours, LDZ));

    zupmtr_("L", "L", "N", &fm_obl, &fm2_obl, AP_obl, tau_obl, Z_obl, &fldz,
            upmtr_work, &iinfo);
    printf("[oblas] info=%d\n", iinfo);
    print_mat_z("Z_post", Z_obl, LDZ, N, m_obl);
    printf("  ortho residual: %.6e\n", orthogonality_residual(Z_obl, N, m_obl, LDZ));

    /* ================================================================ */
    /* End-to-end: ZHPEVX                                               */
    /* ================================================================ */
    printf("\n========================================\n");
    printf("=== End-to-end: ZHPEVX ===\n");
    printf("========================================\n");

    {
        double _Complex AP_tmp[NPACK], Z_tmp[N * N], work_tmp[2 * N];
        double W_tmp[N], rwork_tmp[7 * N];
        int iwork_tmp[5 * N], ifail_tmp[N], m_tmp, info_tmp;

        memcpy(AP_tmp, AP_MACOS, sizeof(AP_tmp));
        zhpevx("V", "A", "L", N, AP_tmp, 0.0, 0.0, 0, N - 1,
               abstol, &m_tmp, W_tmp, Z_tmp, LDZ,
               work_tmp, rwork_tmp, iwork_tmp, ifail_tmp, &info_tmp);
        printf("[ours]  info=%d, m=%d\n", info_tmp, m_tmp);
        print_vec_d("W", W_tmp, m_tmp);
        print_mat_z("Z", Z_tmp, LDZ, N, m_tmp);
        printf("  ortho residual: %.6e\n", orthogonality_residual(Z_tmp, N, m_tmp, LDZ));
    }

    {
        double _Complex AP_tmp[NPACK], Z_tmp[N * N], work_tmp[2 * N];
        double W_tmp[N], rwork_tmp[7 * N];
        int iwork_tmp[5 * N], ifail_tmp[N], m_tmp, info_tmp;
        int fil = 1, fiu = N;
        double fvl = 0.0, fvu = 0.0;

        memcpy(AP_tmp, AP_MACOS, sizeof(AP_tmp));
        zhpevx_("V", "A", "L", &fn, AP_tmp,
                &fvl, &fvu, &fil, &fiu,
                &abstol, &m_tmp, W_tmp, Z_tmp, &fldz,
                work_tmp, rwork_tmp, iwork_tmp, ifail_tmp, &info_tmp);
        printf("[oblas] info=%d, m=%d\n", info_tmp, m_tmp);
        print_vec_d("W", W_tmp, m_tmp);
        print_mat_z("Z", Z_tmp, LDZ, N, m_tmp);
        printf("  ortho residual: %.6e\n", orthogonality_residual(Z_tmp, N, m_tmp, LDZ));
    }

    printf("\n=== DONE ===\n");
    return 0;
}
