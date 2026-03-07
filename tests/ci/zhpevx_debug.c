/**
 * @file zhpevx_debug.c
 * @brief Traces every intermediate value through zhptrd for n=3 lower triangle
 *        using the exact failing matrix from macOS ARM64 CI.
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
#include <string.h>
#include <math.h>
#include <complex.h>
#include <float.h>

#include "semicolon_lapack_complex_double.h"

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

#define N 3
#define LDZ N
#define NPACK (N*(N+1)/2)

#define PZ(label, z) printf("  " label " = (%.17e, %.17e)\n", creal(z), cimag(z))
#define PD(label, d) printf("  " label " = %.17e\n", d)

static const double _Complex AP_MACOS[NPACK] = {
    4.91800740771642708e-01 + 0.0 * I,
    4.26710053799474409e-01 - 4.33152203490123830e-02 * I,
    2.54303979114093304e-01 - 3.61189704644434770e-02 * I,
    3.74049209520915527e-01 + 0.0 * I,
    2.23827572119812329e-01 - 8.94080584792857243e-03 * I,
    1.34150049707441515e-01 + 0.0 * I
};

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

/*
 * Manual inline zhptrd lower for n=3. No CBLAS calls — everything
 * expanded to scalar ops so every multiply and add is visible.
 *
 * Packed lower storage for 3x3:
 *   AP[0] = A(0,0)   AP[1] = A(1,0)   AP[2] = A(2,0)
 *   AP[3] = A(1,1)   AP[4] = A(2,1)
 *   AP[5] = A(2,2)
 *
 * Iteration i=0: reduce column 0, operate on 2x2 submatrix A(1:2,1:2)
 * Iteration i=1: reduce column 1, operate on 1x1 submatrix A(2,2)
 */
static void trace_zhptrd(const double _Complex* AP_in,
                          double* d_out, double* e_out,
                          double _Complex* tau_out,
                          double _Complex* AP_out)
{
    double _Complex AP[NPACK];
    memcpy(AP, AP_in, sizeof(AP));

    printf("  Input AP:\n");
    for (int k = 0; k < NPACK; k++)
        printf("    AP[%d] = (%.17e, %.17e)\n", k, creal(AP[k]), cimag(AP[k]));

    AP[0] = CMPLX(creal(AP[0]), 0.0);

    /* ============================================================ */
    /* i=0: reduce column 0                                         */
    /*   v = [AP[1], AP[2]], alpha = AP[1]                          */
    /*   zlarfg(2, alpha, &AP[2], 1, &tau)                          */
    /* ============================================================ */
    printf("\n  ====== i=0 ======\n");
    double _Complex alpha0 = AP[1];
    PZ("alpha_in = AP[1]", alpha0);
    PZ("AP[2] before zlarfg", AP[2]);

    double _Complex tau0;
    zlarfg(2, &alpha0, &AP[2], 1, &tau0);

    PZ("alpha_out", alpha0);
    PZ("tau0", tau0);
    PZ("AP[2] after zlarfg (v[1])", AP[2]);
    e_out[0] = creal(alpha0);
    PD("e[0]", e_out[0]);

    if (creal(tau0) != 0.0 || cimag(tau0) != 0.0) {
        /* v = [1, AP[2]] */
        AP[1] = CMPLX(1.0, 0.0);
        double _Complex v0 = AP[1]; /* 1+0i */
        double _Complex v1 = AP[2];

        /* A_sub = packed lower 2x2 at AP[3..5]:
         *   A_sub(0,0) = AP[3] = A(1,1)  (real)
         *   A_sub(1,0) = AP[4] = A(2,1)
         *   A_sub(1,1) = AP[5] = A(2,2)  (real)
         */
        double _Complex a00 = AP[3], a10 = AP[4], a11 = AP[5];
        printf("  A_sub: a00=(%.17e, %.17e) a10=(%.17e, %.17e) a11=(%.17e, %.17e)\n",
               creal(a00), cimag(a00), creal(a10), cimag(a10), creal(a11), cimag(a11));

        /* y = tau * A_sub * v (Hermitian, lower) */
        /*   y[0] = tau * (a00*v0 + conj(a10)*v1) */
        /*   y[1] = tau * (a10*v0 + a11*v1)       */
        double _Complex Av0 = a00 * v0 + conj(a10) * v1;
        double _Complex Av1 = a10 * v0 + a11 * v1;
        PZ("A*v[0] = a00*v0 + conj(a10)*v1", Av0);
        PZ("A*v[1] = a10*v0 + a11*v1", Av1);

        double _Complex y0 = tau0 * Av0;
        double _Complex y1 = tau0 * Av1;
        PZ("y[0] = tau*A*v[0]", y0);
        PZ("y[1] = tau*A*v[1]", y1);

        /* dot = y^H * v = conj(y0)*v0 + conj(y1)*v1 */
        double _Complex dot = conj(y0) * v0 + conj(y1) * v1;
        PZ("dot = y^H * v", dot);

        /* alpha_w = -0.5 * tau * dot */
        double _Complex alpha_w = -0.5 * tau0 * dot;
        PZ("alpha_w = -0.5*tau*dot", alpha_w);

        /* w = y + alpha_w * v */
        double _Complex w0 = y0 + alpha_w * v0;
        double _Complex w1 = y1 + alpha_w * v1;
        PZ("w[0]", w0);
        PZ("w[1]", w1);

        /* A_sub -= v*w^H + w*v^H (Hermitian rank-2 update, lower only) */
        /* a00 -= v0*conj(w0) + w0*conj(v0) = 2*Re(v0*conj(w0)) */
        /* a10 -= v1*conj(w0) + w1*conj(v0) */
        /* a11 -= v1*conj(w1) + w1*conj(v1) = 2*Re(v1*conj(w1)) */
        double _Complex new_a00 = a00 - v0*conj(w0) - w0*conj(v0);
        double _Complex new_a10 = a10 - v1*conj(w0) - w1*conj(v0);
        double _Complex new_a11 = a11 - v1*conj(w1) - w1*conj(v1);
        new_a00 = CMPLX(creal(new_a00), 0.0);
        new_a11 = CMPLX(creal(new_a11), 0.0);

        PZ("new a00 (AP[3])", new_a00);
        PZ("new a10 (AP[4])", new_a10);
        PZ("new a11 (AP[5])", new_a11);

        AP[3] = new_a00;
        AP[4] = new_a10;
        AP[5] = new_a11;

        tau_out[0] = tau0;
    } else {
        printf("  tau0=0, skipping rank-2 update\n");
        tau_out[0] = tau0;
    }

    AP[1] = e_out[0];
    d_out[0] = creal(AP[0]);
    PD("d[0]", d_out[0]);

    /* ============================================================ */
    /* i=1: reduce column 1                                         */
    /*   scalar case: n-i-1 = 1, so zlarfg(1, ...)                 */
    /* ============================================================ */
    printf("\n  ====== i=1 ======\n");
    /* ii = 3, i1i1 = 5 */
    double _Complex alpha1 = AP[4]; /* A(2,1) */
    PZ("alpha_in = AP[4]", alpha1);

    double _Complex tau1;
    zlarfg(1, &alpha1, &AP[5], 1, &tau1);

    PZ("alpha_out", alpha1);
    PZ("tau1", tau1);
    e_out[1] = creal(alpha1);
    PD("e[1]", e_out[1]);

    if (creal(tau1) != 0.0 || cimag(tau1) != 0.0) {
        AP[4] = CMPLX(1.0, 0.0);
        double _Complex v_scalar = AP[4]; /* 1+0i */

        /* A_sub = 1x1 at AP[5] */
        double _Complex a_scalar = AP[5];
        PZ("A_sub (AP[5])", a_scalar);

        /* y = tau * a * v */
        double _Complex y_s = tau1 * a_scalar * v_scalar;
        PZ("y = tau*a*v", y_s);

        /* dot = conj(y)*v */
        double _Complex dot_s = conj(y_s) * v_scalar;
        PZ("dot = y^H * v", dot_s);

        /* alpha_w = -0.5*tau*dot */
        double _Complex aw = -0.5 * tau1 * dot_s;
        PZ("alpha_w", aw);

        /* w = y + alpha_w * v */
        double _Complex w_s = y_s + aw * v_scalar;
        PZ("w", w_s);

        /* a -= v*conj(w) + w*conj(v) = 2*Re(v*conj(w)) */
        double _Complex new_a = a_scalar - v_scalar*conj(w_s) - w_s*conj(v_scalar);
        new_a = CMPLX(creal(new_a), 0.0);
        PZ("new AP[5]", new_a);
        AP[5] = new_a;

        tau_out[1] = tau1;
    } else {
        printf("  tau1=0, skipping rank-2 update\n");
        tau_out[1] = tau1;
    }

    AP[4] = e_out[1];
    d_out[1] = creal(AP[3]);
    d_out[2] = creal(AP[5]);
    PD("d[1]", d_out[1]);
    PD("d[2]", d_out[2]);

    memcpy(AP_out, AP, sizeof(AP));
}

int main(void)
{
    printf("=== zhpevx_debug: traced zhptrd n=3 ===\n\n");

    double d_tr[N], e_tr[N];
    double _Complex tau_tr[N], AP_tr[NPACK];

    printf("=== TRACED ZHPTRD (scalar ops) ===\n");
    trace_zhptrd(AP_MACOS, d_tr, e_tr, tau_tr, AP_tr);

    printf("\n========================================\n");
    printf("=== Black-box comparison ===\n");
    printf("========================================\n");

    double _Complex AP_c[NPACK], AP_f[NPACK];
    double d_c[N], e_c[N], d_f[N], e_f[N];
    double _Complex tau_c[N], tau_f[N];
    int iinfo, fn = N;

    memcpy(AP_c, AP_MACOS, sizeof(AP_c));
    zhptrd("L", N, AP_c, d_c, e_c, tau_c, &iinfo);

    memcpy(AP_f, AP_MACOS, sizeof(AP_f));
    zhptrd_("L", &fn, AP_f, d_f, e_f, tau_f, &iinfo);

    printf("[ours]:\n");
    for (int i = 0; i < N; i++) PD("  d", d_c[i]);
    for (int i = 0; i < N-1; i++) PD("  e", e_c[i]);
    for (int i = 0; i < N-1; i++) PZ("  tau", tau_c[i]);
    printf("[oblas]:\n");
    for (int i = 0; i < N; i++) PD("  d", d_f[i]);
    for (int i = 0; i < N-1; i++) PD("  e", e_f[i]);
    for (int i = 0; i < N-1; i++) PZ("  tau", tau_f[i]);

    printf("\n  d diff:");
    for (int i = 0; i < N; i++) printf(" %.3e", fabs(d_c[i] - d_f[i]));
    printf("\n  e diff:");
    for (int i = 0; i < N - 1; i++) printf(" %.3e", fabs(e_c[i] - e_f[i]));
    printf("\n  tau diff:");
    for (int i = 0; i < N - 1; i++) printf(" %.3e", cabs(tau_c[i] - tau_f[i]));
    printf("\n");

    printf("\n  traced vs ours d diff:");
    for (int i = 0; i < N; i++) printf(" %.3e", fabs(d_tr[i] - d_c[i]));
    printf("\n  traced vs ours e diff:");
    for (int i = 0; i < N - 1; i++) printf(" %.3e", fabs(e_tr[i] - e_c[i]));
    printf("\n  traced vs ours tau diff:");
    for (int i = 0; i < N - 1; i++) printf(" %.3e", cabs(tau_tr[i] - tau_c[i]));
    printf("\n");

    /* Continue with dstebz + zstein + zupmtr using ours */
    printf("\n========================================\n");
    printf("=== DSTEBZ ===\n");
    printf("========================================\n");

    double abstol = DBL_MIN + DBL_MIN;
    double w_c[N], w_f[N];
    int iblock_c[N], isplit_c[N], iblock_f[N], isplit_f[N];
    int m_c, m_f, nsplit_c, nsplit_f;
    double stebz_work[4 * N]; int stebz_iwork[3 * N];

    dstebz("A", "B", N, 0.0, 0.0, 0, N - 1, abstol,
           d_c, e_c, &m_c, &nsplit_c, w_c,
           iblock_c, isplit_c, stebz_work, stebz_iwork, &iinfo);
    printf("[ours]  m=%d\n", m_c);
    for (int i = 0; i < m_c; i++)
        printf("  W[%d] = %.17e  iblock=%d\n", i, w_c[i], iblock_c[i]);

    {
        int fil = 1, fiu = N; double fvl = 0.0, fvu = 0.0;
        dstebz_("A", "B", &fn, &fvl, &fvu, &fil, &fiu, &abstol,
                d_f, e_f, &m_f, &nsplit_f, w_f,
                iblock_f, isplit_f, stebz_work, stebz_iwork, &iinfo);
    }
    printf("[oblas] m=%d\n", m_f);
    for (int i = 0; i < m_f; i++)
        printf("  W[%d] = %.17e  iblock=%d\n", i, w_f[i], iblock_f[i]);

    printf("\n========================================\n");
    printf("=== ZSTEIN ===\n");
    printf("========================================\n");

    double _Complex Z_c[N * N], Z_f[N * N];
    double stein_rwork[5 * N]; int stein_iwork[N], ifail_c[N], ifail_f[N];
    int fldz = LDZ;

    memset(Z_c, 0, sizeof(Z_c));
    zstein(N, d_c, e_c, m_c, w_c,
           iblock_c, isplit_c, Z_c, LDZ,
           stein_rwork, stein_iwork, ifail_c, &iinfo);
    printf("[ours]  info=%d, ortho=%.6e\n", iinfo, orthogonality_residual(Z_c, N, m_c, LDZ));
    for (int j = 0; j < m_c; j++)
        for (int i = 0; i < N; i++)
            printf("  Z[%d,%d] = (%.17e, %.17e)\n", i, j,
                   creal(Z_c[i+j*LDZ]), cimag(Z_c[i+j*LDZ]));

    memset(Z_f, 0, sizeof(Z_f));
    zstein_(&fn, d_f, e_f, &m_f, w_f,
            iblock_f, isplit_f, Z_f, &fldz,
            stein_rwork, stein_iwork, ifail_f, &iinfo);
    printf("[oblas] info=%d, ortho=%.6e\n", iinfo, orthogonality_residual(Z_f, N, m_f, LDZ));
    for (int j = 0; j < m_f; j++)
        for (int i = 0; i < N; i++)
            printf("  Z[%d,%d] = (%.17e, %.17e)\n", i, j,
                   creal(Z_f[i+j*LDZ]), cimag(Z_f[i+j*LDZ]));

    printf("\n========================================\n");
    printf("=== ZUPMTR ===\n");
    printf("========================================\n");

    double _Complex upmtr_work[N];
    int fm2_f = m_f;

    zupmtr("L", "L", "N", N, m_c, AP_c, tau_c, Z_c, LDZ, upmtr_work, &iinfo);
    printf("[ours]  info=%d, ortho=%.6e\n", iinfo, orthogonality_residual(Z_c, N, m_c, LDZ));
    for (int j = 0; j < m_c; j++)
        for (int i = 0; i < N; i++)
            printf("  Z[%d,%d] = (%.17e, %.17e)\n", i, j,
                   creal(Z_c[i+j*LDZ]), cimag(Z_c[i+j*LDZ]));

    zupmtr_("L", "L", "N", &fn, &fm2_f, AP_f, tau_f, Z_f, &fldz, upmtr_work, &iinfo);
    printf("[oblas] info=%d, ortho=%.6e\n", iinfo, orthogonality_residual(Z_f, N, m_f, LDZ));
    for (int j = 0; j < m_f; j++)
        for (int i = 0; i < N; i++)
            printf("  Z[%d,%d] = (%.17e, %.17e)\n", i, j,
                   creal(Z_f[i+j*LDZ]), cimag(Z_f[i+j*LDZ]));

    printf("\n=== DONE ===\n");
    return 0;
}
