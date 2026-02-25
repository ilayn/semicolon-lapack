/*
 * MSVC consumer test: verifies that a MinGW-built semilapack.dll can be
 * called from MSVC-compiled code.  Solves a 10x10 complex linear system
 * using zgetrf -> zlange/zgecon -> zgetrs and checks the residual.
 *
 * Compiled in CI with:
 *   cl.exe /std:c11 /O2 /W3 msvc_consumer_test.c semilapack.lib
 */

#include <stdio.h>
#include <math.h>

#define SEMICOLON_USE_SHARED
#include "semicolon_lapack/semicolon_lapack.h"

#define N 10
#define LDA N
#define LDB N
#define NRHS 1

int main(void) {
    /* A = diag(1..10) + some off-diagonal structure so it is non-trivial */
    c128 A[N * N];
    c128 A_save[N * N];
    c128 B[N];
    c128 B_save[N];
    c128 work_con[2 * N];
    f64  rwork[2 * N];
    i32  ipiv[N];
    i32  info;

    /* Build A: A(i,j) = (i+1)*delta(i,j) + 0.1*(i+j+1) + 0.05*I*(i-j)
     * Column-major storage. */
    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            double re = 0.1 * (i + j + 1);
            double im = 0.05 * (i - j);
            if (i == j) re += (i + 1);
#ifdef _MSC_VER
            c128 val = {re, im};
            A[i + j * LDA] = val;
#else
            A[i + j * LDA] = re + im * _Complex_I;
#endif
        }
    }

    /* B = A * x_exact where x_exact = (1, 1, ..., 1) */
    for (int i = 0; i < N; i++) {
#ifdef _MSC_VER
        c128 zero = {0.0, 0.0};
        B[i] = zero;
#else
        B[i] = 0.0;
#endif
        for (int j = 0; j < N; j++) {
#ifdef _MSC_VER
            c128 a = A[i + j * LDA];
            c128 b = B[i];
            c128 sum = {b._Val[0] + a._Val[0], b._Val[1] + a._Val[1]};
            B[i] = sum;
#else
            B[i] += A[i + j * LDA];
#endif
        }
    }

    /* Save copies for residual check */
    for (int i = 0; i < N * N; i++) A_save[i] = A[i];
    for (int i = 0; i < N; i++) B_save[i] = B[i];

    /* Step 1: LU factorization */
    zgetrf(N, N, A, LDA, ipiv, &info);
    if (info != 0) {
        printf("FAIL: zgetrf returned info = %d\n", info);
        return 1;
    }
    printf("zgetrf: OK (info = 0)\n");

    /* Step 2: Condition number estimate */
    f64 anorm = zlange("1", N, N, A_save, LDA, rwork);
    f64 rcond = 0.0;
    zgecon("1", N, A, LDA, anorm, &rcond, work_con, rwork, &info);
    if (info != 0) {
        printf("FAIL: zgecon returned info = %d\n", info);
        return 1;
    }
    printf("zgecon: OK (rcond = %.6e)\n", rcond);

    if (rcond < 1e-15) {
        printf("FAIL: matrix is nearly singular (rcond = %.6e)\n", rcond);
        return 1;
    }

    /* Step 3: Solve */
    zgetrs("N", N, NRHS, A, LDA, ipiv, B, LDB, &info);
    if (info != 0) {
        printf("FAIL: zgetrs returned info = %d\n", info);
        return 1;
    }
    printf("zgetrs: OK (info = 0)\n");

    /* Check: x should be (1, 1, ..., 1).  Compute max |x_i - 1|. */
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
#ifdef _MSC_VER
        double re = B[i]._Val[0] - 1.0;
        double im = B[i]._Val[1];
#else
        double re = creal(B[i]) - 1.0;
        double im = cimag(B[i]);
#endif
        double err = sqrt(re * re + im * im);
        if (err > max_err) max_err = err;
    }

    printf("Max error: %.6e\n", max_err);

    if (max_err > 1e-10) {
        printf("FAIL: solution error too large\n");
        return 1;
    }

    printf("PASS: MSVC consumer test succeeded\n");
    return 0;
}
