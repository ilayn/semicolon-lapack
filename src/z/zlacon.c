/**
 * @file zlacon.c
 * @brief ZLACON estimates the 1-norm of a square matrix, using reverse communication.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLACON estimates the 1-norm of a square, complex matrix A.
 * Reverse communication is used for evaluating matrix-vector products.
 *
 * @param[in] n
 *          The order of the matrix. n >= 1.
 *
 * @param[out] V
 *          Complex*16 array, dimension (n).
 *          On the final return, V = A*W, where EST = norm(V)/norm(W)
 *          (W is not returned).
 *
 * @param[in,out] X
 *          Complex*16 array, dimension (n).
 *          On an intermediate return, X should be overwritten by
 *                A * X,   if kase=1,
 *                A**H * X,  if kase=2,
 *          where A**H is the conjugate transpose of A, and ZLACON must be
 *          re-called with all the other parameters unchanged.
 *
 * @param[in,out] est
 *          On entry with kase = 1 or 2 and jump = 3, est should be
 *          unchanged from the previous call to ZLACON.
 *          On exit, est is an estimate (a lower bound) for norm(A).
 *
 * @param[in,out] kase
 *          On the initial call to ZLACON, kase should be 0.
 *          On an intermediate return, kase will be 1 or 2, indicating
 *          whether X should be overwritten by A * X or A**H * X.
 *          On the final return from ZLACON, kase will again be 0.
 */
void zlacon(
    const int n,
    c128* restrict V,
    c128* restrict X,
    f64* est,
    int* kase)
{
    static const int ITMAX = 5;
    const f64 one = 1.0;
    const f64 two = 2.0;
    const c128 czero = CMPLX(0.0, 0.0);
    const c128 cone = CMPLX(1.0, 0.0);

    static int i, iter, j, jlast, jump;
    static f64 altsgn, estold, safmin, temp;
    f64 absxi;

    safmin = dlamch("Safe minimum");
    if (*kase == 0) {
        for (i = 0; i < n; i++) {
            X[i] = CMPLX(one / (f64)n, 0.0);
        }
        *kase = 1;
        jump = 1;
        return;
    }

    switch (jump) {
    case 1:
        goto L20;
    case 2:
        goto L40;
    case 3:
        goto L70;
    case 4:
        goto L90;
    case 5:
        goto L120;
    }

/*     ................ ENTRY   (JUMP = 1)
       FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY A*X. */

L20:
    if (n == 1) {
        V[0] = X[0];
        *est = cabs(V[0]);
        goto L130;
    }
    *est = dzsum1(n, X, 1);

    for (i = 0; i < n; i++) {
        absxi = cabs(X[i]);
        if (absxi > safmin) {
            X[i] = CMPLX(creal(X[i]) / absxi, cimag(X[i]) / absxi);
        } else {
            X[i] = cone;
        }
    }
    *kase = 2;
    jump = 2;
    return;

/*     ................ ENTRY   (JUMP = 2)
       FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L40:
    j = izmax1(n, X, 1);
    iter = 2;

/*     MAIN LOOP - ITERATIONS 2,3,...,ITMAX. */

L50:
    for (i = 0; i < n; i++) {
        X[i] = czero;
    }
    X[j] = cone;
    *kase = 1;
    jump = 3;
    return;

/*     ................ ENTRY   (JUMP = 3)
       X HAS BEEN OVERWRITTEN BY A*X. */

L70:
    cblas_zcopy(n, X, 1, V, 1);
    estold = *est;
    *est = dzsum1(n, V, 1);

/*     TEST FOR CYCLING. */
    if (*est <= estold)
        goto L100;

    for (i = 0; i < n; i++) {
        absxi = cabs(X[i]);
        if (absxi > safmin) {
            X[i] = CMPLX(creal(X[i]) / absxi, cimag(X[i]) / absxi);
        } else {
            X[i] = cone;
        }
    }
    *kase = 2;
    jump = 4;
    return;

/*     ................ ENTRY   (JUMP = 4)
       X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L90:
    jlast = j;
    j = izmax1(n, X, 1);
    if ((cabs(X[jlast]) != cabs(X[j])) && (iter < ITMAX)) {
        iter = iter + 1;
        goto L50;
    }

/*     ITERATION COMPLETE.  FINAL STAGE. */

L100:
    altsgn = one;
    for (i = 0; i < n; i++) {
        X[i] = CMPLX(altsgn * (one + (f64)i / (f64)(n - 1)), 0.0);
        altsgn = -altsgn;
    }
    *kase = 1;
    jump = 5;
    return;

/*     ................ ENTRY   (JUMP = 5)
       X HAS BEEN OVERWRITTEN BY A*X. */

L120:
    temp = two * (dzsum1(n, X, 1) / (f64)(3 * n));
    if (temp > *est) {
        cblas_zcopy(n, X, 1, V, 1);
        *est = temp;
    }

L130:
    *kase = 0;
}
