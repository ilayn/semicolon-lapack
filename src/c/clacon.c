/**
 * @file clacon.c
 * @brief CLACON estimates the 1-norm of a square matrix, using reverse communication.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLACON estimates the 1-norm of a square, complex matrix A.
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
 *          where A**H is the conjugate transpose of A, and CLACON must be
 *          re-called with all the other parameters unchanged.
 *
 * @param[in,out] est
 *          On entry with kase = 1 or 2 and jump = 3, est should be
 *          unchanged from the previous call to CLACON.
 *          On exit, est is an estimate (a lower bound) for norm(A).
 *
 * @param[in,out] kase
 *          On the initial call to CLACON, kase should be 0.
 *          On an intermediate return, kase will be 1 or 2, indicating
 *          whether X should be overwritten by A * X or A**H * X.
 *          On the final return from CLACON, kase will again be 0.
 */
void clacon(
    const INT n,
    c64* restrict V,
    c64* restrict X,
    f32* est,
    INT* kase)
{
    static const INT ITMAX = 5;
    const f32 one = 1.0f;
    const f32 two = 2.0f;
    const c64 czero = CMPLXF(0.0f, 0.0f);
    const c64 cone = CMPLXF(1.0f, 0.0f);

    static INT i, iter, j, jlast, jump;
    static f32 altsgn, estold, safmin, temp;
    f32 absxi;

    safmin = slamch("Safe minimum");
    if (*kase == 0) {
        for (i = 0; i < n; i++) {
            X[i] = CMPLXF(one / (f32)n, 0.0f);
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
        *est = cabsf(V[0]);
        goto L130;
    }
    *est = scsum1(n, X, 1);

    for (i = 0; i < n; i++) {
        absxi = cabsf(X[i]);
        if (absxi > safmin) {
            X[i] = CMPLXF(crealf(X[i]) / absxi, cimagf(X[i]) / absxi);
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
    j = icmax1(n, X, 1);
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
    cblas_ccopy(n, X, 1, V, 1);
    estold = *est;
    *est = scsum1(n, V, 1);

/*     TEST FOR CYCLING. */
    if (*est <= estold)
        goto L100;

    for (i = 0; i < n; i++) {
        absxi = cabsf(X[i]);
        if (absxi > safmin) {
            X[i] = CMPLXF(crealf(X[i]) / absxi, cimagf(X[i]) / absxi);
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
    j = icmax1(n, X, 1);
    if ((cabsf(X[jlast]) != cabsf(X[j])) && (iter < ITMAX)) {
        iter = iter + 1;
        goto L50;
    }

/*     ITERATION COMPLETE.  FINAL STAGE. */

L100:
    altsgn = one;
    for (i = 0; i < n; i++) {
        X[i] = CMPLXF(altsgn * (one + (f32)i / (f32)(n - 1)), 0.0f);
        altsgn = -altsgn;
    }
    *kase = 1;
    jump = 5;
    return;

/*     ................ ENTRY   (JUMP = 5)
       X HAS BEEN OVERWRITTEN BY A*X. */

L120:
    temp = two * (scsum1(n, X, 1) / (f32)(3 * n));
    if (temp > *est) {
        cblas_ccopy(n, X, 1, V, 1);
        *est = temp;
    }

L130:
    *kase = 0;
}
