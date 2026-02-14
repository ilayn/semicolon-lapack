/**
 * @file zlacn2.c
 * @brief ZLACN2 estimates the 1-norm of a square matrix, using reverse
 *        communication for evaluating matrix-vector products.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLACN2 estimates the 1-norm of a square, complex matrix A.
 * Reverse communication is used for evaluating matrix-vector products.
 *
 * This is a thread safe version of ZLACON, which uses the array isave
 * in place of a SAVE statement, as follows:
 *
 *     ZLACON     ZLACN2
 *      JUMP     isave[0]
 *      J        isave[1]
 *      ITER     isave[2]
 *
 * @param[in]     n     The order of the matrix. n >= 1.
 * @param[out]    V     Complex array, dimension (n).
 *                      On the final return, V = A*W, where EST = norm(V)/norm(W)
 *                      (W is not returned).
 * @param[in,out] X     Complex array, dimension (n).
 *                      On an intermediate return, X should be overwritten by
 *                            A * X,   if kase=1,
 *                            A**H * X,  if kase=2,
 *                      where A**H is the conjugate transpose of A, and ZLACN2
 *                      must be re-called with all the other parameters unchanged.
 * @param[in,out] est   On entry with kase = 1 or 2 and isave[0] = 3, est should
 *                      be unchanged from the previous call to ZLACN2.
 *                      On exit, est is an estimate (a lower bound) for norm(A).
 * @param[in,out] kase  On the initial call to ZLACN2, kase should be 0.
 *                      On an intermediate return, kase will be 1 or 2, indicating
 *                      whether X should be overwritten by A * X or A**H * X.
 *                      On the final return from ZLACN2, kase will again be 0.
 * @param[in,out] isave Integer array of dimension 3 used to save variables
 *                      between calls to ZLACN2.
 */
void zlacn2(
    const int n,
    double complex* const restrict V,
    double complex* const restrict X,
    double* est,
    int* kase,
    int* const restrict isave)
{
    const int ITMAX = 5;
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double complex CZERO = CMPLX(0.0, 0.0);
    const double complex CONE = CMPLX(1.0, 0.0);

    int i, jlast;
    double absxi, altsgn, estold, safmin, temp;

    safmin = dlamch("Safe minimum");
    if (*kase == 0) {
        for (i = 0; i < n; i++) {
            X[i] = CMPLX(ONE / (double)n, 0.0);
        }
        *kase = 1;
        isave[0] = 1;
        return;
    }

    switch (isave[0]) {
    case 1: goto L20;
    case 2: goto L40;
    case 3: goto L70;
    case 4: goto L90;
    case 5: goto L120;
    }

/*     ................ ENTRY   (ISAVE(1) = 1)
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
            X[i] = CONE;
        }
    }
    *kase = 2;
    isave[0] = 2;
    return;

/*     ................ ENTRY   (ISAVE(1) = 2)
       FIRST ITERATION.  X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L40:
    isave[1] = izmax1(n, X, 1);
    isave[2] = 2;

/*     MAIN LOOP - ITERATIONS 2,3,...,ITMAX. */

L50:
    for (i = 0; i < n; i++) {
        X[i] = CZERO;
    }
    X[isave[1]] = CONE;
    *kase = 1;
    isave[0] = 3;
    return;

/*     ................ ENTRY   (ISAVE(1) = 3)
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
            X[i] = CONE;
        }
    }
    *kase = 2;
    isave[0] = 4;
    return;

/*     ................ ENTRY   (ISAVE(1) = 4)
       X HAS BEEN OVERWRITTEN BY CTRANS(A)*X. */

L90:
    jlast = isave[1];
    isave[1] = izmax1(n, X, 1);
    if ((cabs(X[jlast]) != cabs(X[isave[1]])) && (isave[2] < ITMAX)) {
        isave[2] = isave[2] + 1;
        goto L50;
    }

/*     ITERATION COMPLETE.  FINAL STAGE. */

L100:
    altsgn = ONE;
    for (i = 0; i < n; i++) {
        X[i] = CMPLX(altsgn * (ONE + (double)i / (double)(n - 1)), 0.0);
        altsgn = -altsgn;
    }
    *kase = 1;
    isave[0] = 5;
    return;

/*     ................ ENTRY   (ISAVE(1) = 5)
       X HAS BEEN OVERWRITTEN BY A*X. */

L120:
    temp = TWO * (dzsum1(n, X, 1) / (double)(3 * n));
    if (temp > *est) {
        cblas_zcopy(n, X, 1, V, 1);
        *est = temp;
    }

L130:
    *kase = 0;
}
