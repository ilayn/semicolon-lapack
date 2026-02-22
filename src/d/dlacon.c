/**
 * @file dlacon.c
 * @brief DLACON estimates the 1-norm of a square matrix, using reverse communication.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLACON estimates the 1-norm of a square, real matrix A.
 * Reverse communication is used for evaluating matrix-vector products.
 *
 * @param[in] n
 *          The order of the matrix. n >= 1.
 *
 * @param[out] V
 *          Double precision array, dimension (n).
 *          On the final return, V = A*W, where EST = norm(V)/norm(W)
 *          (W is not returned).
 *
 * @param[in,out] X
 *          Double precision array, dimension (n).
 *          On an intermediate return, X should be overwritten by
 *                A * X,   if kase=1,
 *                A**T * X,  if kase=2,
 *          and DLACON must be re-called with all the other parameters
 *          unchanged.
 *
 * @param[out] ISGN
 *          Integer array, dimension (n).
 *
 * @param[in,out] est
 *          On entry with kase = 1 or 2 and jump = 3, est should be
 *          unchanged from the previous call to DLACON.
 *          On exit, est is an estimate (a lower bound) for norm(A).
 *
 * @param[in,out] kase
 *          On the initial call to DLACON, kase should be 0.
 *          On an intermediate return, kase will be 1 or 2, indicating
 *          whether X should be overwritten by A * X or A**T * X.
 *          On the final return from DLACON, kase will again be 0.
 */
void dlacon(
    const INT n,
    f64* restrict V,
    f64* restrict X,
    INT* restrict ISGN,
    f64* est,
    INT* kase)
{
    static const INT ITMAX = 5;
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;

    static INT i, iter, j, jlast, jump;
    static f64 altsgn, estold, temp;

    if (*kase == 0) {
        for (i = 0; i < n; i++) {
            X[i] = one / (f64)n;
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
        goto L110;
    case 5:
        goto L140;
    }

L20:
    if (n == 1) {
        V[0] = X[0];
        *est = fabs(V[0]);
        goto L150;
    }
    *est = cblas_dasum(n, X, 1);

    for (i = 0; i < n; i++) {
        X[i] = (X[i] >= zero) ? one : -one;
        ISGN[i] = (INT)(X[i] + 0.5);
        if (X[i] < zero) ISGN[i] = (INT)(X[i] - 0.5);
    }
    *kase = 2;
    jump = 2;
    return;

L40:
    j = cblas_idamax(n, X, 1);
    iter = 2;

L50:
    for (i = 0; i < n; i++) {
        X[i] = zero;
    }
    X[j] = one;
    *kase = 1;
    jump = 3;
    return;

L70:
    cblas_dcopy(n, X, 1, V, 1);
    estold = *est;
    *est = cblas_dasum(n, V, 1);
    for (i = 0; i < n; i++) {
        INT new_sgn = (X[i] >= zero) ? 1 : -1;
        if (new_sgn != ISGN[i]) {
            goto L90;
        }
    }
    goto L120;

L90:
    if (*est <= estold) {
        goto L120;
    }

    for (i = 0; i < n; i++) {
        X[i] = (X[i] >= zero) ? one : -one;
        ISGN[i] = (INT)(X[i] + 0.5);
        if (X[i] < zero) ISGN[i] = (INT)(X[i] - 0.5);
    }
    *kase = 2;
    jump = 4;
    return;

L110:
    jlast = j;
    j = cblas_idamax(n, X, 1);
    if ((X[jlast] != fabs(X[j])) && (iter < ITMAX)) {
        iter = iter + 1;
        goto L50;
    }

L120:
    altsgn = one;
    for (i = 0; i < n; i++) {
        X[i] = altsgn * (one + (f64)i / (f64)(n - 1));
        altsgn = -altsgn;
    }
    *kase = 1;
    jump = 5;
    return;

L140:
    temp = two * (cblas_dasum(n, X, 1) / (f64)(3 * n));
    if (temp > *est) {
        cblas_dcopy(n, X, 1, V, 1);
        *est = temp;
    }

L150:
    *kase = 0;
}
