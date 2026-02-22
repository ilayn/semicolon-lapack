/**
 * @file slacn2.c
 * @brief Estimates the 1-norm of a square matrix using reverse communication.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLACN2 estimates the 1-norm of a square, real matrix A.
 * Reverse communication is used for evaluating matrix-vector products.
 *
 * This is a thread safe version of SLACON, which uses the array isave
 * in place of a SAVE statement.
 *
 * @param[in]     n     The order of the matrix (n >= 1).
 * @param[out]    V     On the final return, V = A*W, where EST = norm(V)/norm(W)
 *                      (W is not returned). Array of dimension n.
 * @param[in,out] X     On an intermediate return, X should be overwritten by
 *                      A * X if kase=1, or A**T * X if kase=2, and SLACN2 must
 *                      be re-called with all other parameters unchanged.
 *                      Array of dimension n.
 * @param[out]    isgn  Integer array of dimension n.
 * @param[in,out] est   On entry with kase = 1 or 2 and isave[0] = 3, est should be
 *                      unchanged from the previous call.
 *                      On exit, est is an estimate (a lower bound) for norm(A).
 * @param[in,out] kase  On the initial call, kase should be 0.
 *                      On an intermediate return, kase will be 1 or 2, indicating
 *                      whether X should be overwritten by A * X or A**T * X.
 *                      On the final return from SLACN2, kase will again be 0.
 * @param[in,out] isave Integer array of dimension 3 used to save variables between
 *                      calls. isave[0] = JUMP, isave[1] = J, isave[2] = ITER.
 */
void slacn2(
    const INT n,
    f32* restrict V,
    f32* restrict X,
    INT* restrict isgn,
    f32* est,
    INT* kase,
    INT* restrict isave)
{
    const INT ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    INT i, jlast;
    f32 altsgn, estold, temp, xs;

    if (*kase == 0) {
        for (i = 0; i < n; i++) {
            X[i] = ONE / (f32)n;
        }
        *kase = 1;
        isave[0] = 1;  // JUMP = 1
        return;
    }

    switch (isave[0]) {
        case 1: goto L20;
        case 2: goto L40;
        case 3: goto L70;
        case 4: goto L110;
        case 5: goto L140;
    }

L20:
    // ENTRY (isave[0] = 1)
    // FIRST ITERATION. X HAS BEEN OVERWRITTEN BY A*X.

    if (n == 1) {
        V[0] = X[0];
        *est = fabsf(V[0]);
        // QUIT
        goto L150;
    }
    *est = cblas_sasum(n, X, 1);

    for (i = 0; i < n; i++) {
        if (X[i] >= ZERO) {
            X[i] = ONE;
        } else {
            X[i] = -ONE;
        }
        isgn[i] = (INT)(X[i] + 0.5f);  // NINT equivalent
        if (X[i] < 0) isgn[i] = (INT)(X[i] - 0.5f);
    }
    *kase = 2;
    isave[0] = 2;  // JUMP = 2
    return;

L40:
    // ENTRY (isave[0] = 2)
    // FIRST ITERATION. X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.

    isave[1] = cblas_isamax(n, X, 1);  // J (0-based)
    isave[2] = 2;  // ITER = 2

    // MAIN LOOP - ITERATIONS 2,3,...,ITMAX.

L50:
    for (i = 0; i < n; i++) {
        X[i] = ZERO;
    }
    X[isave[1]] = ONE;
    *kase = 1;
    isave[0] = 3;  // JUMP = 3
    return;

L70:
    // ENTRY (isave[0] = 3)
    // X HAS BEEN OVERWRITTEN BY A*X.

    cblas_scopy(n, X, 1, V, 1);
    estold = *est;
    *est = cblas_sasum(n, V, 1);

    for (i = 0; i < n; i++) {
        if (X[i] >= ZERO) {
            xs = ONE;
        } else {
            xs = -ONE;
        }
        INT xs_int = (INT)(xs + 0.5f);
        if (xs < 0) xs_int = (INT)(xs - 0.5f);
        if (xs_int != isgn[i]) {
            goto L90;
        }
    }
    // REPEATED SIGN VECTOR DETECTED, HENCE ALGORITHM HAS CONVERGED.
    goto L120;

L90:
    // TEST FOR CYCLING.
    if (*est <= estold) {
        goto L120;
    }

    for (i = 0; i < n; i++) {
        if (X[i] >= ZERO) {
            X[i] = ONE;
        } else {
            X[i] = -ONE;
        }
        isgn[i] = (INT)(X[i] + 0.5f);
        if (X[i] < 0) isgn[i] = (INT)(X[i] - 0.5f);
    }
    *kase = 2;
    isave[0] = 4;  // JUMP = 4
    return;

L110:
    // ENTRY (isave[0] = 4)
    // X HAS BEEN OVERWRITTEN BY TRANSPOSE(A)*X.

    jlast = isave[1];
    isave[1] = cblas_isamax(n, X, 1);  // J (0-based)
    if ((X[jlast] != fabsf(X[isave[1]])) && (isave[2] < ITMAX)) {
        isave[2] = isave[2] + 1;  // ITER++
        goto L50;
    }

L120:
    // ITERATION COMPLETE. FINAL STAGE.

    altsgn = ONE;
    for (i = 0; i < n; i++) {
        X[i] = altsgn * (ONE + (f32)i / (f32)((n > 1) ? (n - 1) : 1));
        altsgn = -altsgn;
    }
    *kase = 1;
    isave[0] = 5;  // JUMP = 5
    return;

L140:
    // ENTRY (isave[0] = 5)
    // X HAS BEEN OVERWRITTEN BY A*X.

    temp = TWO * (cblas_sasum(n, X, 1) / (f32)(3 * n));
    if (temp > *est) {
        cblas_scopy(n, X, 1, V, 1);
        *est = temp;
    }

L150:
    *kase = 0;
    return;
}
