/**
 * @file dlasy2.c
 * @brief DLASY2 solves the Sylvester matrix equation where the matrices are
 *        of order 1 or 2.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/* Lookup tables for 2x2 pivoting (file scope for thread safety) */
static const int locu12[4] = {2, 3, 0, 1};  /* 0-based: {3,4,1,2} - 1 */
static const int locl21[4] = {1, 0, 3, 2};  /* 0-based: {2,1,4,3} - 1 */
static const int locu22[4] = {3, 2, 1, 0};  /* 0-based: {4,3,2,1} - 1 */
static const int xswpiv[4] = {0, 0, 1, 1};  /* FALSE, FALSE, TRUE, TRUE */
static const int bswpiv[4] = {0, 1, 0, 1};  /* FALSE, TRUE, FALSE, TRUE */

/**
 * DLASY2 solves for the N1 by N2 matrix X, 1 <= N1,N2 <= 2, in
 *
 *        op(TL)*X + ISGN*X*op(TR) = SCALE*B,
 *
 * where TL is N1 by N1, TR is N2 by N2, B is N1 by N2, and ISGN = 1 or
 * -1.  op(T) = T or T**T, where T**T denotes the transpose of T.
 *
 * @param[in]  ltranl  If nonzero, op(TL) = TL**T. Otherwise op(TL) = TL.
 * @param[in]  ltranr  If nonzero, op(TR) = TR**T. Otherwise op(TR) = TR.
 * @param[in]  isgn    The sign in the equation (+1 or -1).
 * @param[in]  n1      The order of matrix TL. N1 may only be 0, 1 or 2.
 * @param[in]  n2      The order of matrix TR. N2 may only be 0, 1 or 2.
 * @param[in]  TL      N1 by N1 matrix TL, dimension (ldtl, 2).
 * @param[in]  ldtl    Leading dimension of TL. ldtl >= max(1, n1).
 * @param[in]  TR      N2 by N2 matrix TR, dimension (ldtr, 2).
 * @param[in]  ldtr    Leading dimension of TR. ldtr >= max(1, n2).
 * @param[in]  B       N1 by N2 right-hand side matrix, dimension (ldb, 2).
 * @param[in]  ldb     Leading dimension of B. ldb >= max(1, n1).
 * @param[out] scale   Scale factor. Chosen <= 1 to prevent overflow.
 * @param[out] X       N1 by N2 solution matrix, dimension (ldx, 2).
 * @param[in]  ldx     Leading dimension of X. ldx >= max(1, n1).
 * @param[out] xnorm   Infinity-norm of the solution.
 * @param[out] info
 *                         - = 0: successful exit.
 *                         - = 1: TL and TR have too close eigenvalues, so TL or
 *                           TR is perturbed to get a nonsingular equation.
 * @note In the interests of speed, this routine does not check the inputs
 *       for errors.
 */
void dlasy2(const int ltranl, const int ltranr, const int isgn,
            const int n1, const int n2,
            const f64* TL, const int ldtl,
            const f64* TR, const int ldtr,
            const f64* B, const int ldb,
            f64* scale, f64* X, const int ldx,
            f64* xnorm, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 HALF = 0.5;
    const f64 EIGHT = 8.0;

    int i, ip, ipiv, ipsv, j, jp, jpsv, k;
    f64 bet, eps, gam, l21, sgn, smin, smlnum, tau1;
    f64 temp, u11, u12, u22, xmax;
    f64 btmp[4], t16[16], tmp[4], x2[2];  /* t16 stored column-major: t16[i + 4*j] */
    int jpiv[4];

    *info = 0;

    /* Quick return if possible */
    if (n1 == 0 || n2 == 0)
        return;

    /* Set constants to control overflow */
    eps = dlamch("P");
    smlnum = dlamch("S") / eps;
    sgn = (f64)isgn;

    k = n1 + n1 + n2 - 2;

    if (k == 1) {
        /* 1 by 1: TL11*X + SGN*X*TR11 = B11 */
        tau1 = TL[0] + sgn * TR[0];
        bet = fabs(tau1);
        if (bet <= smlnum) {
            tau1 = smlnum;
            bet = smlnum;
            *info = 1;
        }

        *scale = ONE;
        gam = fabs(B[0]);
        if (smlnum * gam > bet)
            *scale = ONE / gam;

        X[0] = (B[0] * (*scale)) / tau1;
        *xnorm = fabs(X[0]);
        return;
    }

    if (k == 2) {
        /* 1 by 2:
           TL11*[X11 X12] + ISGN*[X11 X12]*op[TR11 TR12] = [B11 B12]
                                            [TR21 TR22]              */
        smin = fabs(TL[0]);
        temp = fabs(TR[0]);
        if (temp > smin) smin = temp;
        temp = fabs(TR[1]);
        if (temp > smin) smin = temp;
        temp = fabs(TR[ldtr]);
        if (temp > smin) smin = temp;
        temp = fabs(TR[1 + ldtr]);
        if (temp > smin) smin = temp;
        smin = eps * smin;
        if (smin < smlnum) smin = smlnum;

        tmp[0] = TL[0] + sgn * TR[0];
        tmp[3] = TL[0] + sgn * TR[1 + ldtr];
        if (ltranr) {
            tmp[1] = sgn * TR[1];
            tmp[2] = sgn * TR[ldtr];
        } else {
            tmp[1] = sgn * TR[ldtr];
            tmp[2] = sgn * TR[1];
        }
        btmp[0] = B[0];
        btmp[1] = B[ldb];
        /* Fall through to 2x2 solve */
    } else if (k == 3) {
        /* 2 by 1:
           op[TL11 TL12]*[X11] + ISGN*[X11]*TR11 = [B11]
             [TL21 TL22] [X21]        [X21]        [B21] */
        smin = fabs(TR[0]);
        temp = fabs(TL[0]);
        if (temp > smin) smin = temp;
        temp = fabs(TL[1]);
        if (temp > smin) smin = temp;
        temp = fabs(TL[ldtl]);
        if (temp > smin) smin = temp;
        temp = fabs(TL[1 + ldtl]);
        if (temp > smin) smin = temp;
        smin = eps * smin;
        if (smin < smlnum) smin = smlnum;

        tmp[0] = TL[0] + sgn * TR[0];
        tmp[3] = TL[1 + ldtl] + sgn * TR[0];
        if (ltranl) {
            tmp[1] = TL[ldtl];
            tmp[2] = TL[1];
        } else {
            tmp[1] = TL[1];
            tmp[2] = TL[ldtl];
        }
        btmp[0] = B[0];
        btmp[1] = B[1];
        /* Fall through to 2x2 solve */
    }

    if (k == 2 || k == 3) {
        /* Solve 2 by 2 system using complete pivoting.
           Set pivots less than SMIN to SMIN. */
        ipiv = cblas_idamax(4, tmp, 1);
        u11 = tmp[ipiv];
        if (fabs(u11) <= smin) {
            *info = 1;
            u11 = smin;
        }
        u12 = tmp[locu12[ipiv]];
        l21 = tmp[locl21[ipiv]] / u11;
        u22 = tmp[locu22[ipiv]] - u12 * l21;
        if (fabs(u22) <= smin) {
            *info = 1;
            u22 = smin;
        }
        if (bswpiv[ipiv]) {
            temp = btmp[1];
            btmp[1] = btmp[0] - l21 * temp;
            btmp[0] = temp;
        } else {
            btmp[1] = btmp[1] - l21 * btmp[0];
        }
        *scale = ONE;
        if ((TWO * smlnum) * fabs(btmp[1]) > fabs(u22) ||
            (TWO * smlnum) * fabs(btmp[0]) > fabs(u11)) {
            *scale = HALF / (fabs(btmp[0]) > fabs(btmp[1]) ? fabs(btmp[0]) : fabs(btmp[1]));
            btmp[0] = btmp[0] * (*scale);
            btmp[1] = btmp[1] * (*scale);
        }
        x2[1] = btmp[1] / u22;
        x2[0] = btmp[0] / u11 - (u12 / u11) * x2[1];
        if (xswpiv[ipiv]) {
            temp = x2[1];
            x2[1] = x2[0];
            x2[0] = temp;
        }
        X[0] = x2[0];
        if (n1 == 1) {
            X[ldx] = x2[1];
            *xnorm = fabs(X[0]) + fabs(X[ldx]);
        } else {
            X[1] = x2[1];
            *xnorm = fabs(X[0]) > fabs(X[1]) ? fabs(X[0]) : fabs(X[1]);
        }
        return;
    }

    /* k == 4: 2 by 2 case
       op[TL11 TL12]*[X11 X12] + ISGN*[X11 X12]*op[TR11 TR12] = [B11 B12]
         [TL21 TL22] [X21 X22]        [X21 X22]   [TR21 TR22]   [B21 B22]

       Solve equivalent 4 by 4 system using complete pivoting. */

    smin = fabs(TR[0]);
    temp = fabs(TR[1]);
    if (temp > smin) smin = temp;
    temp = fabs(TR[ldtr]);
    if (temp > smin) smin = temp;
    temp = fabs(TR[1 + ldtr]);
    if (temp > smin) smin = temp;
    temp = fabs(TL[0]);
    if (temp > smin) smin = temp;
    temp = fabs(TL[1]);
    if (temp > smin) smin = temp;
    temp = fabs(TL[ldtl]);
    if (temp > smin) smin = temp;
    temp = fabs(TL[1 + ldtl]);
    if (temp > smin) smin = temp;
    smin = eps * smin;
    if (smin < smlnum) smin = smlnum;

    /* Initialize T16 to zero */
    for (i = 0; i < 16; i++)
        t16[i] = ZERO;

    /* Fill in T16 (column-major storage) */
    t16[0 + 4*0] = TL[0] + sgn * TR[0];
    t16[1 + 4*1] = TL[1 + ldtl] + sgn * TR[0];
    t16[2 + 4*2] = TL[0] + sgn * TR[1 + ldtr];
    t16[3 + 4*3] = TL[1 + ldtl] + sgn * TR[1 + ldtr];

    if (ltranl) {
        t16[0 + 4*1] = TL[1];
        t16[1 + 4*0] = TL[ldtl];
        t16[2 + 4*3] = TL[1];
        t16[3 + 4*2] = TL[ldtl];
    } else {
        t16[0 + 4*1] = TL[ldtl];
        t16[1 + 4*0] = TL[1];
        t16[2 + 4*3] = TL[ldtl];
        t16[3 + 4*2] = TL[1];
    }
    if (ltranr) {
        t16[0 + 4*2] = sgn * TR[ldtr];
        t16[1 + 4*3] = sgn * TR[ldtr];
        t16[2 + 4*0] = sgn * TR[1];
        t16[3 + 4*1] = sgn * TR[1];
    } else {
        t16[0 + 4*2] = sgn * TR[1];
        t16[1 + 4*3] = sgn * TR[1];
        t16[2 + 4*0] = sgn * TR[ldtr];
        t16[3 + 4*1] = sgn * TR[ldtr];
    }

    btmp[0] = B[0];
    btmp[1] = B[1];
    btmp[2] = B[ldb];
    btmp[3] = B[1 + ldb];

    /* Perform elimination with complete pivoting */
    for (i = 0; i < 3; i++) {
        xmax = ZERO;
        ipsv = i;
        jpsv = i;
        for (ip = i; ip < 4; ip++) {
            for (jp = i; jp < 4; jp++) {
                if (fabs(t16[ip + 4*jp]) >= xmax) {
                    xmax = fabs(t16[ip + 4*jp]);
                    ipsv = ip;
                    jpsv = jp;
                }
            }
        }
        if (ipsv != i) {
            /* Swap rows ipsv and i */
            cblas_dswap(4, &t16[ipsv], 4, &t16[i], 4);
            temp = btmp[i];
            btmp[i] = btmp[ipsv];
            btmp[ipsv] = temp;
        }
        if (jpsv != i) {
            /* Swap columns jpsv and i */
            cblas_dswap(4, &t16[4*jpsv], 1, &t16[4*i], 1);
        }
        jpiv[i] = jpsv;
        if (fabs(t16[i + 4*i]) < smin) {
            *info = 1;
            t16[i + 4*i] = smin;
        }
        for (j = i + 1; j < 4; j++) {
            t16[j + 4*i] = t16[j + 4*i] / t16[i + 4*i];
            btmp[j] = btmp[j] - t16[j + 4*i] * btmp[i];
            for (k = i + 1; k < 4; k++) {
                t16[j + 4*k] = t16[j + 4*k] - t16[j + 4*i] * t16[i + 4*k];
            }
        }
    }
    if (fabs(t16[3 + 4*3]) < smin) {
        *info = 1;
        t16[3 + 4*3] = smin;
    }

    *scale = ONE;
    if ((EIGHT * smlnum) * fabs(btmp[0]) > fabs(t16[0 + 4*0]) ||
        (EIGHT * smlnum) * fabs(btmp[1]) > fabs(t16[1 + 4*1]) ||
        (EIGHT * smlnum) * fabs(btmp[2]) > fabs(t16[2 + 4*2]) ||
        (EIGHT * smlnum) * fabs(btmp[3]) > fabs(t16[3 + 4*3])) {
        xmax = fabs(btmp[0]);
        if (fabs(btmp[1]) > xmax) xmax = fabs(btmp[1]);
        if (fabs(btmp[2]) > xmax) xmax = fabs(btmp[2]);
        if (fabs(btmp[3]) > xmax) xmax = fabs(btmp[3]);
        *scale = (ONE / EIGHT) / xmax;
        btmp[0] = btmp[0] * (*scale);
        btmp[1] = btmp[1] * (*scale);
        btmp[2] = btmp[2] * (*scale);
        btmp[3] = btmp[3] * (*scale);
    }

    /* Back substitution */
    for (i = 0; i < 4; i++) {
        k = 3 - i;
        temp = ONE / t16[k + 4*k];
        tmp[k] = btmp[k] * temp;
        for (j = k + 1; j < 4; j++) {
            tmp[k] = tmp[k] - (temp * t16[k + 4*j]) * tmp[j];
        }
    }

    /* Undo column permutations */
    for (i = 0; i < 3; i++) {
        if (jpiv[2 - i] != 2 - i) {
            temp = tmp[2 - i];
            tmp[2 - i] = tmp[jpiv[2 - i]];
            tmp[jpiv[2 - i]] = temp;
        }
    }

    X[0] = tmp[0];
    X[1] = tmp[1];
    X[ldx] = tmp[2];
    X[1 + ldx] = tmp[3];
    *xnorm = fabs(tmp[0]) + fabs(tmp[2]);
    temp = fabs(tmp[1]) + fabs(tmp[3]);
    if (temp > *xnorm) *xnorm = temp;
}
