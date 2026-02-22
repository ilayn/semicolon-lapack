/**
 * @file dlag2.c
 * @brief DLAG2 computes the eigenvalues of a 2-by-2 generalized eigenvalue problem.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLAG2 computes the eigenvalues of a 2 x 2 generalized eigenvalue
 * problem  A - w B, with scaling as necessary to avoid over-/underflow.
 *
 * The scaling factor "s" results in a modified eigenvalue equation
 *
 *     s A - w B
 *
 * where  s  is a non-negative scaling factor chosen so that  w,  w B,
 * and  s A  do not overflow and, if possible, do not underflow, either.
 *
 * @param[in]     A       2x2 matrix A. It is assumed that its 1-norm
 *                        is less than 1/SAFMIN.
 * @param[in]     lda     Leading dimension of A. lda >= 2.
 * @param[in]     B       2x2 upper triangular matrix B.
 * @param[in]     ldb     Leading dimension of B. ldb >= 2.
 * @param[in]     safmin  The smallest positive number s.t. 1/SAFMIN does not overflow.
 * @param[out]    scale1  Scaling factor for the first eigenvalue.
 * @param[out]    scale2  Scaling factor for the second eigenvalue.
 * @param[out]    wr1     Real part of the first eigenvalue (scaled by scale1).
 * @param[out]    wr2     Real part of the second eigenvalue (scaled by scale2).
 * @param[out]    wi      Imaginary part of eigenvalues (scaled by scale1). Non-negative.
 */
void dlag2(
    const f64* restrict A,
    const INT lda,
    const f64* restrict B,
    const INT ldb,
    const f64 safmin,
    f64* scale1,
    f64* scale2,
    f64* wr1,
    f64* wr2,
    f64* wi)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 HALF = ONE / TWO;
    const f64 FUZZY1 = ONE + 1.0e-5;

    f64 a11, a12, a21, a22, abi22, anorm, as11, as12;
    f64 as22, ascale, b11, b12, b22, binv11, binv22;
    f64 bmin, bnorm, bscale, bsize, c1, c2, c3, c4, c5;
    f64 diff, discr, pp, qq, r, rtmax, rtmin, s1, s2;
    f64 safmax, shift, ss, sum, wabs, wbig, wdet;
    f64 wscale, wsize, wsmall;

    rtmin = sqrt(safmin);
    rtmax = ONE / rtmin;
    safmax = ONE / safmin;

    anorm = fabs(A[0 + 0 * lda]) + fabs(A[1 + 0 * lda]);
    {
        f64 tmp = fabs(A[0 + 1 * lda]) + fabs(A[1 + 1 * lda]);
        if (tmp > anorm) anorm = tmp;
    }
    if (safmin > anorm) anorm = safmin;
    ascale = ONE / anorm;
    a11 = ascale * A[0 + 0 * lda];
    a21 = ascale * A[1 + 0 * lda];
    a12 = ascale * A[0 + 1 * lda];
    a22 = ascale * A[1 + 1 * lda];

    b11 = B[0 + 0 * ldb];
    b12 = B[0 + 1 * ldb];
    b22 = B[1 + 1 * ldb];
    {
        f64 tmp1 = fabs(b11);
        f64 tmp2 = fabs(b12);
        f64 tmp3 = fabs(b22);
        f64 tmp = tmp1;
        if (tmp2 > tmp) tmp = tmp2;
        if (tmp3 > tmp) tmp = tmp3;
        if (rtmin > tmp) tmp = rtmin;
        bmin = rtmin * tmp;
    }
    if (fabs(b11) < bmin)
        b11 = (b11 >= ZERO) ? bmin : -bmin;
    if (fabs(b22) < bmin)
        b22 = (b22 >= ZERO) ? bmin : -bmin;

    {
        f64 tmp = fabs(b12) + fabs(b22);
        bnorm = fabs(b11);
        if (tmp > bnorm) bnorm = tmp;
        if (safmin > bnorm) bnorm = safmin;
    }
    bsize = fabs(b11);
    if (fabs(b22) > bsize) bsize = fabs(b22);
    bscale = ONE / bsize;
    b11 = b11 * bscale;
    b12 = b12 * bscale;
    b22 = b22 * bscale;

    binv11 = ONE / b11;
    binv22 = ONE / b22;
    s1 = a11 * binv11;
    s2 = a22 * binv22;
    if (fabs(s1) <= fabs(s2)) {
        as12 = a12 - s1 * b12;
        as22 = a22 - s1 * b22;
        ss = a21 * (binv11 * binv22);
        abi22 = as22 * binv22 - ss * b12;
        pp = HALF * abi22;
        shift = s1;
    } else {
        as12 = a12 - s2 * b12;
        as11 = a11 - s2 * b11;
        ss = a21 * (binv11 * binv22);
        abi22 = -ss * b12;
        pp = HALF * (as11 * binv11 + abi22);
        shift = s2;
    }
    qq = ss * as12;
    if (fabs(pp * rtmin) >= ONE) {
        discr = (rtmin * pp) * (rtmin * pp) + qq * safmin;
        r = sqrt(fabs(discr)) * rtmax;
    } else {
        if (pp * pp + fabs(qq) <= safmin) {
            discr = (rtmax * pp) * (rtmax * pp) + qq * safmax;
            r = sqrt(fabs(discr)) * rtmin;
        } else {
            discr = pp * pp + qq;
            r = sqrt(fabs(discr));
        }
    }

    if (discr >= ZERO || r == ZERO) {
        sum = pp + ((pp >= ZERO) ? r : -r);
        diff = pp - ((pp >= ZERO) ? r : -r);
        wbig = shift + sum;

        wsmall = shift + diff;
        if (HALF * fabs(wbig) > fabs(wsmall) && HALF * fabs(wbig) > safmin) {
            wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
            wsmall = wdet / wbig;
        } else if (safmin > fabs(wsmall)) {
            wdet = (a11 * a22 - a12 * a21) * (binv11 * binv22);
            if (HALF * fabs(wbig) > safmin)
                wsmall = wdet / wbig;
        }

        if (pp > abi22) {
            *wr1 = (wbig < wsmall) ? wbig : wsmall;
            *wr2 = (wbig > wsmall) ? wbig : wsmall;
        } else {
            *wr1 = (wbig > wsmall) ? wbig : wsmall;
            *wr2 = (wbig < wsmall) ? wbig : wsmall;
        }
        *wi = ZERO;
    } else {
        *wr1 = shift + pp;
        *wr2 = *wr1;
        *wi = r;
    }

    c1 = bsize * (safmin * (ONE > ascale ? ONE : ascale));
    c2 = safmin * (ONE > bnorm ? ONE : bnorm);
    c3 = bsize * safmin;
    if (ascale <= ONE && bsize <= ONE) {
        f64 tmp = ascale / safmin;
        c4 = (ONE < tmp * bsize) ? ONE : tmp * bsize;
    } else {
        c4 = ONE;
    }
    if (ascale <= ONE || bsize <= ONE) {
        c5 = (ONE < ascale * bsize) ? ONE : ascale * bsize;
    } else {
        c5 = ONE;
    }

    wabs = fabs(*wr1) + fabs(*wi);
    {
        f64 tmp1 = wabs * c2 + c3;
        f64 tmp2 = (wabs > c5) ? wabs : c5;
        f64 tmp3 = (c4 < HALF * tmp2) ? c4 : HALF * tmp2;
        wsize = safmin;
        if (c1 > wsize) wsize = c1;
        if (FUZZY1 * tmp1 > wsize) wsize = FUZZY1 * tmp1;
        if (tmp3 > wsize) wsize = tmp3;
    }
    if (wsize != ONE) {
        wscale = ONE / wsize;
        if (wsize > ONE) {
            f64 tmp1 = (ascale > bsize) ? ascale : bsize;
            f64 tmp2 = (ascale < bsize) ? ascale : bsize;
            *scale1 = (tmp1 * wscale) * tmp2;
        } else {
            f64 tmp1 = (ascale < bsize) ? ascale : bsize;
            f64 tmp2 = (ascale > bsize) ? ascale : bsize;
            *scale1 = (tmp1 * wscale) * tmp2;
        }
        *wr1 = *wr1 * wscale;
        if (*wi != ZERO) {
            *wi = *wi * wscale;
            *wr2 = *wr1;
            *scale2 = *scale1;
        }
    } else {
        *scale1 = ascale * bsize;
        *scale2 = *scale1;
    }

    if (*wi == ZERO) {
        wabs = fabs(*wr2);
        {
            f64 tmp1 = wabs * c2 + c3;
            f64 tmp2 = (wabs > c5) ? wabs : c5;
            f64 tmp3 = (c4 < HALF * tmp2) ? c4 : HALF * tmp2;
            wsize = safmin;
            if (c1 > wsize) wsize = c1;
            if (FUZZY1 * tmp1 > wsize) wsize = FUZZY1 * tmp1;
            if (tmp3 > wsize) wsize = tmp3;
        }
        if (wsize != ONE) {
            wscale = ONE / wsize;
            if (wsize > ONE) {
                f64 tmp1 = (ascale > bsize) ? ascale : bsize;
                f64 tmp2 = (ascale < bsize) ? ascale : bsize;
                *scale2 = (tmp1 * wscale) * tmp2;
            } else {
                f64 tmp1 = (ascale < bsize) ? ascale : bsize;
                f64 tmp2 = (ascale > bsize) ? ascale : bsize;
                *scale2 = (tmp1 * wscale) * tmp2;
            }
            *wr2 = *wr2 * wscale;
        } else {
            *scale2 = ascale * bsize;
        }
    }
}
