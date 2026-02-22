/**
 * @file dlasv2.c
 * @brief DLASV2 computes the singular value decomposition of a 2-by-2 triangular matrix.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * DLASV2 computes the singular value decomposition of a 2-by-2
 * triangular matrix
 *    [  F   G  ]
 *    [  0   H  ].
 * On return, abs(SSMAX) is the larger singular value, abs(SSMIN) is the
 * smaller singular value, and (CSL,SNL) and (CSR,SNR) are the left and
 * right singular vectors for abs(SSMAX), giving the decomposition
 *
 *    [ CSL  SNL ] [  F   G  ] [ CSR -SNR ]  =  [ SSMAX   0   ]
 *    [-SNL  CSL ] [  0   H  ] [ SNR  CSR ]     [  0    SSMIN ].
 *
 * @param[in]  f      The (1,1) element of the 2-by-2 matrix.
 * @param[in]  g      The (1,2) element of the 2-by-2 matrix.
 * @param[in]  h      The (2,2) element of the 2-by-2 matrix.
 * @param[out] ssmin  abs(SSMIN) is the smaller singular value.
 * @param[out] ssmax  abs(SSMAX) is the larger singular value.
 * @param[out] snr    The vector (CSR, SNR) is a unit right singular vector
 *                    for the singular value abs(SSMAX).
 * @param[out] csr    The vector (CSR, SNR) is a unit right singular vector
 *                    for the singular value abs(SSMAX).
 * @param[out] snl    The vector (CSL, SNL) is a unit left singular vector
 *                    for the singular value abs(SSMAX).
 * @param[out] csl    The vector (CSL, SNL) is a unit left singular vector
 *                    for the singular value abs(SSMAX).
 */
void dlasv2(const f64 f, const f64 g, const f64 h,
            f64* ssmin, f64* ssmax,
            f64* snr, f64* csr, f64* snl, f64* csl)
{
    INT gasmal, swap, pmax;
    f64 a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m;
    f64 mm, r, s, slt, srt, t, temp, tsign, tt;
    f64 eps;

    ft = f;
    fa = fabs(ft);
    ht = h;
    ha = fabs(h);

    /*
     * PMAX points to the maximum absolute element of matrix
     *   PMAX = 1 if F largest in absolute values
     *   PMAX = 2 if G largest in absolute values
     *   PMAX = 3 if H largest in absolute values
     */
    pmax = 1;
    swap = (ha > fa);
    if (swap) {
        pmax = 3;
        temp = ft;
        ft = ht;
        ht = temp;
        temp = fa;
        fa = ha;
        ha = temp;
        /* Now FA >= HA */
    }
    gt = g;
    ga = fabs(gt);

    if (ga == 0.0) {
        /* Diagonal matrix */
        *ssmin = ha;
        *ssmax = fa;
        clt = 1.0;
        crt = 1.0;
        slt = 0.0;
        srt = 0.0;
    } else {
        gasmal = 1;
        if (ga > fa) {
            pmax = 2;
            eps = dlamch("E");
            if ((fa / ga) < eps) {
                /* Case of very large GA */
                gasmal = 0;
                *ssmax = ga;
                if (ha > 1.0) {
                    *ssmin = fa / (ga / ha);
                } else {
                    *ssmin = (fa / ga) * ha;
                }
                clt = 1.0;
                slt = ht / gt;
                srt = 1.0;
                crt = ft / gt;
            }
        }
        if (gasmal) {
            /* Normal case */
            d = fa - ha;
            if (d == fa) {
                /* Copes with infinite F or H */
                l = 1.0;
            } else {
                l = d / fa;
            }

            /* Note that 0 <= L <= 1 */
            m = gt / ft;

            /* Note that abs(M) <= 1/macheps */
            t = 2.0 - l;

            /* Note that T >= 1 */
            mm = m * m;
            tt = t * t;
            s = sqrt(tt + mm);

            /* Note that 1 <= S <= 1 + 1/macheps */
            if (l == 0.0) {
                r = fabs(m);
            } else {
                r = sqrt(l * l + mm);
            }

            /* Note that 0 <= R <= 1 + 1/macheps */
            a = 0.5 * (s + r);

            /* Note that 1 <= A <= 1 + abs(M) */
            *ssmin = ha / a;
            *ssmax = fa * a;

            if (mm == 0.0) {
                /* Note that M is very tiny */
                if (l == 0.0) {
                    t = copysign(2.0, ft) * copysign(1.0, gt);
                } else {
                    t = gt / copysign(d, ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0 + a);
            }
            l = sqrt(t * t + 4.0);
            crt = 2.0 / l;
            srt = t / l;
            clt = (crt + srt * m) / a;
            slt = (ht / ft) * srt / a;
        }
    }

    if (swap) {
        *csl = srt;
        *snl = crt;
        *csr = slt;
        *snr = clt;
    } else {
        *csl = clt;
        *snl = slt;
        *csr = crt;
        *snr = srt;
    }

    /* Correct signs of SSMAX and SSMIN */
    if (pmax == 1) {
        tsign = copysign(1.0, *csr) * copysign(1.0, *csl) * copysign(1.0, f);
    } else if (pmax == 2) {
        tsign = copysign(1.0, *snr) * copysign(1.0, *csl) * copysign(1.0, g);
    } else {  /* pmax == 3 */
        tsign = copysign(1.0, *snr) * copysign(1.0, *snl) * copysign(1.0, h);
    }
    *ssmax = copysign(*ssmax, tsign);
    *ssmin = copysign(*ssmin, tsign * copysign(1.0, f) * copysign(1.0, h));
}
