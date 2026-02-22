/**
 * @file slasv2.c
 * @brief SLASV2 computes the singular value decomposition of a 2-by-2 triangular matrix.
 */

#include "semicolon_lapack_single.h"
#include <math.h>

/**
 * SLASV2 computes the singular value decomposition of a 2-by-2
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
void slasv2(const f32 f, const f32 g, const f32 h,
            f32* ssmin, f32* ssmax,
            f32* snr, f32* csr, f32* snl, f32* csl)
{
    INT gasmal, swap, pmax;
    f32 a, clt, crt, d, fa, ft, ga, gt, ha, ht, l, m;
    f32 mm, r, s, slt, srt, t, temp, tsign, tt;
    f32 eps;

    ft = f;
    fa = fabsf(ft);
    ht = h;
    ha = fabsf(h);

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
    ga = fabsf(gt);

    if (ga == 0.0f) {
        /* Diagonal matrix */
        *ssmin = ha;
        *ssmax = fa;
        clt = 1.0f;
        crt = 1.0f;
        slt = 0.0f;
        srt = 0.0f;
    } else {
        gasmal = 1;
        if (ga > fa) {
            pmax = 2;
            eps = slamch("E");
            if ((fa / ga) < eps) {
                /* Case of very large GA */
                gasmal = 0;
                *ssmax = ga;
                if (ha > 1.0f) {
                    *ssmin = fa / (ga / ha);
                } else {
                    *ssmin = (fa / ga) * ha;
                }
                clt = 1.0f;
                slt = ht / gt;
                srt = 1.0f;
                crt = ft / gt;
            }
        }
        if (gasmal) {
            /* Normal case */
            d = fa - ha;
            if (d == fa) {
                /* Copes with infinite F or H */
                l = 1.0f;
            } else {
                l = d / fa;
            }

            /* Note that 0 <= L <= 1 */
            m = gt / ft;

            /* Note that abs(M) <= 1/macheps */
            t = 2.0f - l;

            /* Note that T >= 1 */
            mm = m * m;
            tt = t * t;
            s = sqrtf(tt + mm);

            /* Note that 1 <= S <= 1 + 1/macheps */
            if (l == 0.0f) {
                r = fabsf(m);
            } else {
                r = sqrtf(l * l + mm);
            }

            /* Note that 0 <= R <= 1 + 1/macheps */
            a = 0.5f * (s + r);

            /* Note that 1 <= A <= 1 + abs(M) */
            *ssmin = ha / a;
            *ssmax = fa * a;

            if (mm == 0.0f) {
                /* Note that M is very tiny */
                if (l == 0.0f) {
                    t = copysignf(2.0f, ft) * copysignf(1.0f, gt);
                } else {
                    t = gt / copysignf(d, ft) + m / t;
                }
            } else {
                t = (m / (s + t) + m / (r + l)) * (1.0f + a);
            }
            l = sqrtf(t * t + 4.0f);
            crt = 2.0f / l;
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
        tsign = copysignf(1.0f, *csr) * copysignf(1.0f, *csl) * copysignf(1.0f, f);
    } else if (pmax == 2) {
        tsign = copysignf(1.0f, *snr) * copysignf(1.0f, *csl) * copysignf(1.0f, g);
    } else {  /* pmax == 3 */
        tsign = copysignf(1.0f, *snr) * copysignf(1.0f, *snl) * copysignf(1.0f, h);
    }
    *ssmax = copysignf(*ssmax, tsign);
    *ssmin = copysignf(*ssmin, tsign * copysignf(1.0f, f) * copysignf(1.0f, h));
}
