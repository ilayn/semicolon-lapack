/**
 * @file slaic1.c
 * @brief SLAIC1 applies one step of incremental condition estimation.
 */

#include <math.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SLAIC1 applies one step of incremental condition estimation in
 * its simplest version:
 *
 * Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j
 * lower triangular matrix L, such that
 *          twonorm(L*x) = sest
 * Then SLAIC1 computes sestpr, s, c such that
 * the vector
 *                 [ s*x ]
 *          xhat = [  c  ]
 * is an approximate singular vector of
 *                 [ L       0  ]
 *          Lhat = [ w**T gamma ]
 * in the sense that
 *          twonorm(Lhat*xhat) = sestpr.
 *
 * Depending on JOB, an estimate for the largest or smallest singular
 * value is computed.
 *
 * Note that [s c]**T and sestpr**2 is an eigenpair of the system
 *
 *     diag(sest*sest, 0) + [alpha  gamma] * [ alpha ]
 *                                           [ gamma ]
 *
 * where  alpha =  x**T*w.
 *
 * @param[in]     job    = 1: an estimate for the largest singular value is computed.
 *                         = 2: an estimate for the smallest singular value is computed.
 * @param[in]     j      Length of x and w.
 * @param[in]     x      Double precision array, dimension (j).
 *                        The j-vector x.
 * @param[in]     sest   Estimated singular value of j by j matrix L.
 * @param[in]     w      Double precision array, dimension (j).
 *                        The j-vector w.
 * @param[in]     gamma  The diagonal element gamma.
 * @param[out]    sestpr Estimated singular value of (j+1) by (j+1) matrix Lhat.
 * @param[out]    s      Sine needed in forming xhat.
 * @param[out]    c      Cosine needed in forming xhat.
 */
void slaic1(
    const INT job,
    const INT j,
    const f32* restrict x,
    const f32 sest,
    const f32* restrict w,
    const f32 gamma,
    f32* sestpr,
    f32* s,
    f32* c)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 HALF = 0.5f;
    const f32 FOUR = 4.0f;

    f32 absalp, absest, absgam, alpha, b, cosine, eps,
           norma, s1, s2, sine, t, test, tmp, zeta1, zeta2;

    eps = FLT_EPSILON;
    alpha = cblas_sdot(j, x, 1, w, 1);

    absalp = fabsf(alpha);
    absgam = fabsf(gamma);
    absest = fabsf(sest);

    if (job == 1) {

        /* Estimating largest singular value */

        /* special cases */

        if (sest == ZERO) {
            s1 = fmaxf(absgam, absalp);
            if (s1 == ZERO) {
                *s = ZERO;
                *c = ONE;
                *sestpr = ZERO;
            } else {
                *s = alpha / s1;
                *c = gamma / s1;
                tmp = sqrtf((*s) * (*s) + (*c) * (*c));
                *s = *s / tmp;
                *c = *c / tmp;
                *sestpr = s1 * tmp;
            }
            return;
        } else if (absgam <= eps * absest) {
            *s = ONE;
            *c = ZERO;
            tmp = fmaxf(absest, absalp);
            s1 = absest / tmp;
            s2 = absalp / tmp;
            *sestpr = tmp * sqrtf(s1 * s1 + s2 * s2);
            return;
        } else if (absalp <= eps * absest) {
            s1 = absgam;
            s2 = absest;
            if (s1 <= s2) {
                *s = ONE;
                *c = ZERO;
                *sestpr = s2;
            } else {
                *s = ZERO;
                *c = ONE;
                *sestpr = s1;
            }
            return;
        } else if (absest <= eps * absalp || absest <= eps * absgam) {
            s1 = absgam;
            s2 = absalp;
            if (s1 <= s2) {
                tmp = s1 / s2;
                *s = sqrtf(ONE + tmp * tmp);
                *sestpr = s2 * (*s);
                *c = (gamma / s2) / (*s);
                *s = copysignf(ONE, alpha) / (*s);
            } else {
                tmp = s2 / s1;
                *c = sqrtf(ONE + tmp * tmp);
                *sestpr = s1 * (*c);
                *s = (alpha / s1) / (*c);
                *c = copysignf(ONE, gamma) / (*c);
            }
            return;
        } else {

            /* normal case */

            zeta1 = alpha / absest;
            zeta2 = gamma / absest;

            b = (ONE - zeta1 * zeta1 - zeta2 * zeta2) * HALF;
            *c = zeta1 * zeta1;
            if (b > ZERO) {
                t = *c / (b + sqrtf(b * b + *c));
            } else {
                t = sqrtf(b * b + *c) - b;
            }

            sine = -zeta1 / t;
            cosine = -zeta2 / (ONE + t);
            tmp = sqrtf(sine * sine + cosine * cosine);
            *s = sine / tmp;
            *c = cosine / tmp;
            *sestpr = sqrtf(t + ONE) * absest;
            return;
        }

    } else if (job == 2) {

        /* Estimating smallest singular value */

        /* special cases */

        if (sest == ZERO) {
            *sestpr = ZERO;
            if (fmaxf(absgam, absalp) == ZERO) {
                sine = ONE;
                cosine = ZERO;
            } else {
                sine = -gamma;
                cosine = alpha;
            }
            s1 = fmaxf(fabsf(sine), fabsf(cosine));
            *s = sine / s1;
            *c = cosine / s1;
            tmp = sqrtf((*s) * (*s) + (*c) * (*c));
            *s = *s / tmp;
            *c = *c / tmp;
            return;
        } else if (absgam <= eps * absest) {
            *s = ZERO;
            *c = ONE;
            *sestpr = absgam;
            return;
        } else if (absalp <= eps * absest) {
            s1 = absgam;
            s2 = absest;
            if (s1 <= s2) {
                *s = ZERO;
                *c = ONE;
                *sestpr = s1;
            } else {
                *s = ONE;
                *c = ZERO;
                *sestpr = s2;
            }
            return;
        } else if (absest <= eps * absalp || absest <= eps * absgam) {
            s1 = absgam;
            s2 = absalp;
            if (s1 <= s2) {
                tmp = s1 / s2;
                *c = sqrtf(ONE + tmp * tmp);
                *sestpr = absest * (tmp / *c);
                *s = -(gamma / s2) / *c;
                *c = copysignf(ONE, alpha) / *c;
            } else {
                tmp = s2 / s1;
                *s = sqrtf(ONE + tmp * tmp);
                *sestpr = absest / *s;
                *c = (alpha / s1) / *s;
                *s = -copysignf(ONE, gamma) / *s;
            }
            return;
        } else {

            /* normal case */

            zeta1 = alpha / absest;
            zeta2 = gamma / absest;

            norma = fmaxf(ONE + zeta1 * zeta1 + fabsf(zeta1 * zeta2),
                         fabsf(zeta1 * zeta2) + zeta2 * zeta2);

            /* See if root is closer to zero or to ONE */

            test = ONE + TWO * (zeta1 - zeta2) * (zeta1 + zeta2);
            if (test >= ZERO) {

                /* root is close to zero, compute directly */

                b = (zeta1 * zeta1 + zeta2 * zeta2 + ONE) * HALF;
                *c = zeta2 * zeta2;
                t = *c / (b + sqrtf(fabsf(b * b - *c)));
                sine = zeta1 / (ONE - t);
                cosine = -zeta2 / t;
                *sestpr = sqrtf(t + FOUR * eps * eps * norma) * absest;
            } else {

                /* root is closer to ONE, shift by that amount */

                b = (zeta2 * zeta2 + zeta1 * zeta1 - ONE) * HALF;
                *c = zeta1 * zeta1;
                if (b >= ZERO) {
                    t = -*c / (b + sqrtf(b * b + *c));
                } else {
                    t = b - sqrtf(b * b + *c);
                }
                sine = -zeta1 / t;
                cosine = -zeta2 / (ONE + t);
                *sestpr = sqrtf(ONE + t + FOUR * eps * eps * norma) * absest;
            }
            tmp = sqrtf(sine * sine + cosine * cosine);
            *s = sine / tmp;
            *c = cosine / tmp;
            return;

        }
    }
    return;
}
