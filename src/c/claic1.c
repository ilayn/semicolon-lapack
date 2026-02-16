/**
 * @file claic1.c
 * @brief CLAIC1 applies one step of incremental condition estimation.
 */

#include <math.h>
#include <complex.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLAIC1 applies one step of incremental condition estimation in
 * its simplest version:
 *
 * Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j
 * lower triangular matrix L, such that
 *          twonorm(L*x) = sest
 * Then CLAIC1 computes sestpr, s, c such that
 * the vector
 *                 [ s*x ]
 *          xhat = [  c  ]
 * is an approximate singular vector of
 *                 [ L       0  ]
 *          Lhat = [ w**H gamma ]
 * in the sense that
 *          twonorm(Lhat*xhat) = sestpr.
 *
 * Depending on JOB, an estimate for the largest or smallest singular
 * value is computed.
 *
 * Note that [s c]**H and sestpr**2 is an eigenpair of the system
 *
 *     diag(sest*sest, 0) + [alpha  gamma] * [ conjg(alpha) ]
 *                                           [ conjg(gamma) ]
 *
 * where  alpha =  x**H * w.
 *
 * @param[in]     job    = 1: an estimate for the largest singular value is computed.
 *                         = 2: an estimate for the smallest singular value is computed.
 * @param[in]     j      Length of x and w.
 * @param[in]     x      Complex*16 array, dimension (j).
 *                        The j-vector x.
 * @param[in]     sest   Estimated singular value of j by j matrix L.
 * @param[in]     w      Complex*16 array, dimension (j).
 *                        The j-vector w.
 * @param[in]     gamma_ The diagonal element gamma.
 * @param[out]    sestpr Estimated singular value of (j+1) by (j+1) matrix Lhat.
 * @param[out]    s      Sine needed in forming xhat.
 * @param[out]    c      Cosine needed in forming xhat.
 */
void claic1(
    const int job,
    const int j,
    const c64* restrict x,
    const f32 sest,
    const c64* restrict w,
    const c64 gamma_,
    f32* sestpr,
    c64* s,
    c64* c)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 HALF = 0.5f;
    const f32 FOUR = 4.0f;

    f32 absalp, absest, absgam, b, eps,
           norma, s1, s2, scl, t, test, tmp, zeta1, zeta2;
    c64 alpha, cosine, sine;

    eps = FLT_EPSILON;
    cblas_cdotc_sub(j, x, 1, w, 1, &alpha);

    absalp = cabsf(alpha);
    absgam = cabsf(gamma_);
    absest = fabsf(sest);

    if (job == 1) {

        /* Estimating largest singular value */

        /* special cases */

        if (sest == ZERO) {
            s1 = fmaxf(absgam, absalp);
            if (s1 == ZERO) {
                *s = CMPLXF(ZERO, 0.0f);
                *c = CMPLXF(ONE, 0.0f);
                *sestpr = ZERO;
            } else {
                *s = alpha / s1;
                *c = gamma_ / s1;
                tmp = crealf(csqrtf((*s) * conjf(*s) + (*c) * conjf(*c)));
                *s = *s / tmp;
                *c = *c / tmp;
                *sestpr = s1 * tmp;
            }
            return;
        } else if (absgam <= eps * absest) {
            *s = CMPLXF(ONE, 0.0f);
            *c = CMPLXF(ZERO, 0.0f);
            tmp = fmaxf(absest, absalp);
            s1 = absest / tmp;
            s2 = absalp / tmp;
            *sestpr = tmp * sqrtf(s1 * s1 + s2 * s2);
            return;
        } else if (absalp <= eps * absest) {
            s1 = absgam;
            s2 = absest;
            if (s1 <= s2) {
                *s = CMPLXF(ONE, 0.0f);
                *c = CMPLXF(ZERO, 0.0f);
                *sestpr = s2;
            } else {
                *s = CMPLXF(ZERO, 0.0f);
                *c = CMPLXF(ONE, 0.0f);
                *sestpr = s1;
            }
            return;
        } else if (absest <= eps * absalp || absest <= eps * absgam) {
            s1 = absgam;
            s2 = absalp;
            if (s1 <= s2) {
                tmp = s1 / s2;
                scl = sqrtf(ONE + tmp * tmp);
                *sestpr = s2 * scl;
                *s = (alpha / s2) / scl;
                *c = (gamma_ / s2) / scl;
            } else {
                tmp = s2 / s1;
                scl = sqrtf(ONE + tmp * tmp);
                *sestpr = s1 * scl;
                *s = (alpha / s1) / scl;
                *c = (gamma_ / s1) / scl;
            }
            return;
        } else {

            /* normal case */

            zeta1 = absalp / absest;
            zeta2 = absgam / absest;

            b = (ONE - zeta1 * zeta1 - zeta2 * zeta2) * HALF;
            t = zeta1 * zeta1;
            if (b > ZERO) {
                t = t / (b + sqrtf(b * b + t));
            } else {
                t = sqrtf(b * b + t) - b;
            }

            sine = -(alpha / absest) / t;
            cosine = -(gamma_ / absest) / (ONE + t);
            tmp = crealf(csqrtf(sine * conjf(sine)
                + cosine * conjf(cosine)));

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
                sine = CMPLXF(ONE, 0.0f);
                cosine = CMPLXF(ZERO, 0.0f);
            } else {
                sine = -conjf(gamma_);
                cosine = conjf(alpha);
            }
            s1 = fmaxf(cabsf(sine), cabsf(cosine));
            *s = sine / s1;
            *c = cosine / s1;
            tmp = crealf(csqrtf((*s) * conjf(*s) + (*c) * conjf(*c)));
            *s = *s / tmp;
            *c = *c / tmp;
            return;
        } else if (absgam <= eps * absest) {
            *s = CMPLXF(ZERO, 0.0f);
            *c = CMPLXF(ONE, 0.0f);
            *sestpr = absgam;
            return;
        } else if (absalp <= eps * absest) {
            s1 = absgam;
            s2 = absest;
            if (s1 <= s2) {
                *s = CMPLXF(ZERO, 0.0f);
                *c = CMPLXF(ONE, 0.0f);
                *sestpr = s1;
            } else {
                *s = CMPLXF(ONE, 0.0f);
                *c = CMPLXF(ZERO, 0.0f);
                *sestpr = s2;
            }
            return;
        } else if (absest <= eps * absalp || absest <= eps * absgam) {
            s1 = absgam;
            s2 = absalp;
            if (s1 <= s2) {
                tmp = s1 / s2;
                scl = sqrtf(ONE + tmp * tmp);
                *sestpr = absest * (tmp / scl);
                *s = -(conjf(gamma_) / s2) / scl;
                *c = (conjf(alpha) / s2) / scl;
            } else {
                tmp = s2 / s1;
                scl = sqrtf(ONE + tmp * tmp);
                *sestpr = absest / scl;
                *s = -(conjf(gamma_) / s1) / scl;
                *c = (conjf(alpha) / s1) / scl;
            }
            return;
        } else {

            /* normal case */

            zeta1 = absalp / absest;
            zeta2 = absgam / absest;

            norma = fmaxf(ONE + zeta1 * zeta1 + zeta1 * zeta2,
                         zeta1 * zeta2 + zeta2 * zeta2);

            /* See if root is closer to zero or to ONE */

            test = ONE + TWO * (zeta1 - zeta2) * (zeta1 + zeta2);
            if (test >= ZERO) {

                /* root is close to zero, compute directly */

                b = (zeta1 * zeta1 + zeta2 * zeta2 + ONE) * HALF;
                t = zeta2 * zeta2;
                t = t / (b + sqrtf(fabsf(b * b - t)));
                sine = (alpha / absest) / (ONE - t);
                cosine = -(gamma_ / absest) / t;
                *sestpr = sqrtf(t + FOUR * eps * eps * norma) * absest;
            } else {

                /* root is closer to ONE, shift by that amount */

                b = (zeta2 * zeta2 + zeta1 * zeta1 - ONE) * HALF;
                t = zeta1 * zeta1;
                if (b >= ZERO) {
                    t = -t / (b + sqrtf(b * b + t));
                } else {
                    t = b - sqrtf(b * b + t);
                }
                sine = -(alpha / absest) / t;
                cosine = -(gamma_ / absest) / (ONE + t);
                *sestpr = sqrtf(ONE + t + FOUR * eps * eps * norma) * absest;
            }
            tmp = crealf(csqrtf(sine * conjf(sine)
                + cosine * conjf(cosine)));
            *s = sine / tmp;
            *c = cosine / tmp;
            return;

        }
    }
    return;
}
