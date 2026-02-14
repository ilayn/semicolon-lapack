/**
 * @file dlaic1.c
 * @brief DLAIC1 applies one step of incremental condition estimation.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLAIC1 applies one step of incremental condition estimation in
 * its simplest version:
 *
 * Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j
 * lower triangular matrix L, such that
 *          twonorm(L*x) = sest
 * Then DLAIC1 computes sestpr, s, c such that
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
void dlaic1(
    const int job,
    const int j,
    const f64* restrict x,
    const f64 sest,
    const f64* restrict w,
    const f64 gamma,
    f64* sestpr,
    f64* s,
    f64* c)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 HALF = 0.5;
    const f64 FOUR = 4.0;

    f64 absalp, absest, absgam, alpha, b, cosine, eps,
           norma, s1, s2, sine, t, test, tmp, zeta1, zeta2;

    eps = DBL_EPSILON;
    alpha = cblas_ddot(j, x, 1, w, 1);

    absalp = fabs(alpha);
    absgam = fabs(gamma);
    absest = fabs(sest);

    if (job == 1) {

        /* Estimating largest singular value */

        /* special cases */

        if (sest == ZERO) {
            s1 = fmax(absgam, absalp);
            if (s1 == ZERO) {
                *s = ZERO;
                *c = ONE;
                *sestpr = ZERO;
            } else {
                *s = alpha / s1;
                *c = gamma / s1;
                tmp = sqrt((*s) * (*s) + (*c) * (*c));
                *s = *s / tmp;
                *c = *c / tmp;
                *sestpr = s1 * tmp;
            }
            return;
        } else if (absgam <= eps * absest) {
            *s = ONE;
            *c = ZERO;
            tmp = fmax(absest, absalp);
            s1 = absest / tmp;
            s2 = absalp / tmp;
            *sestpr = tmp * sqrt(s1 * s1 + s2 * s2);
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
                *s = sqrt(ONE + tmp * tmp);
                *sestpr = s2 * (*s);
                *c = (gamma / s2) / (*s);
                *s = copysign(ONE, alpha) / (*s);
            } else {
                tmp = s2 / s1;
                *c = sqrt(ONE + tmp * tmp);
                *sestpr = s1 * (*c);
                *s = (alpha / s1) / (*c);
                *c = copysign(ONE, gamma) / (*c);
            }
            return;
        } else {

            /* normal case */

            zeta1 = alpha / absest;
            zeta2 = gamma / absest;

            b = (ONE - zeta1 * zeta1 - zeta2 * zeta2) * HALF;
            *c = zeta1 * zeta1;
            if (b > ZERO) {
                t = *c / (b + sqrt(b * b + *c));
            } else {
                t = sqrt(b * b + *c) - b;
            }

            sine = -zeta1 / t;
            cosine = -zeta2 / (ONE + t);
            tmp = sqrt(sine * sine + cosine * cosine);
            *s = sine / tmp;
            *c = cosine / tmp;
            *sestpr = sqrt(t + ONE) * absest;
            return;
        }

    } else if (job == 2) {

        /* Estimating smallest singular value */

        /* special cases */

        if (sest == ZERO) {
            *sestpr = ZERO;
            if (fmax(absgam, absalp) == ZERO) {
                sine = ONE;
                cosine = ZERO;
            } else {
                sine = -gamma;
                cosine = alpha;
            }
            s1 = fmax(fabs(sine), fabs(cosine));
            *s = sine / s1;
            *c = cosine / s1;
            tmp = sqrt((*s) * (*s) + (*c) * (*c));
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
                *c = sqrt(ONE + tmp * tmp);
                *sestpr = absest * (tmp / *c);
                *s = -(gamma / s2) / *c;
                *c = copysign(ONE, alpha) / *c;
            } else {
                tmp = s2 / s1;
                *s = sqrt(ONE + tmp * tmp);
                *sestpr = absest / *s;
                *c = (alpha / s1) / *s;
                *s = -copysign(ONE, gamma) / *s;
            }
            return;
        } else {

            /* normal case */

            zeta1 = alpha / absest;
            zeta2 = gamma / absest;

            norma = fmax(ONE + zeta1 * zeta1 + fabs(zeta1 * zeta2),
                         fabs(zeta1 * zeta2) + zeta2 * zeta2);

            /* See if root is closer to zero or to ONE */

            test = ONE + TWO * (zeta1 - zeta2) * (zeta1 + zeta2);
            if (test >= ZERO) {

                /* root is close to zero, compute directly */

                b = (zeta1 * zeta1 + zeta2 * zeta2 + ONE) * HALF;
                *c = zeta2 * zeta2;
                t = *c / (b + sqrt(fabs(b * b - *c)));
                sine = zeta1 / (ONE - t);
                cosine = -zeta2 / t;
                *sestpr = sqrt(t + FOUR * eps * eps * norma) * absest;
            } else {

                /* root is closer to ONE, shift by that amount */

                b = (zeta2 * zeta2 + zeta1 * zeta1 - ONE) * HALF;
                *c = zeta1 * zeta1;
                if (b >= ZERO) {
                    t = -*c / (b + sqrt(b * b + *c));
                } else {
                    t = b - sqrt(b * b + *c);
                }
                sine = -zeta1 / t;
                cosine = -zeta2 / (ONE + t);
                *sestpr = sqrt(ONE + t + FOUR * eps * eps * norma) * absest;
            }
            tmp = sqrt(sine * sine + cosine * cosine);
            *s = sine / tmp;
            *c = cosine / tmp;
            return;

        }
    }
    return;
}
