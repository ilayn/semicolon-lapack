/**
 * @file zlaic1.c
 * @brief ZLAIC1 applies one step of incremental condition estimation.
 */

#include <math.h>
#include <complex.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLAIC1 applies one step of incremental condition estimation in
 * its simplest version:
 *
 * Let x, twonorm(x) = 1, be an approximate singular vector of an j-by-j
 * lower triangular matrix L, such that
 *          twonorm(L*x) = sest
 * Then ZLAIC1 computes sestpr, s, c such that
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
 * @param[in]     gamma  The diagonal element gamma.
 * @param[out]    sestpr Estimated singular value of (j+1) by (j+1) matrix Lhat.
 * @param[out]    s      Sine needed in forming xhat.
 * @param[out]    c      Cosine needed in forming xhat.
 */
void zlaic1(
    const int job,
    const int j,
    const c128* const restrict x,
    const f64 sest,
    const c128* const restrict w,
    const c128 gamma_,
    f64* sestpr,
    c128* s,
    c128* c)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 HALF = 0.5;
    const f64 FOUR = 4.0;

    f64 absalp, absest, absgam, b, eps,
           norma, s1, s2, scl, t, test, tmp, zeta1, zeta2;
    c128 alpha, cosine, sine;

    eps = DBL_EPSILON;
    cblas_zdotc_sub(j, x, 1, w, 1, &alpha);

    absalp = cabs(alpha);
    absgam = cabs(gamma_);
    absest = fabs(sest);

    if (job == 1) {

        /* Estimating largest singular value */

        /* special cases */

        if (sest == ZERO) {
            s1 = fmax(absgam, absalp);
            if (s1 == ZERO) {
                *s = CMPLX(ZERO, 0.0);
                *c = CMPLX(ONE, 0.0);
                *sestpr = ZERO;
            } else {
                *s = alpha / s1;
                *c = gamma_ / s1;
                tmp = creal(csqrt((*s) * conj(*s) + (*c) * conj(*c)));
                *s = *s / tmp;
                *c = *c / tmp;
                *sestpr = s1 * tmp;
            }
            return;
        } else if (absgam <= eps * absest) {
            *s = CMPLX(ONE, 0.0);
            *c = CMPLX(ZERO, 0.0);
            tmp = fmax(absest, absalp);
            s1 = absest / tmp;
            s2 = absalp / tmp;
            *sestpr = tmp * sqrt(s1 * s1 + s2 * s2);
            return;
        } else if (absalp <= eps * absest) {
            s1 = absgam;
            s2 = absest;
            if (s1 <= s2) {
                *s = CMPLX(ONE, 0.0);
                *c = CMPLX(ZERO, 0.0);
                *sestpr = s2;
            } else {
                *s = CMPLX(ZERO, 0.0);
                *c = CMPLX(ONE, 0.0);
                *sestpr = s1;
            }
            return;
        } else if (absest <= eps * absalp || absest <= eps * absgam) {
            s1 = absgam;
            s2 = absalp;
            if (s1 <= s2) {
                tmp = s1 / s2;
                scl = sqrt(ONE + tmp * tmp);
                *sestpr = s2 * scl;
                *s = (alpha / s2) / scl;
                *c = (gamma_ / s2) / scl;
            } else {
                tmp = s2 / s1;
                scl = sqrt(ONE + tmp * tmp);
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
                t = t / (b + sqrt(b * b + t));
            } else {
                t = sqrt(b * b + t) - b;
            }

            sine = -(alpha / absest) / t;
            cosine = -(gamma_ / absest) / (ONE + t);
            tmp = creal(csqrt(sine * conj(sine)
                + cosine * conj(cosine)));

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
                sine = CMPLX(ONE, 0.0);
                cosine = CMPLX(ZERO, 0.0);
            } else {
                sine = -conj(gamma_);
                cosine = conj(alpha);
            }
            s1 = fmax(cabs(sine), cabs(cosine));
            *s = sine / s1;
            *c = cosine / s1;
            tmp = creal(csqrt((*s) * conj(*s) + (*c) * conj(*c)));
            *s = *s / tmp;
            *c = *c / tmp;
            return;
        } else if (absgam <= eps * absest) {
            *s = CMPLX(ZERO, 0.0);
            *c = CMPLX(ONE, 0.0);
            *sestpr = absgam;
            return;
        } else if (absalp <= eps * absest) {
            s1 = absgam;
            s2 = absest;
            if (s1 <= s2) {
                *s = CMPLX(ZERO, 0.0);
                *c = CMPLX(ONE, 0.0);
                *sestpr = s1;
            } else {
                *s = CMPLX(ONE, 0.0);
                *c = CMPLX(ZERO, 0.0);
                *sestpr = s2;
            }
            return;
        } else if (absest <= eps * absalp || absest <= eps * absgam) {
            s1 = absgam;
            s2 = absalp;
            if (s1 <= s2) {
                tmp = s1 / s2;
                scl = sqrt(ONE + tmp * tmp);
                *sestpr = absest * (tmp / scl);
                *s = -(conj(gamma_) / s2) / scl;
                *c = (conj(alpha) / s2) / scl;
            } else {
                tmp = s2 / s1;
                scl = sqrt(ONE + tmp * tmp);
                *sestpr = absest / scl;
                *s = -(conj(gamma_) / s1) / scl;
                *c = (conj(alpha) / s1) / scl;
            }
            return;
        } else {

            /* normal case */

            zeta1 = absalp / absest;
            zeta2 = absgam / absest;

            norma = fmax(ONE + zeta1 * zeta1 + zeta1 * zeta2,
                         zeta1 * zeta2 + zeta2 * zeta2);

            /* See if root is closer to zero or to ONE */

            test = ONE + TWO * (zeta1 - zeta2) * (zeta1 + zeta2);
            if (test >= ZERO) {

                /* root is close to zero, compute directly */

                b = (zeta1 * zeta1 + zeta2 * zeta2 + ONE) * HALF;
                t = zeta2 * zeta2;
                t = t / (b + sqrt(fabs(b * b - t)));
                sine = (alpha / absest) / (ONE - t);
                cosine = -(gamma_ / absest) / t;
                *sestpr = sqrt(t + FOUR * eps * eps * norma) * absest;
            } else {

                /* root is closer to ONE, shift by that amount */

                b = (zeta2 * zeta2 + zeta1 * zeta1 - ONE) * HALF;
                t = zeta1 * zeta1;
                if (b >= ZERO) {
                    t = -t / (b + sqrt(b * b + t));
                } else {
                    t = b - sqrt(b * b + t);
                }
                sine = -(alpha / absest) / t;
                cosine = -(gamma_ / absest) / (ONE + t);
                *sestpr = sqrt(ONE + t + FOUR * eps * eps * norma) * absest;
            }
            tmp = creal(csqrt(sine * conj(sine)
                + cosine * conj(cosine)));
            *s = sine / tmp;
            *c = cosine / tmp;
            return;

        }
    }
    return;
}
