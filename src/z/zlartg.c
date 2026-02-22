/**
 * @file zlartg.c
 * @brief ZLARTG generates a plane rotation with real cosine and complex sine.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/** @cond */
static inline f64 abssq(c128 t)
{
    return creal(t) * creal(t) + cimag(t) * cimag(t);
}
/** @endcond */

/**
 * ZLARTG generates a plane rotation so that
 *
 *    [  C         S  ] . [ F ]  =  [ R ]
 *    [ -conjg(S)  C  ]   [ G ]     [ 0 ]
 *
 * where C is real and C**2 + |S|**2 = 1.
 *
 * The mathematical formulas used for C and S are
 *
 *    sgn(x) = {  x / |x|,   x != 0
 *             {  1,         x  = 0
 *
 *    R = sgn(F) * sqrt(|F|**2 + |G|**2)
 *
 *    C = |F| / sqrt(|F|**2 + |G|**2)
 *
 *    S = sgn(F) * conjg(G) / sqrt(|F|**2 + |G|**2)
 *
 * Special conditions:
 *    If G=0, then C=1 and S=0.
 *    If F=0, then C=0 and S is chosen so that R is real.
 *
 * When F and G are real, the formulas simplify to C = F/R and
 * S = G/R, and the returned values of C, S, and R should be
 * identical to those returned by DLARTG.
 *
 * @param[in]  f  The first component of vector to be rotated.
 * @param[in]  g  The second component of vector to be rotated.
 * @param[out] c  The cosine of the rotation.
 * @param[out] s  The sine of the rotation.
 * @param[out] r  The nonzero component of the rotated vector.
 */
void zlartg(const c128 f, const c128 g,
            f64* c, c128* s, c128* r)
{
    f64 safmin = dlamch("S");
    f64 safmax = 1.0 / safmin;
    f64 rtmin = sqrt(safmin);

    f64 d, f1, f2, g1, g2, h2, u, v, w, rtmax;
    c128 fs, gs;

    if (g == CMPLX(0.0, 0.0)) {
        *c = 1.0;
        *s = CMPLX(0.0, 0.0);
        *r = f;
    } else if (f == CMPLX(0.0, 0.0)) {
        *c = 0.0;
        if (creal(g) == 0.0) {
            *r = fabs(cimag(g));
            *s = conj(g) / *r;
        } else if (cimag(g) == 0.0) {
            *r = fabs(creal(g));
            *s = conj(g) / *r;
        } else {
            g1 = fmax(fabs(creal(g)), fabs(cimag(g)));
            rtmax = sqrt(safmax / 2.0);
            if (g1 > rtmin && g1 < rtmax) {

                /* Use unscaled algorithm */

                g2 = abssq(g);
                d = sqrt(g2);
                *s = conj(g) / d;
                *r = d;
            } else {

                /* Use scaled algorithm */

                u = fmin(safmax, fmax(safmin, g1));
                gs = g / u;
                g2 = abssq(gs);
                d = sqrt(g2);
                *s = conj(gs) / d;
                *r = d * u;
            }
        }
    } else {
        f1 = fmax(fabs(creal(f)), fabs(cimag(f)));
        g1 = fmax(fabs(creal(g)), fabs(cimag(g)));
        rtmax = sqrt(safmax / 4.0);
        if (f1 > rtmin && f1 < rtmax &&
            g1 > rtmin && g1 < rtmax) {

            /* Use unscaled algorithm */

            f2 = abssq(f);
            g2 = abssq(g);
            h2 = f2 + g2;
            /* safmin <= f2 <= h2 <= safmax */
            if (f2 >= h2 * safmin) {
                /* safmin <= f2/h2 <= 1, and h2/f2 is finite */
                *c = sqrt(f2 / h2);
                *r = f / *c;
                rtmax = rtmax * 2.0;
                if (f2 > rtmin && h2 < rtmax) {
                    /* safmin <= sqrt( f2*h2 ) <= safmax */
                    *s = conj(g) * (f / sqrt(f2 * h2));
                } else {
                    *s = conj(g) * (*r / h2);
                }
            } else {
                d = sqrt(f2 * h2);
                *c = f2 / d;
                if (*c >= safmin) {
                    *r = f / *c;
                } else {
                    *r = f * (h2 / d);
                }
                *s = conj(g) * (f / d);
            }
        } else {

            /* Use scaled algorithm */

            u = fmin(safmax, fmax(safmin, fmax(f1, g1)));
            gs = g / u;
            g2 = abssq(gs);
            if (f1 / u < rtmin) {

                v = fmin(safmax, fmax(safmin, f1));
                w = v / u;
                fs = f / v;
                f2 = abssq(fs);
                h2 = f2 * w * w + g2;
            } else {

                w = 1.0;
                fs = f / u;
                f2 = abssq(fs);
                h2 = f2 + g2;
            }
            /* safmin <= f2 <= h2 <= safmax */
            if (f2 >= h2 * safmin) {
                /* safmin <= f2/h2 <= 1, and h2/f2 is finite */
                *c = sqrt(f2 / h2);
                *r = fs / *c;
                rtmax = rtmax * 2.0;
                if (f2 > rtmin && h2 < rtmax) {
                    /* safmin <= sqrt( f2*h2 ) <= safmax */
                    *s = conj(gs) * (fs / sqrt(f2 * h2));
                } else {
                    *s = conj(gs) * (*r / h2);
                }
            } else {
                d = sqrt(f2 * h2);
                *c = f2 / d;
                if (*c >= safmin) {
                    *r = fs / *c;
                } else {
                    *r = fs * (h2 / d);
                }
                *s = conj(gs) * (fs / d);
            }
            /* Rescale c and r */
            *c = *c * w;
            *r = *r * u;
        }
    }
}
