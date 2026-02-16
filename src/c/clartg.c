/**
 * @file clartg.c
 * @brief CLARTG generates a plane rotation with real cosine and complex sine.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/** @cond */
static inline f32 abssq(c64 t)
{
    return crealf(t) * crealf(t) + cimagf(t) * cimagf(t);
}
/** @endcond */

/**
 * CLARTG generates a plane rotation so that
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
 * identical to those returned by SLARTG.
 *
 * @param[in]  f  The first component of vector to be rotated.
 * @param[in]  g  The second component of vector to be rotated.
 * @param[out] c  The cosine of the rotation.
 * @param[out] s  The sine of the rotation.
 * @param[out] r  The nonzero component of the rotated vector.
 */
void clartg(const c64 f, const c64 g,
            f32* c, c64* s, c64* r)
{
    f32 safmin = slamch("S");
    f32 safmax = 1.0f / safmin;
    f32 rtmin = sqrtf(safmin);

    f32 d, f1, f2, g1, g2, h2, u, v, w, rtmax;
    c64 fs, gs;

    if (g == CMPLXF(0.0f, 0.0f)) {
        *c = 1.0f;
        *s = CMPLXF(0.0f, 0.0f);
        *r = f;
    } else if (f == CMPLXF(0.0f, 0.0f)) {
        *c = 0.0f;
        if (crealf(g) == 0.0f) {
            *r = fabsf(cimagf(g));
            *s = conjf(g) / *r;
        } else if (cimagf(g) == 0.0f) {
            *r = fabsf(crealf(g));
            *s = conjf(g) / *r;
        } else {
            g1 = fmaxf(fabsf(crealf(g)), fabsf(cimagf(g)));
            rtmax = sqrtf(safmax / 2.0f);
            if (g1 > rtmin && g1 < rtmax) {

                /* Use unscaled algorithm */

                g2 = abssq(g);
                d = sqrtf(g2);
                *s = conjf(g) / d;
                *r = d;
            } else {

                /* Use scaled algorithm */

                u = fminf(safmax, fmaxf(safmin, g1));
                gs = g / u;
                g2 = abssq(gs);
                d = sqrtf(g2);
                *s = conjf(gs) / d;
                *r = d * u;
            }
        }
    } else {
        f1 = fmaxf(fabsf(crealf(f)), fabsf(cimagf(f)));
        g1 = fmaxf(fabsf(crealf(g)), fabsf(cimagf(g)));
        rtmax = sqrtf(safmax / 4.0f);
        if (f1 > rtmin && f1 < rtmax &&
            g1 > rtmin && g1 < rtmax) {

            /* Use unscaled algorithm */

            f2 = abssq(f);
            g2 = abssq(g);
            h2 = f2 + g2;
            /* safmin <= f2 <= h2 <= safmax */
            if (f2 >= h2 * safmin) {
                /* safmin <= f2/h2 <= 1, and h2/f2 is finite */
                *c = sqrtf(f2 / h2);
                *r = f / *c;
                rtmax = rtmax * 2.0f;
                if (f2 > rtmin && h2 < rtmax) {
                    /* safmin <= sqrt( f2*h2 ) <= safmax */
                    *s = conjf(g) * (f / sqrtf(f2 * h2));
                } else {
                    *s = conjf(g) * (*r / h2);
                }
            } else {
                d = sqrtf(f2 * h2);
                *c = f2 / d;
                if (*c >= safmin) {
                    *r = f / *c;
                } else {
                    *r = f * (h2 / d);
                }
                *s = conjf(g) * (f / d);
            }
        } else {

            /* Use scaled algorithm */

            u = fminf(safmax, fmaxf(safmin, fmaxf(f1, g1)));
            gs = g / u;
            g2 = abssq(gs);
            if (f1 / u < rtmin) {

                v = fminf(safmax, fmaxf(safmin, f1));
                w = v / u;
                fs = f / v;
                f2 = abssq(fs);
                h2 = f2 * w * w + g2;
            } else {

                w = 1.0f;
                fs = f / u;
                f2 = abssq(fs);
                h2 = f2 + g2;
            }
            /* safmin <= f2 <= h2 <= safmax */
            if (f2 >= h2 * safmin) {
                /* safmin <= f2/h2 <= 1, and h2/f2 is finite */
                *c = sqrtf(f2 / h2);
                *r = fs / *c;
                rtmax = rtmax * 2.0f;
                if (f2 > rtmin && h2 < rtmax) {
                    /* safmin <= sqrt( f2*h2 ) <= safmax */
                    *s = conjf(gs) * (fs / sqrtf(f2 * h2));
                } else {
                    *s = conjf(gs) * (*r / h2);
                }
            } else {
                d = sqrtf(f2 * h2);
                *c = f2 / d;
                if (*c >= safmin) {
                    *r = fs / *c;
                } else {
                    *r = fs * (h2 / d);
                }
                *s = conjf(gs) * (fs / d);
            }
            /* Rescale c and r */
            *c = *c * w;
            *r = *r * u;
        }
    }
}
