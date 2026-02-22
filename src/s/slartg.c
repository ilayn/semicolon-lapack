/**
 * @file slartg.c
 * @brief SLARTG generates a plane rotation with real cosine and real sine.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include "semicolon_lapack_single.h"

/**
 * SLARTG generates a plane rotation so that
 *
 *    [  C  S  ]  .  [ F ]  =  [ R ]
 *    [ -S  C  ]     [ G ]     [ 0 ]
 *
 * where C**2 + S**2 = 1.
 *
 * This is a more accurate version of the BLAS1 routine DROTG,
 * with the following other differences:
 *    F and G are unchanged on return.
 *    If G=0, then C=1 and S=0.
 *    If F=0 and (G .ne. 0), then C=0 and S=sign(1,G).
 *
 * @param[in]  f  The first component of vector to be rotated.
 * @param[in]  g  The second component of vector to be rotated.
 * @param[out] c  The cosine of the rotation.
 * @param[out] s  The sine of the rotation.
 * @param[out] r  The nonzero component of the rotated vector.
 */
void slartg(const f32 f, const f32 g, f32* c, f32* s, f32* r)
{
    f32 safmin = slamch("S");
    f32 safmax = 1.0f / safmin;
    f32 rtmin = sqrtf(safmin);
    f32 rtmax = sqrtf(safmax / 2.0f);

    f32 f1 = fabsf(f);
    f32 g1 = fabsf(g);

    if (g == 0.0f) {
        *c = 1.0f;
        *s = 0.0f;
        *r = f;
    } else if (f == 0.0f) {
        *c = 0.0f;
        *s = (g > 0.0f) ? 1.0f : -1.0f;
        *r = g1;
    } else if (f1 > rtmin && f1 < rtmax &&
               g1 > rtmin && g1 < rtmax) {
        /* Both f and g are in safe range */
        f32 d = sqrtf(f * f + g * g);
        *c = f1 / d;
        *r = (f > 0.0f) ? d : -d;
        *s = g / (*r);
    } else {
        /* Use scaled algorithm */
        f32 u = fminf(safmax, fmaxf(safmin, fmaxf(f1, g1)));
        f32 fs = f / u;
        f32 gs = g / u;
        f32 d = sqrtf(fs * fs + gs * gs);
        *c = fabsf(fs) / d;
        *r = (f > 0.0f) ? d : -d;
        *s = gs / (*r);
        *r = (*r) * u;
    }
}
