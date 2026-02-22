/**
 * @file dlartg.c
 * @brief DLARTG generates a plane rotation with real cosine and real sine.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DLARTG generates a plane rotation so that
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
void dlartg(const f64 f, const f64 g, f64* c, f64* s, f64* r)
{
    f64 safmin = dlamch("S");
    f64 safmax = 1.0 / safmin;
    f64 rtmin = sqrt(safmin);
    f64 rtmax = sqrt(safmax / 2.0);

    f64 f1 = fabs(f);
    f64 g1 = fabs(g);

    if (g == 0.0) {
        *c = 1.0;
        *s = 0.0;
        *r = f;
    } else if (f == 0.0) {
        *c = 0.0;
        *s = (g > 0.0) ? 1.0 : -1.0;
        *r = g1;
    } else if (f1 > rtmin && f1 < rtmax &&
               g1 > rtmin && g1 < rtmax) {
        /* Both f and g are in safe range */
        f64 d = sqrt(f * f + g * g);
        *c = f1 / d;
        *r = (f > 0.0) ? d : -d;
        *s = g / (*r);
    } else {
        /* Use scaled algorithm */
        f64 u = fmin(safmax, fmax(safmin, fmax(f1, g1)));
        f64 fs = f / u;
        f64 gs = g / u;
        f64 d = sqrt(fs * fs + gs * gs);
        *c = fabs(fs) / d;
        *r = (f > 0.0) ? d : -d;
        *s = gs / (*r);
        *r = (*r) * u;
    }
}
