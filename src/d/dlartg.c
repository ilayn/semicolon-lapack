/**
 * @file dlartg.c
 * @brief DLARTG generates a plane rotation with real cosine and real sine.
 */

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
void dlartg(const double f, const double g, double* c, double* s, double* r)
{
    double safmin = dlamch("S");
    double safmax = 1.0 / safmin;
    double rtmin = sqrt(safmin);
    double rtmax = sqrt(safmax / 2.0);

    double f1 = fabs(f);
    double g1 = fabs(g);

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
        double d = sqrt(f * f + g * g);
        *c = f1 / d;
        *r = (f > 0.0) ? d : -d;
        *s = g / (*r);
    } else {
        /* Use scaled algorithm */
        double u = fmin(safmax, fmax(safmin, fmax(f1, g1)));
        double fs = f / u;
        double gs = g / u;
        double d = sqrt(fs * fs + gs * gs);
        *c = fabs(fs) / d;
        *r = (f > 0.0) ? d : -d;
        *s = gs / (*r);
        *r = (*r) * u;
    }
}
