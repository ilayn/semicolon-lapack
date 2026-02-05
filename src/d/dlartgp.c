/**
 * @file dlartgp.c
 * @brief DLARTGP generates a plane rotation so that the diagonal is nonnegative.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLARTGP generates a plane rotation so that
 *
 *    [  CS  SN  ]  .  [ F ]  =  [ R ]   where CS**2 + SN**2 = 1.
 *    [ -SN  CS  ]     [ G ]     [ 0 ]
 *
 * This is a slower, more accurate version of the Level 1 BLAS routine DROTG,
 * with the following other differences:
 *    F and G are unchanged on return.
 *    If G=0, then CS=(+/-)1 and SN=0.
 *    If F=0 and (G .ne. 0), then CS=0 and SN=(+/-)1.
 *
 * The sign is chosen so that R >= 0.
 *
 * @param[in]  f   The first component of vector to be rotated.
 * @param[in]  g   The second component of vector to be rotated.
 * @param[out] cs  The cosine of the rotation.
 * @param[out] sn  The sine of the rotation.
 * @param[out] r   The nonzero component of the rotated vector.
 */
void dlartgp(const double f, const double g, double* cs, double* sn, double* r)
{
    double safmin = dlamch("S");
    double eps = dlamch("E");
    double safmn2 = pow(dlamch("B"), (int)(log(safmin / eps) / log(dlamch("B")) / 2.0));
    double safmx2 = 1.0 / safmn2;

    if (g == 0.0) {
        *cs = copysign(1.0, f);
        *sn = 0.0;
        *r = fabs(f);
    } else if (f == 0.0) {
        *cs = 0.0;
        *sn = copysign(1.0, g);
        *r = fabs(g);
    } else {
        double f1 = f;
        double g1 = g;
        double scale = fmax(fabs(f1), fabs(g1));

        if (scale >= safmx2) {
            int count = 0;
            do {
                count++;
                f1 = f1 * safmn2;
                g1 = g1 * safmn2;
                scale = fmax(fabs(f1), fabs(g1));
            } while (scale >= safmx2 && count < 20);

            *r = sqrt(f1 * f1 + g1 * g1);
            *cs = f1 / (*r);
            *sn = g1 / (*r);
            for (int i = 0; i < count; i++) {
                *r = (*r) * safmx2;
            }
        } else if (scale <= safmn2) {
            int count = 0;
            do {
                count++;
                f1 = f1 * safmx2;
                g1 = g1 * safmx2;
                scale = fmax(fabs(f1), fabs(g1));
            } while (scale <= safmn2);

            *r = sqrt(f1 * f1 + g1 * g1);
            *cs = f1 / (*r);
            *sn = g1 / (*r);
            for (int i = 0; i < count; i++) {
                *r = (*r) * safmn2;
            }
        } else {
            *r = sqrt(f1 * f1 + g1 * g1);
            *cs = f1 / (*r);
            *sn = g1 / (*r);
        }

        if (*r < 0.0) {
            *cs = -(*cs);
            *sn = -(*sn);
            *r = -(*r);
        }
    }
}
