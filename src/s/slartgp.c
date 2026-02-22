/**
 * @file slartgp.c
 * @brief SLARTGP generates a plane rotation so that the diagonal is nonnegative.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLARTGP generates a plane rotation so that
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
void slartgp(const f32 f, const f32 g, f32* cs, f32* sn, f32* r)
{
    f32 safmin = slamch("S");
    f32 eps = slamch("E");
    f32 safmn2 = powf(slamch("B"), (INT)(logf(safmin / eps) / logf(slamch("B")) / 2.0f));
    f32 safmx2 = 1.0f / safmn2;

    if (g == 0.0f) {
        *cs = copysignf(1.0f, f);
        *sn = 0.0f;
        *r = fabsf(f);
    } else if (f == 0.0f) {
        *cs = 0.0f;
        *sn = copysignf(1.0f, g);
        *r = fabsf(g);
    } else {
        f32 f1 = f;
        f32 g1 = g;
        f32 scale = fmaxf(fabsf(f1), fabsf(g1));

        if (scale >= safmx2) {
            INT count = 0;
            do {
                count++;
                f1 = f1 * safmn2;
                g1 = g1 * safmn2;
                scale = fmaxf(fabsf(f1), fabsf(g1));
            } while (scale >= safmx2 && count < 20);

            *r = sqrtf(f1 * f1 + g1 * g1);
            *cs = f1 / (*r);
            *sn = g1 / (*r);
            for (INT i = 0; i < count; i++) {
                *r = (*r) * safmx2;
            }
        } else if (scale <= safmn2) {
            INT count = 0;
            do {
                count++;
                f1 = f1 * safmx2;
                g1 = g1 * safmx2;
                scale = fmaxf(fabsf(f1), fabsf(g1));
            } while (scale <= safmn2);

            *r = sqrtf(f1 * f1 + g1 * g1);
            *cs = f1 / (*r);
            *sn = g1 / (*r);
            for (INT i = 0; i < count; i++) {
                *r = (*r) * safmn2;
            }
        } else {
            *r = sqrtf(f1 * f1 + g1 * g1);
            *cs = f1 / (*r);
            *sn = g1 / (*r);
        }

        if (*r < 0.0f) {
            *cs = -(*cs);
            *sn = -(*sn);
            *r = -(*r);
        }
    }
}
