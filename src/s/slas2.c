/**
 * @file slas2.c
 * @brief SLAS2 computes singular values of a 2-by-2 triangular matrix.
 */

#include "semicolon_lapack_single.h"
#include <math.h>

/**
 * SLAS2 computes the singular values of the 2-by-2 matrix
 *    [  F   G  ]
 *    [  0   H  ].
 * On return, SSMIN is the smaller singular value and SSMAX is the
 * larger singular value.
 *
 * @param[in]  f      The (1,1) element of the 2-by-2 matrix.
 * @param[in]  g      The (1,2) element of the 2-by-2 matrix.
 * @param[in]  h      The (2,2) element of the 2-by-2 matrix.
 * @param[out] ssmin  The smaller singular value.
 * @param[out] ssmax  The larger singular value.
 */
void slas2(const f32 f, const f32 g, const f32 h,
           f32* ssmin, f32* ssmax)
{
    f32 fa, ga, ha, fhmn, fhmx;
    f32 as, at, au, c;

    fa = fabsf(f);
    ga = fabsf(g);
    ha = fabsf(h);
    fhmn = (fa < ha) ? fa : ha;
    fhmx = (fa > ha) ? fa : ha;

    if (fhmn == 0.0f) {
        *ssmin = 0.0f;
        if (fhmx == 0.0f) {
            *ssmax = ga;
        } else {
            f32 mn = (fhmx < ga) ? fhmx : ga;
            f32 mx = (fhmx > ga) ? fhmx : ga;
            *ssmax = mx * sqrtf(1.0f + (mn / mx) * (mn / mx));
        }
    } else {
        if (ga < fhmx) {
            as = 1.0f + fhmn / fhmx;
            at = (fhmx - fhmn) / fhmx;
            au = (ga / fhmx) * (ga / fhmx);
            c = 2.0f / (sqrtf(as * as + au) + sqrtf(at * at + au));
            *ssmin = fhmn * c;
            *ssmax = fhmx / c;
        } else {
            au = fhmx / ga;
            if (au == 0.0f) {
                /*
                 * Avoid possible harmful underflow if exponent range
                 * asymmetric (true SSMIN may not underflow even if
                 * AU underflows)
                 */
                *ssmin = (fhmn * fhmx) / ga;
                *ssmax = ga;
            } else {
                as = 1.0f + fhmn / fhmx;
                at = (fhmx - fhmn) / fhmx;
                c = 1.0f / (sqrtf(1.0f + (as * au) * (as * au)) +
                           sqrtf(1.0f + (at * au) * (at * au)));
                *ssmin = (fhmn * c) * au;
                *ssmin = *ssmin + *ssmin;
                *ssmax = ga / (c + c);
            }
        }
    }
}
