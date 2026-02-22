/**
 * @file dlas2.c
 * @brief DLAS2 computes singular values of a 2-by-2 triangular matrix.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * DLAS2 computes the singular values of the 2-by-2 matrix
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
void dlas2(const f64 f, const f64 g, const f64 h,
           f64* ssmin, f64* ssmax)
{
    f64 fa, ga, ha, fhmn, fhmx;
    f64 as, at, au, c;

    fa = fabs(f);
    ga = fabs(g);
    ha = fabs(h);
    fhmn = (fa < ha) ? fa : ha;
    fhmx = (fa > ha) ? fa : ha;

    if (fhmn == 0.0) {
        *ssmin = 0.0;
        if (fhmx == 0.0) {
            *ssmax = ga;
        } else {
            f64 mn = (fhmx < ga) ? fhmx : ga;
            f64 mx = (fhmx > ga) ? fhmx : ga;
            *ssmax = mx * sqrt(1.0 + (mn / mx) * (mn / mx));
        }
    } else {
        if (ga < fhmx) {
            as = 1.0 + fhmn / fhmx;
            at = (fhmx - fhmn) / fhmx;
            au = (ga / fhmx) * (ga / fhmx);
            c = 2.0 / (sqrt(as * as + au) + sqrt(at * at + au));
            *ssmin = fhmn * c;
            *ssmax = fhmx / c;
        } else {
            au = fhmx / ga;
            if (au == 0.0) {
                /*
                 * Avoid possible harmful underflow if exponent range
                 * asymmetric (true SSMIN may not underflow even if
                 * AU underflows)
                 */
                *ssmin = (fhmn * fhmx) / ga;
                *ssmax = ga;
            } else {
                as = 1.0 + fhmn / fhmx;
                at = (fhmx - fhmn) / fhmx;
                c = 1.0 / (sqrt(1.0 + (as * au) * (as * au)) +
                           sqrt(1.0 + (at * au) * (at * au)));
                *ssmin = (fhmn * c) * au;
                *ssmin = *ssmin + *ssmin;
                *ssmax = ga / (c + c);
            }
        }
    }
}
