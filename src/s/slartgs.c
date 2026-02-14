/**
 * @file slartgs.c
 * @brief SLARTGS generates a plane rotation designed to introduce a bulge in
 *        implicit QR iteration for the bidiagonal SVD problem.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLARTGS generates a plane rotation designed to introduce a bulge in
 * Golub-Reinsch-style implicit QR iteration for the bidiagonal SVD
 * problem. X and Y are the top-row entries, and SIGMA is the shift.
 * The computed CS and SN define a plane rotation satisfying
 *
 *    [  CS  SN  ]  .  [ X^2 - SIGMA ]  =  [ R ],
 *    [ -SN  CS  ]     [    X * Y    ]     [ 0 ]
 *
 * with R nonnegative. If X^2 - SIGMA and X * Y are 0, then the
 * rotation is by PI/2.
 *
 * @param[in]  x      The (1,1) entry of an upper bidiagonal matrix.
 * @param[in]  y      The (1,2) entry of an upper bidiagonal matrix.
 * @param[in]  sigma  The shift.
 * @param[out] cs     The cosine of the rotation.
 * @param[out] sn     The sine of the rotation.
 */
void slartgs(const f32 x, const f32 y, const f32 sigma,
             f32* cs, f32* sn)
{
    f32 thresh = slamch("E");
    f32 z, w, r;

    if ((sigma == 0.0f && fabsf(x) < thresh) ||
        (fabsf(x) == sigma && y == 0.0f)) {
        z = 0.0f;
        w = 0.0f;
    } else if (sigma == 0.0f) {
        if (x >= 0.0f) {
            z = x;
            w = y;
        } else {
            z = -x;
            w = -y;
        }
    } else if (fabsf(x) < thresh) {
        z = -sigma * sigma;
        w = 0.0f;
    } else {
        f32 s;
        if (x >= 0.0f) {
            s = 1.0f;
        } else {
            s = -1.0f;
        }
        z = s * (fabsf(x) - sigma) * (s + sigma / x);
        w = s * y;
    }

    slartgp(w, z, sn, cs, &r);
}
