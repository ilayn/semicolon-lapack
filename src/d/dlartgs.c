/**
 * @file dlartgs.c
 * @brief DLARTGS generates a plane rotation designed to introduce a bulge in
 *        implicit QR iteration for the bidiagonal SVD problem.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLARTGS generates a plane rotation designed to introduce a bulge in
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
void dlartgs(const f64 x, const f64 y, const f64 sigma,
             f64* cs, f64* sn)
{
    f64 thresh = dlamch("E");
    f64 z, w, r;

    if ((sigma == 0.0 && fabs(x) < thresh) ||
        (fabs(x) == sigma && y == 0.0)) {
        z = 0.0;
        w = 0.0;
    } else if (sigma == 0.0) {
        if (x >= 0.0) {
            z = x;
            w = y;
        } else {
            z = -x;
            w = -y;
        }
    } else if (fabs(x) < thresh) {
        z = -sigma * sigma;
        w = 0.0;
    } else {
        f64 s;
        if (x >= 0.0) {
            s = 1.0;
        } else {
            s = -1.0;
        }
        z = s * (fabs(x) - sigma) * (s + sigma / x);
        w = s * y;
    }

    dlartgp(w, z, sn, cs, &r);
}
