/**
 * @file drscl.c
 * @brief Multiplies a vector by the reciprocal of a real scalar.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DRSCL multiplies an n-element real vector x by the real scalar 1/a.
 * This is done without overflow or underflow as long as the final
 * result x/a does not overflow or underflow.
 *
 * @param[in]     n     The number of components of the vector x.
 * @param[in]     sa    The scalar a which is used to divide each component of x.
 *                      sa must be >= 0, or the subroutine will divide by zero.
 * @param[in,out] sx    The n-element vector x. Array of dimension (1+(n-1)*abs(incx)).
 * @param[in]     incx  The increment between successive values of the vector sx.
 *                      incx > 0.
 */
void drscl(
    const int n,
    const f64 sa,
    f64 * const restrict sx,
    const int incx)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int done;
    f64 bignum, cden, cden1, cnum, cnum1, mul, smlnum;

    // Quick return if possible
    if (n <= 0) {
        return;
    }

    // Get machine parameters
    smlnum = DBL_MIN;
    bignum = ONE / smlnum;

    // Initialize the denominator to SA and the numerator to 1
    cden = sa;
    cnum = ONE;

    do {
        cden1 = cden * smlnum;
        cnum1 = cnum / bignum;

        if (fabs(cden1) > fabs(cnum) && cnum != ZERO) {
            // Pre-multiply X by SMLNUM if CDEN is large compared to CNUM
            mul = smlnum;
            done = 0;
            cden = cden1;
        } else if (fabs(cnum1) > fabs(cden)) {
            // Pre-multiply X by BIGNUM if CDEN is small compared to CNUM
            mul = bignum;
            done = 0;
            cnum = cnum1;
        } else {
            // Multiply X by CNUM / CDEN and return
            mul = cnum / cden;
            done = 1;
        }

        // Scale the vector X by MUL
        cblas_dscal(n, mul, sx, incx);

    } while (!done);
}
