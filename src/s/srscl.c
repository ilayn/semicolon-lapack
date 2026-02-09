/**
 * @file srscl.c
 * @brief Multiplies a vector by the reciprocal of a real scalar.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SRSCL multiplies an n-element real vector x by the real scalar 1/a.
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
void srscl(
    const int n,
    const float sa,
    float * const restrict sx,
    const int incx)
{
    const float ONE = 1.0f;
    const float ZERO = 0.0f;

    int done;
    float bignum, cden, cden1, cnum, cnum1, mul, smlnum;

    // Quick return if possible
    if (n <= 0) {
        return;
    }

    // Get machine parameters
    smlnum = FLT_MIN;
    bignum = ONE / smlnum;

    // Initialize the denominator to SA and the numerator to 1
    cden = sa;
    cnum = ONE;

    do {
        cden1 = cden * smlnum;
        cnum1 = cnum / bignum;

        if (fabsf(cden1) > fabsf(cnum) && cnum != ZERO) {
            // Pre-multiply X by SMLNUM if CDEN is large compared to CNUM
            mul = smlnum;
            done = 0;
            cden = cden1;
        } else if (fabsf(cnum1) > fabsf(cden)) {
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
        cblas_sscal(n, mul, sx, incx);

    } while (!done);
}
