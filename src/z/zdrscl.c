/**
 * @file zdrscl.c
 * @brief ZDRSCL multiplies a vector by the reciprocal of a real scalar.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZDRSCL multiplies an n-element complex vector x by the real scalar 1/a.
 * This is done without overflow or underflow as long as the final
 * result x/a does not overflow or underflow.
 *
 * @param[in]     n     The number of components of the vector x.
 * @param[in]     sa    The scalar a which is used to divide each component of x.
 *                      sa must be >= 0, or the subroutine will divide by zero.
 * @param[in,out] sx    Complex*16 array, dimension (1+(n-1)*abs(incx)).
 *                      The n-element vector x.
 * @param[in]     incx  The increment between successive values of the vector sx.
 *                      incx > 0.
 */
void zdrscl(
    const int n,
    const f64 sa,
    c128* const restrict sx,
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

    if (sa > DBL_MAX || sa < -DBL_MAX) {
        cblas_zdscal(n, sa, sx, incx);
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
        cblas_zdscal(n, mul, sx, incx);

    } while (!done);
}
