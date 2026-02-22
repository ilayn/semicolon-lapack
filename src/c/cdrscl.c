/**
 * @file cdrscl.c
 * @brief CDRSCL multiplies a vector by the reciprocal of a real scalar.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CDRSCL multiplies an n-element complex vector x by the real scalar 1/a.
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
void cdrscl(
    const INT n,
    const f32 sa,
    c64* restrict sx,
    const INT incx)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT done;
    f32 bignum, cden, cden1, cnum, cnum1, mul, smlnum;

    // Quick return if possible
    if (n <= 0) {
        return;
    }

    if (sa > FLT_MAX || sa < -FLT_MAX) {
        cblas_csscal(n, sa, sx, incx);
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
        cblas_csscal(n, mul, sx, incx);

    } while (!done);
}
