/**
 * @file scsum1.c
 * @brief SCSUM1 takes the sum of the absolute values of a complex
 *        vector and returns a f32 precision result.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

/**
 * SCSUM1 takes the sum of the absolute values of a complex
 * vector and returns a f32 precision result.
 *
 * Based on DZASUM from the Level 1 BLAS.
 * The change is to use the 'genuine' absolute value.
 *
 * @param[in] n     The number of elements in the vector CX.
 * @param[in] CX    Complex array, dimension (n).
 *                   The vector whose elements will be summed.
 * @param[in] incx  The spacing between successive values of CX. incx > 0.
 */
f32 scsum1(const INT n, const c64* restrict CX, const INT incx)
{
    f32 stemp;
    INT i, nincx;

    stemp = 0.0f;
    if (n <= 0)
        return 0.0f;
    if (incx == 1) {
        for (i = 0; i < n; i++) {
            stemp = stemp + cabsf(CX[i]);
        }
    } else {
        nincx = n * incx;
        for (i = 0; i < nincx; i += incx) {
            stemp = stemp + cabsf(CX[i]);
        }
    }
    return stemp;
}
