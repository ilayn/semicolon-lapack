/**
 * @file dzsum1.c
 * @brief DZSUM1 takes the sum of the absolute values of a complex
 *        vector and returns a f64 precision result.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"

/**
 * DZSUM1 takes the sum of the absolute values of a complex
 * vector and returns a f64 precision result.
 *
 * Based on DZASUM from the Level 1 BLAS.
 * The change is to use the 'genuine' absolute value.
 *
 * @param[in] n     The number of elements in the vector CX.
 * @param[in] CX    Complex array, dimension (n).
 *                   The vector whose elements will be summed.
 * @param[in] incx  The spacing between successive values of CX. incx > 0.
 */
f64 dzsum1(const int n, const c128* restrict CX, const int incx)
{
    f64 stemp;
    int i, nincx;

    stemp = 0.0;
    if (n <= 0)
        return 0.0;
    if (incx == 1) {
        for (i = 0; i < n; i++) {
            stemp = stemp + cabs(CX[i]);
        }
    } else {
        nincx = n * incx;
        for (i = 0; i < nincx; i += incx) {
            stemp = stemp + cabs(CX[i]);
        }
    }
    return stemp;
}
