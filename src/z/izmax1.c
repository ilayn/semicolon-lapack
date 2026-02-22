/**
 * @file izmax1.c
 * @brief IZMAX1 finds the index of the first vector element of maximum
 *        absolute value.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_double.h"

/**
 * IZMAX1 finds the index of the first vector element of maximum absolute value.
 *
 * Based on IZAMAX from Level 1 BLAS.
 * The change is to use the 'genuine' absolute value.
 *
 * @param[in] n     The number of elements in the vector ZX.
 * @param[in] ZX    Complex array, dimension (n).
 *                   The vector ZX. The IZMAX1 function returns the index of its
 *                   first element of maximum absolute value.
 * @param[in] incx  The spacing between successive values of ZX. incx >= 1.
 */
INT izmax1(const INT n, const c128* restrict ZX, const INT incx)
{
    f64 dmax;
    INT i, ix, imax;

    if (n < 1 || incx <= 0)
        return 0;
    if (n == 1)
        return 0;
    if (incx == 1) {
        dmax = cabs(ZX[0]);
        imax = 0;
        for (i = 1; i < n; i++) {
            if (cabs(ZX[i]) > dmax) {
                imax = i;
                dmax = cabs(ZX[i]);
            }
        }
    } else {
        ix = 0;
        dmax = cabs(ZX[0]);
        imax = 0;
        ix = ix + incx;
        for (i = 1; i < n; i++) {
            if (cabs(ZX[ix]) > dmax) {
                imax = i;
                dmax = cabs(ZX[ix]);
            }
            ix = ix + incx;
        }
    }
    return imax;
}
