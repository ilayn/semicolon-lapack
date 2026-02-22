/**
 * @file zlacgv.c
 * @brief ZLACGV conjugates a complex vector.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLACGV conjugates a complex vector of length N.
 *
 * @param[in]     n     The length of the vector X. n >= 0.
 * @param[in,out] X     Double complex array, dimension (1+(n-1)*abs(incx)).
 *                      On entry, the vector of length n to be conjugated.
 *                      On exit, X is overwritten with conjg(X).
 * @param[in]     incx  The spacing between successive elements of X.
 */
void zlacgv(
    const INT n,
    c128* restrict X,
    const INT incx)
{
    INT i, ioff;

    if (incx == 1) {
        for (i = 0; i < n; i++) {
            X[i] = conj(X[i]);
        }
    } else {
        ioff = 0;
        if (incx < 0) {
            ioff = -(n - 1) * incx;
        }
        for (i = 0; i < n; i++) {
            X[ioff] = conj(X[ioff]);
            ioff += incx;
        }
    }
}
