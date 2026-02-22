/**
 * @file clacgv.c
 * @brief CLACGV conjugates a complex vector.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLACGV conjugates a complex vector of length N.
 *
 * @param[in]     n     The length of the vector X. n >= 0.
 * @param[in,out] X     Single complex array, dimension (1+(n-1)*abs(incx)).
 *                      On entry, the vector of length n to be conjugated.
 *                      On exit, X is overwritten with conjg(X).
 * @param[in]     incx  The spacing between successive elements of X.
 */
void clacgv(
    const INT n,
    c64* restrict X,
    const INT incx)
{
    INT i, ioff;

    if (incx == 1) {
        for (i = 0; i < n; i++) {
            X[i] = conjf(X[i]);
        }
    } else {
        ioff = 0;
        if (incx < 0) {
            ioff = -(n - 1) * incx;
        }
        for (i = 0; i < n; i++) {
            X[ioff] = conjf(X[ioff]);
            ioff += incx;
        }
    }
}
