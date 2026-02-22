/**
 * @file zladiv.c
 * @brief ZLADIV performs complex division in real arithmetic, avoiding
 *        unnecessary overflow.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLADIV := X / Y, where X and Y are complex.  The computation of X / Y
 * will not overflow on an intermediary step unless the results
 * overflows.
 *
 * @param[in] X     Double complex scalar.
 * @param[in] Y     Double complex scalar.
 *                   The complex scalars X and Y.
 *
 * @return The complex quotient X / Y.
 */
c128 zladiv(const c128 X, const c128 Y)
{
    f64 zi, zr;

    dladiv(creal(X), cimag(X), creal(Y), cimag(Y), &zr, &zi);
    return CMPLX(zr, zi);
}
