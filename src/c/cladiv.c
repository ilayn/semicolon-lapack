/**
 * @file cladiv.c
 * @brief CLADIV performs complex division in real arithmetic, avoiding
 *        unnecessary overflow.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLADIV := X / Y, where X and Y are complex.  The computation of X / Y
 * will not overflow on an intermediary step unless the results
 * overflows.
 *
 * @param[in] X     Single complex scalar.
 * @param[in] Y     Single complex scalar.
 *                   The complex scalars X and Y.
 *
 * @return The complex quotient X / Y.
 */
c64 cladiv(const c64 X, const c64 Y)
{
    f32 zi, zr;

    sladiv(crealf(X), cimagf(X), crealf(Y), cimagf(Y), &zr, &zi);
    return CMPLXF(zr, zi);
}
