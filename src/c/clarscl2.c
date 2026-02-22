/**
 * @file clarscl2.c
 * @brief CLARSCL2 performs reciprocal diagonal scaling on a matrix.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLARSCL2 performs a reciprocal diagonal scaling on a matrix:
 *   x <-- inv(D) * x
 * where the DOUBLE PRECISION diagonal matrix D is stored as a vector.
 *
 * Eventually to be replaced by BLAS_zge_diag_scale in the new BLAS
 * standard.
 *
 * @param[in]     m     The number of rows of D and X. m >= 0.
 * @param[in]     n     The number of columns of X. n >= 0.
 * @param[in]     D     Single precision array, length m.
 *                      Diagonal matrix D, stored as a vector of length m.
 * @param[in,out] X     Complex*16 array, dimension (ldx, n).
 *                      On entry, the matrix X to be scaled by D.
 *                      On exit, the scaled matrix.
 * @param[in]     ldx   The leading dimension of the matrix X. ldx >= m.
 */
void clarscl2(
    const INT m,
    const INT n,
    const f32* restrict D,
    c64* restrict X,
    const INT ldx)
{
    INT i, j;

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            X[i + j * ldx] = X[i + j * ldx] / D[i];
        }
    }
}
