/**
 * @file dlarscl2.c
 * @brief DLARSCL2 performs reciprocal diagonal scaling on a matrix.
 */

#include "semicolon_lapack_double.h"

/**
 * DLARSCL2 performs a reciprocal diagonal scaling on a matrix:
 *   X becomes inv(D) * X
 * where the diagonal matrix D is stored as a vector.
 *
 * Eventually to be replaced by BLAS_dge_diag_scale in the new BLAS
 * standard.
 *
 * @param[in]     m     The number of rows of D and X. m >= 0.
 * @param[in]     n     The number of columns of X. n >= 0.
 * @param[in]     D     Double precision array, length m.
 *                      Diagonal matrix D, stored as a vector of length m.
 * @param[in,out] X     Double precision array, dimension (ldx, n).
 *                      On entry, the matrix X to be scaled by D.
 *                      On exit, the scaled matrix.
 * @param[in]     ldx   The leading dimension of the matrix X. ldx >= m.
 */
void dlarscl2(
    const INT m,
    const INT n,
    const f64* restrict D,
    f64* restrict X,
    const INT ldx)
{
    INT i, j;

    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            X[i + j * ldx] = X[i + j * ldx] / D[i];
        }
    }
}
