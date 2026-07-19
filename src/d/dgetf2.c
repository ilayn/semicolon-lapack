/**
 * @file dgetf2.c
 * @brief Unblocked LU factorization with explicit loops.
 *
 * Deviation from reference LAPACK: pivot search, row swap, and column
 * scaling use explicit loops instead of BLAS calls (idamax, dswap, dscal)
 * to avoid function call overhead on small per-column operations.
 * Inspired by faer (https://codeberg.org/sarah-quinones/faer).
 */

#include <math.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * Computes an LU factorization of a general `m`-by-`n` matrix `A` using partial
 * pivoting with row interchanges (unblocked algorithm).
 *
 * The factorization has the form
 * @rst
 * .. code-block:: text
 *
 *     A = P * L * U
 * @endrst
 * where P is a permutation matrix, L is lower triangular with unit diagonal
 * elements, and U is upper triangular.
 *
 * This is the right-looking Level 2 BLAS version of the algorithm, processing
 * one column at a time. It uses explicit loops for pivot search, row swap,
 * and column scaling to minimize function call overhead for small matrices.
 *
 * @param[in]     m     The number of rows of the matrix `A`. `m>=0`.
 * @param[in]     n     The number of columns of the matrix `A`. `n>=0`.
 * @param[in,out] A     Array of dimension (`lda`, `n`).
 *                      On entry, the `m`-by-`n` matrix to be factored.
 *                      On exit, the factors L and U from the factorization;
 *                      the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,m)`.
 * @param[out]    ipiv  Array of dimension `min(m,n)`.
 *                      The pivot indices; row i was interchanged with row
 *                      `ipiv[i]`.
 * @param[out]    info
 *                          - `info=0`: successful exit
 *                          - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                            value
 *                          - `info>0`: if `info=i`, U(i,i) is exactly zero. The
 *                            factorization has been completed, but U is exactly
 *                            singular.
 */
void dgetf2(
    const INT m,
    const INT n,
    f64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 sfmin = DBL_MIN;

    INT i, j, k, jp;
    INT minmn = m < n ? m : n;
    f64 abs_val, max_val, pivot, inv, tmp;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("DGETF2", -(*info));
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    for (j = 0; j < minmn; j++) {
        // Find pivot: explicit loop instead of cblas_idamax
        jp = j;
        max_val = fabs(A[j + j * lda]);
        for (i = j + 1; i < m; i++) {
            abs_val = fabs(A[i + j * lda]);
            if (abs_val > max_val) {
                max_val = abs_val;
                jp = i;
            }
        }
        ipiv[j] = jp;

        if (A[jp + j * lda] != ZERO) {
            // Swap rows: explicit loop instead of cblas_dswap
            if (jp != j) {
                for (k = 0; k < n; k++) {
                    tmp = A[j + k * lda];
                    A[j + k * lda] = A[jp + k * lda];
                    A[jp + k * lda] = tmp;
                }
            }

            // Scale column: explicit loop instead of cblas_dscal
            if (j < m - 1) {
                pivot = A[j + j * lda];
                if (fabs(pivot) >= sfmin) {
                    inv = 1.0 / pivot;
                    for (i = j + 1; i < m; i++) {
                        A[i + j * lda] *= inv;
                    }
                } else {
                    for (i = j + 1; i < m; i++) {
                        A[i + j * lda] /= pivot;
                    }
                }
            }

        } else if (*info == 0) {
            *info = j + 1;
        }

        // Rank-1 update: keep cblas_dger for the trailing submatrix
        if (j < minmn - 1) {
            cblas_dger(CblasColMajor, m - j - 1, n - j - 1, -1.0,
                       &A[j + 1 + j * lda], 1,
                       &A[j + (j + 1) * lda], lda,
                       &A[j + 1 + (j + 1) * lda], lda);
        }
    }
}
