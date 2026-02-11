/**
 * @file sgetf2.c
 * @brief Unblocked LU factorization with explicit loops.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * Computes an LU factorization of a general M-by-N matrix A using partial
 * pivoting with row interchanges (unblocked algorithm).
 *
 * The factorization has the form:
 *
 *     A = P * L * U
 *
 * where P is a permutation matrix, L is lower triangular with unit diagonal
 * elements, and U is upper triangular.
 *
 * This is the right-looking Level 2 BLAS version of the algorithm, processing
 * one column at a time. It uses explicit loops for pivot search, row swap,
 * and column scaling to minimize function call overhead for small matrices.
 *
 * @param[in]     m     The number of rows of the matrix A (m >= 0).
 * @param[in]     n     The number of columns of the matrix A (n >= 0).
 * @param[in,out] A     On entry, the M-by-N matrix to be factored.
 *                      On exit, the factors L and U from the factorization;
 *                      the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,m)).
 * @param[out]    ipiv  The pivot indices; row i was interchanged with row
 *                      ipiv[i]. Array of dimension min(m,n), 0-based.
 *
 * @param[out]   info
 *                           Exit status.
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but U is exactly singular.
 */
void sgetf2(
    const int m,
    const int n,
    float * const restrict A,
    const int lda,
    int * const restrict ipiv,
    int *info)
{
    const float ZERO = 0.0f;
    const float sfmin = FLT_MIN;

    int i, j, k, jp;
    int minmn = m < n ? m : n;
    float abs_val, max_val, pivot, inv, tmp;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("SGETF2", -(*info));
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    for (j = 0; j < minmn; j++) {
        // Find pivot: explicit loop instead of cblas_idamax
        jp = j;
        max_val = fabsf(A[j + j * lda]);
        for (i = j + 1; i < m; i++) {
            abs_val = fabsf(A[i + j * lda]);
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
                if (fabsf(pivot) >= sfmin) {
                    inv = 1.0f / pivot;
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
            cblas_sger(CblasColMajor, m - j - 1, n - j - 1, -1.0f,
                       &A[j + 1 + j * lda], 1,
                       &A[j + (j + 1) * lda], lda,
                       &A[j + 1 + (j + 1) * lda], lda);
        }
    }
}
