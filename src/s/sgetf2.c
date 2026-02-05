/**
 * @file sgetf2.c
 * @brief Unblocked LU factorization using Level 2 BLAS (single precision).
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
 * This is the right-looking Level 2 BLAS version of the algorithm.
 *
 * @param[in]     m     The number of rows of the matrix A (m >= 0).
 * @param[in]     n     The number of columns of the matrix A (n >= 0).
 * @param[in,out] A     On entry, the M-by-N matrix to be factored.
 *                      On exit, the factors L and U from the factorization;
 *                      the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,m)).
 * @param[out]    ipiv  The pivot indices; row i was interchanged with row
 *                      ipiv[i]. Array of dimension min(m,n), 0-based.
 * @param[out]    info  Exit status.
 *                      - = 0: successful exit
 *                      - < 0: if info = -i, the i-th argument had an illegal value
 *                      - > 0: if info = i, U(i-1,i-1) is exactly zero.
 */
void sgetf2(
    const int m,
    const int n,
    float * const restrict A,
    const int lda,
    int * const restrict ipiv,
    int *info)
{
    const float ONE = 1.0f;
    const float NEG_ONE = -1.0f;
    const float ZERO = 0.0f;
    const float sfmin = FLT_MIN;

    int j, jp;
    int minmn = m < n ? m : n;

    // Test the input parameters
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

    // Quick return if possible
    if (m == 0 || n == 0) {
        return;
    }

    for (j = 0; j < minmn; j++) {
        // Find pivot and test for singularity
        jp = j + cblas_isamax(m - j, &A[j + j * lda], 1);
        ipiv[j] = jp;

        if (A[jp + j * lda] != ZERO) {
            // Apply the interchange to columns 0:n-1
            if (jp != j) {
                cblas_sswap(n, &A[j], lda, &A[jp], lda);
            }

            // Compute elements j+1:m-1 of j-th column
            if (j < m - 1) {
                float pivot = A[j + j * lda];
                if (fabsf(pivot) >= sfmin) {
                    cblas_sscal(m - j - 1, ONE / pivot, &A[j + 1 + j * lda], 1);
                } else {
                    // Avoid division by very small numbers
                    for (int i = 1; i < m - j; i++) {
                        A[j + i + j * lda] /= pivot;
                    }
                }
            }

        } else if (*info == 0) {
            // First zero pivot encountered
            *info = j + 1;
        }

        // Update trailing submatrix
        if (j < minmn - 1) {
            cblas_sger(CblasColMajor, m - j - 1, n - j - 1, NEG_ONE,
                       &A[j + 1 + j * lda], 1,
                       &A[j + (j + 1) * lda], lda,
                       &A[j + 1 + (j + 1) * lda], lda);
        }
    }
}
