/**
 * @file ilaslc.c
 * @brief ILASLC scans a matrix for its last non-zero column.
 */

#include "semicolon_lapack_single.h"

/**
 * ILASLC scans A for its last non-zero column.
 *
 * @param[in] m
 *          The number of rows of the matrix A.
 *
 * @param[in] n
 *          The number of columns of the matrix A.
 *
 * @param[in] A
 *          Double precision array, dimension (lda, n).
 *          The m by n matrix A.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @return The column count up to and including the last non-zero column,
 *         or 0 if the matrix is empty or all zero.
 */
int ilaslc(
    const int m,
    const int n,
    const float* const restrict A,
    const int lda)
{
    const float zero = 0.0f;
    int i, j;

    if (n == 0) {
        return 0;
    } else if (A[0 + (n - 1) * lda] != zero || A[(m - 1) + (n - 1) * lda] != zero) {
        return n;
    } else {
        for (j = n - 1; j >= 0; j--) {
            for (i = 0; i < m; i++) {
                if (A[i + j * lda] != zero) {
                    return j + 1;
                }
            }
        }
        return 0;
    }
}
