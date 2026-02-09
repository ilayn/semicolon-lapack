/**
 * @file ilaslr.c
 * @brief ILASLR scans a matrix for its last non-zero row.
 */

#include "semicolon_lapack_single.h"

/**
 * ILASLR scans A for its last non-zero row.
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
 * @return The row count up to and including the last non-zero row,
 *         or 0 if the matrix is empty or all zero.
 */
int ilaslr(
    const int m,
    const int n,
    const float* const restrict A,
    const int lda)
{
    const float zero = 0.0f;
    int i, j, result;

    if (m == 0) {
        return 0;
    } else if (A[(m - 1) + 0 * lda] != zero || A[(m - 1) + (n - 1) * lda] != zero) {
        return m;
    } else {
        result = 0;
        for (j = 0; j < n; j++) {
            i = m - 1;
            while (i >= 0 && A[i + j * lda] == zero) {
                i--;
            }
            if (i + 1 > result) {
                result = i + 1;
            }
        }
        return result;
    }
}
