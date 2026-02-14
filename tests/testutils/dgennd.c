/**
 * @file dgennd.c
 * @brief DGENND tests that its argument has a non-negative diagonal.
 */

#include "verify.h"

/**
 * DGENND tests that its argument has a non-negative diagonal.
 *
 * @param[in] m   The number of rows in A.
 * @param[in] n   The number of columns in A.
 * @param[in] A   The matrix, dimension (lda, n).
 * @param[in] lda Leading dimension of A.
 * @return 1 if diagonal is non-negative, 0 otherwise.
 */
int dgennd(const int m, const int n, const f64* const restrict A, const int lda)
{
    int k = m < n ? m : n;
    for (int i = 0; i < k; i++) {
        if (A[i + i * lda] < 0.0) {
            return 0;
        }
    }
    return 1;
}
