/**
 * @file sgennd.c
 * @brief SGENND tests that its argument has a non-negative diagonal.
 */

#include "verify.h"

/**
 * SGENND tests that its argument has a non-negative diagonal.
 *
 * @param[in] m   The number of rows in A.
 * @param[in] n   The number of columns in A.
 * @param[in] A   The matrix, dimension (lda, n).
 * @param[in] lda Leading dimension of A.
 * @return 1 if diagonal is non-negative, 0 otherwise.
 */
INT sgennd(const INT m, const INT n, const f32* const restrict A, const INT lda)
{
    INT k = m < n ? m : n;
    for (INT i = 0; i < k; i++) {
        if (A[i + i * lda] < 0.0f) {
            return 0;
        }
    }
    return 1;
}
