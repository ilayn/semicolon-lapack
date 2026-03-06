/**
 * @file cgennd.c
 * @brief CGENND tests that its argument has a real, non-negative diagonal.
 */

#include "verify.h"

/**
 * CGENND tests that its argument has a real, non-negative diagonal.
 *
 * @param[in] m   The number of rows in A.
 * @param[in] n   The number of columns in A.
 * @param[in] A   The matrix, dimension (lda, n).
 * @param[in] lda Leading dimension of A.
 * @return 1 if diagonal is real and non-negative, 0 otherwise.
 */
INT cgennd(const INT m, const INT n, const c64* const restrict A, const INT lda)
{
    INT k = m < n ? m : n;
    for (INT i = 0; i < k; i++) {
        c64 aii = A[i + i * lda];
        if (crealf(aii) < 0.0f || cimagf(aii) != 0.0f) {
            return 0;
        }
    }
    return 1;
}
