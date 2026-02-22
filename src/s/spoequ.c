/**
 * @file spoequ.c
 * @brief SPOEQU computes row and column scalings for equilibration.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SPOEQU computes row and column scalings intended to equilibrate a
 * symmetric positive definite matrix A and reduce its condition number
 * (with respect to the two-norm). S contains the scale factors,
 * S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
 * elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal. This
 * choice of S puts the condition number of B within a factor N of the
 * smallest possible condition number over all possible diagonal
 * scalings.
 *
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     A      The n-by-n symmetric positive definite matrix whose
 *                       scaling factors are to be computed. Only the diagonal
 *                       elements of A are referenced. Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= max(1, n).
 * @param[out]    S      If info = 0, S contains the scale factors for A.
 *                       Array of dimension (n).
 * @param[out]    scond  If info = 0, S contains the ratio of the smallest S(i)
 *                       to the largest S(i). If scond >= 0.1 and amax is
 *                       neither too large nor too small, it is not worth
 *                       scaling by S.
 * @param[out]    amax   Absolute value of largest matrix element. If amax is
 *                       very close to overflow or very close to underflow, the
 *                       matrix should be scaled.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element is nonpositive.
 */
void spoequ(
    const INT n,
    const f32* restrict A,
    const INT lda,
    f32* restrict S,
    f32* scond,
    f32* amax,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    // Test the input parameters
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("SPOEQU", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) {
        *scond = ONE;
        *amax = ZERO;
        return;
    }

    // Find the minimum and maximum diagonal elements.
    S[0] = A[0];
    f32 smin = S[0];
    *amax = S[0];
    for (INT i = 1; i < n; i++) {
        S[i] = A[i + i * lda];
        if (S[i] < smin) smin = S[i];
        if (S[i] > *amax) *amax = S[i];
    }

    if (smin <= ZERO) {
        // Find the first non-positive diagonal element and return.
        for (INT i = 0; i < n; i++) {
            if (S[i] <= ZERO) {
                *info = i + 1;  // 1-based for error reporting
                return;
            }
        }
    } else {
        // Set the scale factors to the reciprocals of the diagonal elements.
        for (INT i = 0; i < n; i++) {
            S[i] = ONE / sqrtf(S[i]);
        }

        // Compute SCOND = min(S(I)) / max(S(I))
        *scond = sqrtf(smin) / sqrtf(*amax);
    }
}
