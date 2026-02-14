/**
 * @file sppequ.c
 * @brief SPPEQU computes row and column scalings for equilibration of packed symmetric matrices.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SPPEQU computes row and column scalings intended to equilibrate a
 * symmetric positive definite matrix A in packed storage and reduce
 * its condition number (with respect to the two-norm). S contains the
 * scale factors, S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix
 * B with elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.
 * This choice of S puts the condition number of B within a factor N of
 * the smallest possible condition number over all possible diagonal
 * scalings.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     AP     The upper or lower triangle of the symmetric matrix A,
 *                       packed columnwise in a linear array. The j-th column
 *                       of A is stored in the array AP as follows:
 *                       if uplo = 'U', AP[i + j*(j+1)/2] = A(i,j) for 0<=i<=j;
 *                       if uplo = 'L', AP[i + j*(2*n-j-1)/2] = A(i,j) for j<=i<n.
 *                       Array of dimension (n*(n+1)/2).
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
void sppequ(
    const char* uplo,
    const int n,
    const f32* const restrict AP,
    f32* const restrict S,
    f32* scond,
    f32* amax,
    int* info)
{
    // sppequ.f lines 131-133: Parameters
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    // sppequ.f lines 154-164: Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("SPPEQU", -(*info));
        return;
    }

    // sppequ.f lines 168-172: Quick return if possible
    if (n == 0) {
        *scond = ONE;
        *amax = ZERO;
        return;
    }

    // sppequ.f lines 176-178: Initialize SMIN and AMAX
    S[0] = AP[0];
    f32 smin = S[0];
    *amax = S[0];

    if (upper) {
        // sppequ.f lines 180-191: UPLO = 'U': Upper triangle of A is stored.
        // Find the minimum and maximum diagonal elements.
        // In upper packed storage, diagonal element A(i,i) is at position
        // i + i*(i+1)/2 (0-based), which equals i*(i+3)/2
        // The Fortran code uses JJ = JJ + I for I = 2..N (1-based)
        // which corresponds to jj = jj + (i+1) for i = 1..n-1 (0-based)
        int jj = 0;  // sppequ.f line 185: JJ = 1 (but 0-based here)
        for (int i = 1; i < n; i++) {
            // sppequ.f line 187: JJ = JJ + I (where I is 1-based, so I = i+1)
            jj = jj + (i + 1);
            S[i] = AP[jj];
            if (S[i] < smin) smin = S[i];
            if (S[i] > *amax) *amax = S[i];
        }
    } else {
        // sppequ.f lines 193-205: UPLO = 'L': Lower triangle of A is stored.
        // Find the minimum and maximum diagonal elements.
        // In lower packed storage, diagonal element A(i,i) is at position
        // i + (i-1)*(2*n-i)/2 (1-based Fortran)
        // The Fortran code uses JJ = JJ + N - I + 2 for I = 2..N (1-based)
        int jj = 0;  // sppequ.f line 198: JJ = 1 (but 0-based here)
        for (int i = 1; i < n; i++) {
            // sppequ.f line 200: JJ = JJ + N - I + 2 (where I is 1-based, so I = i+1)
            // JJ = JJ + N - (i+1) + 2 = JJ + N - i + 1
            jj = jj + n - i + 1;
            S[i] = AP[jj];
            if (S[i] < smin) smin = S[i];
            if (S[i] > *amax) *amax = S[i];
        }
    }

    if (smin <= ZERO) {
        // sppequ.f lines 207-216: Find the first non-positive diagonal element and return.
        for (int i = 0; i < n; i++) {
            if (S[i] <= ZERO) {
                *info = i + 1;  // 1-based for error reporting
                return;
            }
        }
    } else {
        // sppequ.f lines 217-224: Set the scale factors to the reciprocals
        // of the diagonal elements.
        for (int i = 0; i < n; i++) {
            S[i] = ONE / sqrtf(S[i]);
        }

        // sppequ.f line 228: Compute SCOND = min(S(I)) / max(S(I))
        *scond = sqrtf(smin) / sqrtf(*amax);
    }
}
