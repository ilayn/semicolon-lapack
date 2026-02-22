/**
 * @file zppequ.c
 * @brief ZPPEQU computes row and column scalings for equilibration of packed Hermitian matrices.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPPEQU computes row and column scalings intended to equilibrate a
 * Hermitian positive definite matrix A in packed storage and reduce
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
 * @param[in]     AP     The upper or lower triangle of the Hermitian matrix A,
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
void zppequ(
    const char* uplo,
    const INT n,
    const c128* restrict AP,
    f64* restrict S,
    f64* scond,
    f64* amax,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("ZPPEQU", -(*info));
        return;
    }

    if (n == 0) {
        *scond = ONE;
        *amax = ZERO;
        return;
    }

    S[0] = creal(AP[0]);
    f64 smin = S[0];
    *amax = S[0];

    if (upper) {
        INT jj = 0;
        for (INT i = 1; i < n; i++) {
            jj = jj + (i + 1);
            S[i] = creal(AP[jj]);
            if (S[i] < smin) smin = S[i];
            if (S[i] > *amax) *amax = S[i];
        }
    } else {
        INT jj = 0;
        for (INT i = 1; i < n; i++) {
            jj = jj + n - i + 1;
            S[i] = creal(AP[jj]);
            if (S[i] < smin) smin = S[i];
            if (S[i] > *amax) *amax = S[i];
        }
    }

    if (smin <= ZERO) {
        for (INT i = 0; i < n; i++) {
            if (S[i] <= ZERO) {
                *info = i + 1;
                return;
            }
        }
    } else {
        for (INT i = 0; i < n; i++) {
            S[i] = ONE / sqrt(S[i]);
        }

        *scond = sqrt(smin) / sqrt(*amax);
    }
}
