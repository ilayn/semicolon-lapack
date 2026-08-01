/**
 * @file zpbequ.c
 * @brief ZPBEQU computes row and column scalings to equilibrate a Hermitian positive definite band matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBEQU computes row and column scalings intended to equilibrate a
 * Hermitian positive definite band matrix A and reduce its condition
 * number (with respect to the two-norm). S contains the scale factors,
 * S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
 * elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal. This
 * choice of S puts the condition number of B within a factor N of the
 * smallest possible condition number over all possible diagonal
 * scalings.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular of A is stored
 *                       - `'L'`: Lower triangular of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in]     AB     Array of dimension (`ldab`, `n`).
 *                       The upper or lower triangle of the Hermitian band
 *                       matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of A is stored in the j-th
 *                       column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = A(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = A(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[out]    S      Array of dimension `n`.
 *                       If `info=0`, `S` contains the scale factors for A.
 * @param[out]    scond  If `info=0`, `S` contains the ratio of the smallest S(i)
 *                       to the largest S(i). If `scond>=0.1` and `amax` is
 *                       neither too large nor too small, it is not worth
 *                       scaling by `S`.
 * @param[out]    amax   Absolute value of largest matrix element. If `amax` is
 *                       very close to overflow or very close to underflow, the
 *                       matrix should be scaled.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, the i-th diagonal element is
 *                           nonpositive.
 */
void zpbequ(
    const char* uplo,
    const INT n,
    const INT kd,
    const c128* restrict AB,
    const INT ldab,
    f64* restrict S,
    f64* scond,
    f64* amax,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT upper;
    INT i, j;
    f64 smin;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (ldab < kd + 1) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("ZPBEQU", -(*info));
        return;
    }

    if (n == 0) {
        *scond = ONE;
        *amax = ZERO;
        return;
    }

    if (upper) {
        j = kd;
    } else {
        j = 0;
    }

    // Initialize smin and amax
    S[0] = creal(AB[j + 0 * ldab]);
    smin = S[0];
    *amax = S[0];

    // Find the minimum and maximum diagonal elements
    for (i = 1; i < n; i++) {
        S[i] = creal(AB[j + i * ldab]);
        if (smin > S[i]) smin = S[i];
        if (*amax < S[i]) *amax = S[i];
    }

    if (smin <= ZERO) {
        // Find the first non-positive diagonal element and return
        for (i = 0; i < n; i++) {
            if (S[i] <= ZERO) {
                *info = i + 1;
                return;
            }
        }
    } else {
        // Set the scale factors to the reciprocals of the diagonal elements
        for (i = 0; i < n; i++) {
            S[i] = ONE / sqrt(S[i]);
        }

        // Compute scond = min(S(i)) / max(S(i))
        *scond = sqrt(smin) / sqrt(*amax);
    }
}
