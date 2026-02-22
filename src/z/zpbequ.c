/**
 * @file zpbequ.c
 * @brief ZPBEQU computes row and column scalings to equilibrate a Hermitian positive definite band matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBEQU computes row and column scalings intended to equilibrate a
 * Hermitian positive definite band matrix A and reduce its condition
 * number (with respect to the two-norm). S contains the scale factors,
 * S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
 * elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.
 *
 * @param[in]     uplo   = 'U': Upper triangular of A is stored
 *                        = 'L': Lower triangular of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     AB     The banded matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    S      The scale factors for A. Array of dimension (n).
 * @param[out]    scond  Ratio of smallest to largest S(i).
 * @param[out]    amax   Absolute value of largest matrix element.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element is nonpositive.
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
