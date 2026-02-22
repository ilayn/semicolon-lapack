/**
 * @file zpoequb.c
 * @brief ZPOEQUB computes row and column scalings for equilibrating a Hermitian positive definite matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPOEQUB computes row and column scalings intended to equilibrate a
 * Hermitian positive definite matrix A and reduce its condition number
 * (with respect to the two-norm).  S contains the scale factors,
 * S(i) = 1/sqrt(A(i,i)), chosen so that the scaled matrix B with
 * elements B(i,j) = S(i)*A(i,j)*S(j) has ones on the diagonal.  This
 * choice of S puts the condition number of B within a factor N of the
 * smallest possible condition number over all possible diagonal
 * scalings.
 *
 * This routine differs from ZPOEQU by restricting the scaling factors
 * to a power of the radix.  Barring over- and underflow, scaling by
 * these factors introduces no additional rounding errors.  However, the
 * scaled diagonal entries are no longer approximately 1 but lie
 * between sqrt(radix) and 1/sqrt(radix).
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] A
 *          Double complex array, dimension (lda, n).
 *          The N-by-N Hermitian positive definite matrix whose scaling
 *          factors are to be computed. Only the diagonal elements of A
 *          are referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] S
 *          Double precision array, dimension (n).
 *          If info = 0, S contains the scale factors for A.
 *
 * @param[out] scond
 *          If info = 0, S contains the ratio of the smallest S(i) to
 *          the largest S(i). If scond >= 0.1 and amax is neither too
 *          large nor too small, it is not worth scaling by S.
 *
 * @param[out] amax
 *          Absolute value of largest matrix element. If amax is very
 *          close to overflow or very close to underflow, the matrix
 *          should be scaled.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element is nonpositive.
 */
void zpoequb(
    const INT n,
    const c128* restrict A,
    const INT lda,
    f64* restrict S,
    f64* scond,
    f64* amax,
    INT* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    INT i;
    f64 smin, base, tmp;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("ZPOEQUB", -(*info));
        return;
    }

    if (n == 0) {
        *scond = one;
        *amax = zero;
        return;
    }

    base = dlamch("B");
    tmp = -0.5 / log(base);

    S[0] = creal(A[0 + 0 * lda]);
    smin = S[0];
    *amax = S[0];
    for (i = 1; i < n; i++) {
        S[i] = creal(A[i + i * lda]);
        if (S[i] < smin) smin = S[i];
        if (S[i] > *amax) *amax = S[i];
    }

    if (smin <= zero) {
        for (i = 0; i < n; i++) {
            if (S[i] <= zero) {
                *info = i + 1;
                return;
            }
        }
    } else {
        for (i = 0; i < n; i++) {
            S[i] = pow(base, (INT)(tmp * log(S[i])));
        }

        *scond = sqrt(smin) / sqrt(*amax);
    }
}
