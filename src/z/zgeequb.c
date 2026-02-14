/**
 * @file zgeequb.c
 * @brief Computes row and column scaling factors with radix constraint.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGEEQUB computes row and column scalings intended to equilibrate an
 * M-by-N matrix A and reduce its condition number. R returns the row
 * scale factors and C the column scale factors, chosen to try to make
 * the largest element in each row and column of the matrix B with
 * elements B(i,j)=R(i)*A(i,j)*C(j) have an absolute value of at most
 * the radix.
 *
 * R(i) and C(j) are restricted to be a power of the radix between
 * SMLNUM = smallest safe number and BIGNUM = largest safe number. Use
 * of these scaling factors is not guaranteed to reduce the condition
 * number of A but works well in practice.
 *
 * This routine differs from ZGEEQU by restricting the scaling factors
 * to a power of the radix. Barring over- and underflow, scaling by
 * these factors introduces no additional rounding errors. However, the
 * scaled entries' magnitudes are no longer approximately 1 but lie
 * between sqrt(radix) and 1/sqrt(radix).
 *
 * @param[in]     m       The number of rows of the matrix A (m >= 0).
 * @param[in]     n       The number of columns of the matrix A (n >= 0).
 * @param[in]     A       The M-by-N matrix whose equilibration factors are
 *                        to be computed. Array of dimension (lda, n).
 * @param[in]     lda     The leading dimension of the array A (lda >= max(1,m)).
 * @param[out]    R       If info = 0 or info > m, R contains the row scale factors
 *                        for A. Array of dimension m.
 * @param[out]    C       If info = 0, C contains the column scale factors for A.
 *                        Array of dimension n.
 * @param[out]    rowcnd  If info = 0 or info > m, rowcnd contains the ratio of the
 *                        smallest R(i) to the largest R(i). If rowcnd >= 0.1 and
 *                        amax is neither too large nor too small, it is not worth
 *                        scaling by R.
 * @param[out]    colcnd  If info = 0, colcnd contains the ratio of the smallest
 *                        C(j) to the largest C(j). If colcnd >= 0.1, it is not
 *                        worth scaling by C.
 * @param[out]    amax    Absolute value of largest matrix element. If amax is very
 *                        close to overflow or very close to underflow, the matrix
 *                        should be scaled.
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, and i is
 *                         - <= m: the i-th row of A is exactly zero (1-based)
 *                         - > m: the (i-m)-th column of A is exactly zero (1-based)
 */
void zgeequb(
    const int m,
    const int n,
    const c128* restrict A,
    const int lda,
    f64* restrict R,
    f64* restrict C,
    f64* rowcnd,
    f64* colcnd,
    f64* amax,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int i, j;
    f64 bignum, rcmax, rcmin, smlnum, radix, logrdx;

    // Test the input parameters
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZGEEQUB", -(*info));
        return;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        *rowcnd = ONE;
        *colcnd = ONE;
        *amax = ZERO;
        return;
    }

    // Get machine constants. Assume SMLNUM is a power of the radix.
    smlnum = DBL_MIN;
    bignum = ONE / smlnum;
    radix = FLT_RADIX;
    logrdx = log(radix);

    // Compute row scale factors
    for (i = 0; i < m; i++) {
        R[i] = ZERO;
    }

    // Find the maximum element in each row
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            f64 abs_val = fabs(creal(A[i + j * lda])) + fabs(cimag(A[i + j * lda]));
            if (abs_val > R[i]) {
                R[i] = abs_val;
            }
        }
    }

    // Round to power of radix
    for (i = 0; i < m; i++) {
        if (R[i] > ZERO) {
            R[i] = pow(radix, (int)(log(R[i]) / logrdx));
        }
    }

    // Find the maximum and minimum scale factors
    rcmin = bignum;
    rcmax = ZERO;
    for (i = 0; i < m; i++) {
        if (R[i] > rcmax) {
            rcmax = R[i];
        }
        if (R[i] < rcmin) {
            rcmin = R[i];
        }
    }
    *amax = rcmax;

    if (rcmin == ZERO) {
        // Find the first zero scale factor and return an error code
        for (i = 0; i < m; i++) {
            if (R[i] == ZERO) {
                *info = i + 1;  // 1-based index for error reporting
                return;
            }
        }
    } else {
        // Invert the scale factors
        for (i = 0; i < m; i++) {
            f64 ri = R[i];
            if (ri < smlnum) {
                ri = smlnum;
            }
            if (ri > bignum) {
                ri = bignum;
            }
            R[i] = ONE / ri;
        }

        // Compute rowcnd = min(R(i)) / max(R(i))
        f64 rcmin_clamped = rcmin > smlnum ? rcmin : smlnum;
        f64 rcmax_clamped = rcmax < bignum ? rcmax : bignum;
        *rowcnd = rcmin_clamped / rcmax_clamped;
    }

    // Compute column scale factors
    for (j = 0; j < n; j++) {
        C[j] = ZERO;
    }

    // Find the maximum element in each column,
    // assuming the row scaling computed above
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            f64 scaled_val = (fabs(creal(A[i + j * lda])) + fabs(cimag(A[i + j * lda]))) * R[i];
            if (scaled_val > C[j]) {
                C[j] = scaled_val;
            }
        }
        // Round to power of radix
        if (C[j] > ZERO) {
            C[j] = pow(radix, (int)(log(C[j]) / logrdx));
        }
    }

    // Find the maximum and minimum scale factors
    rcmin = bignum;
    rcmax = ZERO;
    for (j = 0; j < n; j++) {
        if (C[j] < rcmin) {
            rcmin = C[j];
        }
        if (C[j] > rcmax) {
            rcmax = C[j];
        }
    }

    if (rcmin == ZERO) {
        // Find the first zero scale factor and return an error code
        for (j = 0; j < n; j++) {
            if (C[j] == ZERO) {
                *info = m + j + 1;  // 1-based index for error reporting
                return;
            }
        }
    } else {
        // Invert the scale factors
        for (j = 0; j < n; j++) {
            f64 cj = C[j];
            if (cj < smlnum) {
                cj = smlnum;
            }
            if (cj > bignum) {
                cj = bignum;
            }
            C[j] = ONE / cj;
        }

        // Compute colcnd = min(C(j)) / max(C(j))
        f64 rcmin_clamped = rcmin > smlnum ? rcmin : smlnum;
        f64 rcmax_clamped = rcmax < bignum ? rcmax : bignum;
        *colcnd = rcmin_clamped / rcmax_clamped;
    }
}
