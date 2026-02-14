/**
 * @file zgeequ.c
 * @brief Computes row and column scaling factors to equilibrate a matrix.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGEEQU computes row and column scalings intended to equilibrate an
 * M-by-N matrix A and reduce its condition number. R returns the row
 * scale factors and C the column scale factors, chosen to try to make
 * the largest element in each row and column of the matrix B with
 * elements B(i,j)=R(i)*A(i,j)*C(j) have absolute value 1.
 *
 * R(i) and C(j) are restricted to be between SMLNUM = smallest safe
 * number and BIGNUM = largest safe number. Use of these scaling
 * factors is not guaranteed to reduce the condition number of A but
 * works well in practice.
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
void zgeequ(
    const int m,
    const int n,
    const double complex* const restrict A,
    const int lda,
    double* const restrict R,
    double* const restrict C,
    double* rowcnd,
    double* colcnd,
    double* amax,
    int* info)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    int i, j;
    double bignum, rcmax, rcmin, smlnum;

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
        xerbla("ZGEEQU", -(*info));
        return;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        *rowcnd = ONE;
        *colcnd = ONE;
        *amax = ZERO;
        return;
    }

    // Get machine constants
    // DLAMCH("S") returns safe minimum, such that 1/sfmin does not overflow
    smlnum = DBL_MIN;
    bignum = ONE / smlnum;

    // Compute row scale factors
    for (i = 0; i < m; i++) {
        R[i] = ZERO;
    }

    // Find the maximum element in each row
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            double abs_val = cabs1(A[i + j * lda]);
            if (abs_val > R[i]) {
                R[i] = abs_val;
            }
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
            double ri = R[i];
            if (ri < smlnum) {
                ri = smlnum;
            }
            if (ri > bignum) {
                ri = bignum;
            }
            R[i] = ONE / ri;
        }

        // Compute rowcnd = min(R(i)) / max(R(i))
        double rcmin_clamped = rcmin > smlnum ? rcmin : smlnum;
        double rcmax_clamped = rcmax < bignum ? rcmax : bignum;
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
            double scaled_val = cabs1(A[i + j * lda]) * R[i];
            if (scaled_val > C[j]) {
                C[j] = scaled_val;
            }
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
            double cj = C[j];
            if (cj < smlnum) {
                cj = smlnum;
            }
            if (cj > bignum) {
                cj = bignum;
            }
            C[j] = ONE / cj;
        }

        // Compute colcnd = min(C(j)) / max(C(j))
        double rcmin_clamped = rcmin > smlnum ? rcmin : smlnum;
        double rcmax_clamped = rcmax < bignum ? rcmax : bignum;
        *colcnd = rcmin_clamped / rcmax_clamped;
    }
}
