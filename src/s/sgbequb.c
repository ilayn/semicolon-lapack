/**
 * @file sgbequb.c
 * @brief Computes row and column scaling factors with radix constraint for band matrices.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_single.h"

/**
 * SGBEQUB computes row and column scalings intended to equilibrate an
 * M-by-N band matrix A and reduce its condition number. R returns the row
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
 * This routine differs from SGBEQU by restricting the scaling factors
 * to a power of the radix. Barring over- and underflow, scaling by
 * these factors introduces no additional rounding errors. However, the
 * scaled entries' magnitudes are no longer approximately 1 but lie
 * between sqrt(radix) and 1/sqrt(radix).
 *
 * @param[in]     m       The number of rows of the matrix A (m >= 0).
 * @param[in]     n       The number of columns of the matrix A (n >= 0).
 * @param[in]     kl      The number of subdiagonals within the band of A (kl >= 0).
 * @param[in]     ku      The number of superdiagonals within the band of A (ku >= 0).
 * @param[in]     AB      The band matrix A, stored in band format.
 *                        Array of dimension (ldab, n).
 *                        The matrix A is stored in rows 0 to kl+ku, so that
 *                        AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku) <= i <= min(m-1,j+kl).
 * @param[in]     ldab    The leading dimension of the array AB (ldab >= kl+ku+1).
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
 * @param[out]    info    Exit status:
 *                        - = 0: successful exit
 *                        - < 0: if info = -i, the i-th argument had an illegal value
 *                        - > 0: if info = i, and i is
 *                               <= m: the i-th row of A is exactly zero (1-based)
 *                               > m: the (i-m)-th column of A is exactly zero (1-based)
 */
void sgbequb(
    const int m,
    const int n,
    const int kl,
    const int ku,
    const float * const restrict AB,
    const int ldab,
    float * const restrict R,
    float * const restrict C,
    float *rowcnd,
    float *colcnd,
    float *amax,
    int *info)
{
    const float ONE = 1.0f;
    const float ZERO = 0.0f;

    int i, j;
    float bignum, rcmax, rcmin, smlnum, radix, logrdx;

    /* Test the input parameters */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kl < 0) {
        *info = -3;
    } else if (ku < 0) {
        *info = -4;
    } else if (ldab < kl + ku + 1) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SGBEQUB", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0) {
        *rowcnd = ONE;
        *colcnd = ONE;
        *amax = ZERO;
        return;
    }

    /* Get machine constants. Assume SMLNUM is a power of the radix. */
    smlnum = FLT_MIN;
    bignum = ONE / smlnum;
    radix = FLT_RADIX;  /* Base of the floating-point representation */
    logrdx = logf(radix);

    /* Compute row scale factors */
    for (i = 0; i < m; i++) {
        R[i] = ZERO;
    }

    /* Find the maximum element in each row */
    for (j = 0; j < n; j++) {
        /*
         * In 0-based indexing:
         * Row range for column j: max(0, j-ku) to min(m-1, j+kl)
         * Band storage: AB[ku + i - j + j*ldab] = A(i,j)
         */
        int i_start = (j - ku > 0) ? j - ku : 0;
        int i_end = (j + kl < m - 1) ? j + kl : m - 1;
        for (i = i_start; i <= i_end; i++) {
            float abs_val = fabsf(AB[ku + i - j + j * ldab]);
            if (abs_val > R[i]) {
                R[i] = abs_val;
            }
        }
    }

    /* Round to power of radix */
    for (i = 0; i < m; i++) {
        if (R[i] > ZERO) {
            R[i] = powf(radix, (int)(logf(R[i]) / logrdx));
        }
    }

    /* Find the maximum and minimum scale factors */
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
        /* Find the first zero scale factor and return an error code */
        for (i = 0; i < m; i++) {
            if (R[i] == ZERO) {
                *info = i + 1;  /* 1-based index for error reporting */
                return;
            }
        }
    } else {
        /* Invert the scale factors */
        for (i = 0; i < m; i++) {
            float ri = R[i];
            if (ri < smlnum) {
                ri = smlnum;
            }
            if (ri > bignum) {
                ri = bignum;
            }
            R[i] = ONE / ri;
        }

        /* Compute rowcnd = min(R(i)) / max(R(i)) */
        float rcmin_clamped = rcmin > smlnum ? rcmin : smlnum;
        float rcmax_clamped = rcmax < bignum ? rcmax : bignum;
        *rowcnd = rcmin_clamped / rcmax_clamped;
    }

    /* Compute column scale factors */
    for (j = 0; j < n; j++) {
        C[j] = ZERO;
    }

    /* Find the maximum element in each column,
     * assuming the row scaling computed above */
    for (j = 0; j < n; j++) {
        int i_start = (j - ku > 0) ? j - ku : 0;
        int i_end = (j + kl < m - 1) ? j + kl : m - 1;
        for (i = i_start; i <= i_end; i++) {
            float scaled_val = fabsf(AB[ku + i - j + j * ldab]) * R[i];
            if (scaled_val > C[j]) {
                C[j] = scaled_val;
            }
        }
        /* Round to power of radix */
        if (C[j] > ZERO) {
            C[j] = powf(radix, (int)(logf(C[j]) / logrdx));
        }
    }

    /* Find the maximum and minimum scale factors */
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
        /* Find the first zero scale factor and return an error code */
        for (j = 0; j < n; j++) {
            if (C[j] == ZERO) {
                *info = m + j + 1;  /* 1-based index for error reporting */
                return;
            }
        }
    } else {
        /* Invert the scale factors */
        for (j = 0; j < n; j++) {
            float cj = C[j];
            if (cj < smlnum) {
                cj = smlnum;
            }
            if (cj > bignum) {
                cj = bignum;
            }
            C[j] = ONE / cj;
        }

        /* Compute colcnd = min(C(j)) / max(C(j)) */
        float rcmin_clamped = rcmin > smlnum ? rcmin : smlnum;
        float rcmax_clamped = rcmax < bignum ? rcmax : bignum;
        *colcnd = rcmin_clamped / rcmax_clamped;
    }
}
