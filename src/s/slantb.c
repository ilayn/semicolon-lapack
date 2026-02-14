/**
 * @file slantb.c
 * @brief SLANTB returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element of
 *        a triangular band matrix.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLANTB returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of an
 * n by n triangular band matrix A, with (k + 1) diagonals.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (max column sum)
 *                  = 'I' or 'i': normI(A) (max row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] uplo  Specifies whether the matrix A is upper or lower triangular.
 *                  = 'U': Upper triangular
 *                  = 'L': Lower triangular
 * @param[in] diag  Specifies whether or not the matrix A is unit triangular.
 *                  = 'N': Non-unit triangular
 *                  = 'U': Unit triangular
 * @param[in] n     The order of the matrix A. n >= 0.
 *                  When n = 0, slantb returns zero.
 * @param[in] k     The number of super-diagonals of the matrix A if uplo = "U",
 *                  or the number of sub-diagonals of the matrix A if uplo = 'L'.
 *                  k >= 0.
 * @param[in] AB    Double precision array, dimension (ldab, n).
 *                  The upper or lower triangular band matrix A, stored in the
 *                  first k+1 rows of AB. The j-th column of A is stored
 *                  in the j-th column of the array AB as follows:
 *                  if uplo = "U", AB[k+i-j + j*ldab] = A(i,j) for max(0,j-k) <= i <= j;
 *                  if uplo = "L", AB[i-j + j*ldab]   = A(i,j) for j <= i <= min(n-1,j+k).
 *                  Note that when diag = "U", the elements of the array AB
 *                  corresponding to the diagonal elements of the matrix A are
 *                  not referenced, but are assumed to be one.
 * @param[in] ldab  The leading dimension of the array AB. ldab >= k+1.
 * @param[out] work Double precision array, dimension (max(1, lwork)).
 *                  where lwork >= n when norm = 'I'; otherwise, work is not
 *                  referenced.
 *
 * @return The computed norm value.
 */
f32 slantb(
    const char* norm,
    const char* uplo,
    const char* diag,
    const int n,
    const int k,
    const f32 * const restrict AB,
    const int ldab,
    f32 * const restrict work)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int i, j, l;
    f32 scale, sum, value;
    int udiag;

    /* Quick return if possible */
    if (n == 0) {
        return ZERO;
    }

    udiag = (diag[0] == 'U' || diag[0] == 'u');

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        if (udiag) {
            value = ONE;
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                for (j = 0; j < n; j++) {
                    int i_start = (k - j > 0) ? k - j : 0;
                    for (i = i_start; i < k; i++) {
                        sum = fabsf(AB[i + j * ldab]);
                        if (value < sum || isnan(sum)) {
                            value = sum;
                        }
                    }
                }
            } else {
                for (j = 0; j < n; j++) {
                    int i_end = (n - j < k + 1) ? n - j : k + 1;
                    for (i = 1; i < i_end; i++) {
                        sum = fabsf(AB[i + j * ldab]);
                        if (value < sum || isnan(sum)) {
                            value = sum;
                        }
                    }
                }
            }
        } else {
            value = ZERO;
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                for (j = 0; j < n; j++) {
                    int i_start = (k - j > 0) ? k - j : 0;
                    for (i = i_start; i <= k; i++) {
                        sum = fabsf(AB[i + j * ldab]);
                        if (value < sum || isnan(sum)) {
                            value = sum;
                        }
                    }
                }
            } else {
                for (j = 0; j < n; j++) {
                    int i_end = (n - j < k + 1) ? n - j : k + 1;
                    for (i = 0; i < i_end; i++) {
                        sum = fabsf(AB[i + j * ldab]);
                        if (value < sum || isnan(sum)) {
                            value = sum;
                        }
                    }
                }
            }
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find norm1(A) */
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                if (udiag) {
                    sum = ONE;
                    int i_start = (k - j > 0) ? k - j : 0;
                    for (i = i_start; i < k; i++) {
                        sum += fabsf(AB[i + j * ldab]);
                    }
                } else {
                    sum = ZERO;
                    int i_start = (k - j > 0) ? k - j : 0;
                    for (i = i_start; i <= k; i++) {
                        sum += fabsf(AB[i + j * ldab]);
                    }
                }
                if (value < sum || isnan(sum)) {
                    value = sum;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                if (udiag) {
                    sum = ONE;
                    int i_end = (n - j < k + 1) ? n - j : k + 1;
                    for (i = 1; i < i_end; i++) {
                        sum += fabsf(AB[i + j * ldab]);
                    }
                } else {
                    sum = ZERO;
                    int i_end = (n - j < k + 1) ? n - j : k + 1;
                    for (i = 0; i < i_end; i++) {
                        sum += fabsf(AB[i + j * ldab]);
                    }
                }
                if (value < sum || isnan(sum)) {
                    value = sum;
                }
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        /* Find normI(A) */
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            if (udiag) {
                for (i = 0; i < n; i++) {
                    work[i] = ONE;
                }
                for (j = 0; j < n; j++) {
                    l = k - j;
                    int row_start = (j - k > 0) ? j - k : 0;
                    for (i = row_start; i < j; i++) {
                        work[i] += fabsf(AB[l + i + j * ldab]);
                    }
                }
            } else {
                for (i = 0; i < n; i++) {
                    work[i] = ZERO;
                }
                for (j = 0; j < n; j++) {
                    l = k - j;
                    int row_start = (j - k > 0) ? j - k : 0;
                    for (i = row_start; i <= j; i++) {
                        work[i] += fabsf(AB[l + i + j * ldab]);
                    }
                }
            }
        } else {
            if (udiag) {
                for (i = 0; i < n; i++) {
                    work[i] = ONE;
                }
                for (j = 0; j < n; j++) {
                    l = -j;
                    int row_end = (j + k < n - 1) ? j + k : n - 1;
                    for (i = j + 1; i <= row_end; i++) {
                        work[i] += fabsf(AB[l + i + j * ldab]);
                    }
                }
            } else {
                for (i = 0; i < n; i++) {
                    work[i] = ZERO;
                }
                for (j = 0; j < n; j++) {
                    l = -j;
                    int row_end = (j + k < n - 1) ? j + k : n - 1;
                    for (i = j; i <= row_end; i++) {
                        work[i] += fabsf(AB[l + i + j * ldab]);
                    }
                }
            }
        }
        for (i = 0; i < n; i++) {
            sum = work[i];
            if (value < sum || isnan(sum)) {
                value = sum;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) */
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            if (udiag) {
                scale = ONE;
                sum = (f32)n;
                if (k > 0) {
                    for (j = 1; j < n; j++) {
                        int count = (j < k) ? j : k;
                        int start = (k - j > 0) ? k - j : 0;
                        slassq(count, &AB[start + j * ldab], 1, &scale, &sum);
                    }
                }
            } else {
                scale = ZERO;
                sum = ONE;
                for (j = 0; j < n; j++) {
                    int count = (j + 1 < k + 1) ? j + 1 : k + 1;
                    int start = (k - j > 0) ? k - j : 0;
                    slassq(count, &AB[start + j * ldab], 1, &scale, &sum);
                }
            }
        } else {
            if (udiag) {
                scale = ONE;
                sum = (f32)n;
                if (k > 0) {
                    for (j = 0; j < n - 1; j++) {
                        int count = (n - 1 - j < k) ? n - 1 - j : k;
                        slassq(count, &AB[1 + j * ldab], 1, &scale, &sum);
                    }
                }
            } else {
                scale = ZERO;
                sum = ONE;
                for (j = 0; j < n; j++) {
                    int count = (n - j < k + 1) ? n - j : k + 1;
                    slassq(count, &AB[j * ldab], 1, &scale, &sum);
                }
            }
        }
        value = scale * sqrtf(sum);
    } else {
        value = ZERO;
    }

    return value;
}
