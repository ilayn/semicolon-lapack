/**
 * @file clange.c
 * @brief CLANGE returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element of
 *        a general rectangular matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLANGE returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (max column sum)
 *                  = 'I' or 'i': normI(A) (max row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] m     The number of rows of the matrix A. m >= 0.
 *                  When m = 0, clange returns zero.
 * @param[in] n     The number of columns of the matrix A. n >= 0.
 *                  When n = 0, clange returns zero.
 * @param[in] A     Complex*16 array, dimension (lda, n).
 *                  The m by n matrix A.
 * @param[in] lda   The leading dimension of the array A. lda >= max(m, 1).
 * @param[out] work Single precision array, dimension (max(1, lwork)).
 *                  where lwork >= m when norm = 'I'; otherwise, work is
 *                  not referenced.
 *
 * @return The computed norm value.
 */
f32 clange(
    const char* norm,
    const int m,
    const int n,
    const c64* restrict A,
    const int lda,
    f32* restrict work)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int i, j;
    f32 scale, sum, value, temp;

    /* Quick return if possible */
    int minmn = (m < n) ? m : n;
    if (minmn == 0) {
        return ZERO;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        value = ZERO;
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                temp = cabsf(A[i + j * lda]);
                if (value < temp || isnan(temp)) {
                    value = temp;
                }
            }
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find norm1(A) - maximum column sum */
        value = ZERO;
        for (j = 0; j < n; j++) {
            sum = ZERO;
            for (i = 0; i < m; i++) {
                sum += cabsf(A[i + j * lda]);
            }
            if (value < sum || isnan(sum)) {
                value = sum;
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        /* Find normI(A) - maximum row sum */
        for (i = 0; i < m; i++) {
            work[i] = ZERO;
        }
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                work[i] += cabsf(A[i + j * lda]);
            }
        }
        value = ZERO;
        for (i = 0; i < m; i++) {
            temp = work[i];
            if (value < temp || isnan(temp)) {
                value = temp;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) - Frobenius norm using classq */
        scale = ZERO;
        sum = ONE;
        for (j = 0; j < n; j++) {
            classq(m, &A[j * lda], 1, &scale, &sum);
        }
        value = scale * sqrtf(sum);
    } else {
        value = ZERO;
    }

    return value;
}
