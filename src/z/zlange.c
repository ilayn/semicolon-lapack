/**
 * @file zlange.c
 * @brief ZLANGE returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element of
 *        a general rectangular matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLANGE returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (max column sum)
 *                  = 'I' or 'i': normI(A) (max row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] m     The number of rows of the matrix A. m >= 0.
 *                  When m = 0, zlange returns zero.
 * @param[in] n     The number of columns of the matrix A. n >= 0.
 *                  When n = 0, zlange returns zero.
 * @param[in] A     Complex*16 array, dimension (lda, n).
 *                  The m by n matrix A.
 * @param[in] lda   The leading dimension of the array A. lda >= max(m, 1).
 * @param[out] work Double precision array, dimension (max(1, lwork)).
 *                  where lwork >= m when norm = 'I'; otherwise, work is
 *                  not referenced.
 *
 * @return The computed norm value.
 */
f64 zlange(
    const char* norm,
    const int m,
    const int n,
    const c128* const restrict A,
    const int lda,
    f64* const restrict work)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i, j;
    f64 scale, sum, value, temp;

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
                temp = cabs(A[i + j * lda]);
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
                sum += cabs(A[i + j * lda]);
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
                work[i] += cabs(A[i + j * lda]);
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
        /* Find normF(A) - Frobenius norm using zlassq */
        scale = ZERO;
        sum = ONE;
        for (j = 0; j < n; j++) {
            zlassq(m, &A[j * lda], 1, &scale, &sum);
        }
        value = scale * sqrt(sum);
    } else {
        value = ZERO;
    }

    return value;
}
