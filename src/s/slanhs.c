/**
 * @file slanhs.c
 * @brief SLANHS returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element
 *        of an upper Hessenberg matrix.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLANHS returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * Hessenberg matrix A.
 *
 * @param[in] norm    Specifies the value to be returned:
 *                    = 'M' or 'm': max(abs(A(i,j)))
 *                    = '1', 'O' or 'o': norm1(A) (max column sum)
 *                    = 'I' or 'i': normI(A) (max row sum)
 *                    = 'F', 'f', 'E' or 'e': normF(A) (Frobenius norm)
 * @param[in] n       The order of the matrix A. n >= 0. When n = 0,
 *                    slanhs returns zero.
 * @param[in] A       Double precision array, dimension (lda, n).
 *                    The n by n upper Hessenberg matrix A; the part of A
 *                    below the first sub-diagonal is not referenced.
 * @param[in] lda     The leading dimension of the array A. lda >= max(n, 1).
 * @param[out] work   Double precision array, dimension (max(1, lwork)).
 *                    where lwork >= n when norm = 'I'; otherwise, work is
 *                    not referenced.
 *
 * @return The computed norm value.
 */
float slanhs(
    const char* norm,
    const int n,
    const float* const restrict A,
    const int lda,
    float* const restrict work)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int i, j;
    float scale, sum, value, temp;
    int jmax;

    if (n == 0) {
        return ZERO;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        value = ZERO;
        for (j = 0; j < n; j++) {
            /* For Hessenberg matrix, only access rows 0 to min(n-1, j+1) */
            jmax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
            for (i = 0; i <= jmax; i++) {
                temp = fabsf(A[i + j * lda]);
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
            /* For Hessenberg matrix, only access rows 0 to min(n-1, j+1) */
            jmax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
            for (i = 0; i <= jmax; i++) {
                sum += fabsf(A[i + j * lda]);
            }
            if (value < sum || isnan(sum)) {
                value = sum;
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        /* Find normI(A) - maximum row sum */
        for (i = 0; i < n; i++) {
            work[i] = ZERO;
        }
        for (j = 0; j < n; j++) {
            /* For Hessenberg matrix, only access rows 0 to min(n-1, j+1) */
            jmax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
            for (i = 0; i <= jmax; i++) {
                work[i] += fabsf(A[i + j * lda]);
            }
        }
        value = ZERO;
        for (i = 0; i < n; i++) {
            temp = work[i];
            if (value < temp || isnan(temp)) {
                value = temp;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) - Frobenius norm using slassq */
        scale = ZERO;
        sum = ONE;
        for (j = 0; j < n; j++) {
            /* For Hessenberg matrix, only access rows 0 to min(n-1, j+1) */
            jmax = (j + 1 < n - 1) ? (j + 1) : (n - 1);
            /* Number of elements in this column: jmax + 1 */
            slassq(jmax + 1, &A[j * lda], 1, &scale, &sum);
        }
        value = scale * sqrtf(sum);
    } else {
        value = ZERO;
    }

    return value;
}
