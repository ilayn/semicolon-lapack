/**
 * @file slansp.c
 * @brief SLANSP returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the element of largest absolute value of a symmetric matrix supplied in packed form.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLANSP returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real symmetric matrix A, supplied in packed form.
 *
 * SLANSP = ( max(abs(A(i,j))), NORM = 'M' or 'm'
 *          (
 *          ( norm1(A),         NORM = '1', 'O' or 'o'
 *          (
 *          ( normI(A),         NORM = 'I' or 'i'
 *          (
 *          ( normF(A),         NORM = 'F', 'f', 'E' or 'e'
 *
 * where norm1 denotes the one norm of a matrix (maximum column sum),
 * normI denotes the infinity norm of a matrix (maximum row sum) and
 * normF denotes the Frobenius norm of a matrix (square root of sum of
 * squares). Note that max(abs(A(i,j))) is not a consistent matrix norm.
 *
 * @param[in]     norm   Specifies the value to be returned:
 *                       = 'M' or 'm': max(abs(A(i,j)))
 *                       = '1', 'O' or 'o': norm1(A)
 *                       = 'I' or 'i': normI(A)
 *                       = 'F', 'f', 'E' or 'e': normF(A)
 * @param[in]     uplo   = 'U': Upper triangular part of A is supplied
 *                       = 'L': Lower triangular part of A is supplied
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     AP     The upper or lower triangle of the symmetric matrix A,
 *                       packed columnwise in a linear array.
 *                       Array of dimension (n*(n+1)/2).
 * @param[out]    work   Workspace array of dimension (max(1,lwork)),
 *                       where lwork >= n when norm = 'I' or '1' or 'O';
 *                       otherwise, work is not referenced.
 *
 * @return The computed norm value.
 */
f32 slansp(
    const char* norm,
    const char* uplo,
    const INT n,
    const f32* restrict AP,
    f32* restrict work)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    INT i, j, k;
    f32 absa, scale, sum, value;

    if (n == 0) {
        value = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        // Find max(abs(A(i,j)))
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            k = 0;
            for (j = 0; j < n; j++) {
                for (i = k; i < k + j + 1; i++) {
                    sum = fabsf(AP[i]);
                    if (value < sum || sisnan(sum)) {
                        value = sum;
                    }
                }
                k = k + (j + 1);
            }
        } else {
            k = 0;
            for (j = 0; j < n; j++) {
                for (i = k; i < k + n - j; i++) {
                    sum = fabsf(AP[i]);
                    if (value < sum || sisnan(sum)) {
                        value = sum;
                    }
                }
                k = k + (n - j);
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' ||
               norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        // Find normI(A) ( = norm1(A), since A is symmetric)
        value = ZERO;
        k = 0;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                sum = ZERO;
                for (i = 0; i < j; i++) {
                    absa = fabsf(AP[k]);
                    sum = sum + absa;
                    work[i] = work[i] + absa;
                    k = k + 1;
                }
                work[j] = sum + fabsf(AP[k]);
                k = k + 1;
            }
            for (i = 0; i < n; i++) {
                sum = work[i];
                if (value < sum || sisnan(sum)) {
                    value = sum;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                work[i] = ZERO;
            }
            for (j = 0; j < n; j++) {
                sum = work[j] + fabsf(AP[k]);
                k = k + 1;
                for (i = j + 1; i < n; i++) {
                    absa = fabsf(AP[k]);
                    sum = sum + absa;
                    work[i] = work[i] + absa;
                    k = k + 1;
                }
                if (value < sum || sisnan(sum)) {
                    value = sum;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        // Find normF(A)
        scale = ZERO;
        sum = ONE;
        k = 1;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 1; j < n; j++) {
                slassq(j, &AP[k], 1, &scale, &sum);
                k = k + (j + 1);
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                slassq(n - j - 1, &AP[k], 1, &scale, &sum);
                k = k + (n - j);
            }
        }
        sum = 2 * sum;
        k = 0;
        for (i = 0; i < n; i++) {
            if (AP[k] != ZERO) {
                absa = fabsf(AP[k]);
                if (scale < absa) {
                    sum = ONE + sum * (scale / absa) * (scale / absa);
                    scale = absa;
                } else {
                    sum = sum + (absa / scale) * (absa / scale);
                }
            }
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                k = k + (i + 2);
            } else {
                k = k + (n - i);
            }
        }
        value = scale * sqrtf(sum);
    } else {
        value = ZERO;
    }

    return value;
}
