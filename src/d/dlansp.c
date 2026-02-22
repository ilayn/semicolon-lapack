/**
 * @file dlansp.c
 * @brief DLANSP returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the element of largest absolute value of a symmetric matrix supplied in packed form.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLANSP returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real symmetric matrix A, supplied in packed form.
 *
 * DLANSP = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
f64 dlansp(
    const char* norm,
    const char* uplo,
    const INT n,
    const f64* restrict AP,
    f64* restrict work)
{
    // dlansp.f lines 129-130: Parameters
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    // dlansp.f lines 133-134: Local Scalars
    INT i, j, k;
    f64 absa, scale, sum, value;

    // dlansp.f lines 148-252: Main logic
    if (n == 0) {
        // dlansp.f line 149
        value = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        // dlansp.f lines 152-173: Find max(abs(A(i,j)))
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // dlansp.f lines 156-163
            k = 0;  // dlansp.f line 156: K = 1 (0-based: k = 0)
            for (j = 0; j < n; j++) {  // dlansp.f line 157: DO 20 J = 1, N
                for (i = k; i < k + j + 1; i++) {  // dlansp.f line 158: DO 10 I = K, K + J - 1
                    sum = fabs(AP[i]);  // dlansp.f line 159
                    if (value < sum || disnan(sum)) {  // dlansp.f line 160
                        value = sum;
                    }
                }
                k = k + (j + 1);  // dlansp.f line 162: K = K + J
            }
        } else {
            // dlansp.f lines 165-172
            k = 0;  // dlansp.f line 165: K = 1 (0-based: k = 0)
            for (j = 0; j < n; j++) {  // dlansp.f line 166: DO 40 J = 1, N
                for (i = k; i < k + n - j; i++) {  // dlansp.f line 167: DO 30 I = K, K + N - J
                    sum = fabs(AP[i]);  // dlansp.f line 168
                    if (value < sum || disnan(sum)) {  // dlansp.f line 169
                        value = sum;
                    }
                }
                k = k + (n - j);  // dlansp.f line 171: K = K + N - J + 1
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' ||
               norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        // dlansp.f lines 178-213: Find normI(A) (= norm1(A), since A is symmetric)
        value = ZERO;
        k = 0;  // dlansp.f line 181: K = 1 (0-based: k = 0)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // dlansp.f lines 183-197
            for (j = 0; j < n; j++) {  // dlansp.f line 183: DO 60 J = 1, N
                sum = ZERO;  // dlansp.f line 184
                for (i = 0; i < j; i++) {  // dlansp.f line 185: DO 50 I = 1, J - 1
                    absa = fabs(AP[k]);  // dlansp.f line 186
                    sum = sum + absa;  // dlansp.f line 187
                    work[i] = work[i] + absa;  // dlansp.f line 188
                    k = k + 1;  // dlansp.f line 189
                }
                work[j] = sum + fabs(AP[k]);  // dlansp.f line 191
                k = k + 1;  // dlansp.f line 192
            }
            for (i = 0; i < n; i++) {  // dlansp.f line 194: DO 70 I = 1, N
                sum = work[i];  // dlansp.f line 195
                if (value < sum || disnan(sum)) {  // dlansp.f line 196
                    value = sum;
                }
            }
        } else {
            // dlansp.f lines 199-212
            for (i = 0; i < n; i++) {  // dlansp.f line 199: DO 80 I = 1, N
                work[i] = ZERO;  // dlansp.f line 200
            }
            for (j = 0; j < n; j++) {  // dlansp.f line 202: DO 100 J = 1, N
                sum = work[j] + fabs(AP[k]);  // dlansp.f line 203
                k = k + 1;  // dlansp.f line 204
                for (i = j + 1; i < n; i++) {  // dlansp.f line 205: DO 90 I = J + 1, N
                    absa = fabs(AP[k]);  // dlansp.f line 206
                    sum = sum + absa;  // dlansp.f line 207
                    work[i] = work[i] + absa;  // dlansp.f line 208
                    k = k + 1;  // dlansp.f line 209
                }
                if (value < sum || disnan(sum)) {  // dlansp.f line 211
                    value = sum;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        // dlansp.f lines 217-251: Find normF(A)
        scale = ZERO;  // dlansp.f line 219
        sum = ONE;  // dlansp.f line 220
        k = 1;  // dlansp.f line 221: K = 2 (0-based: k = 1)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // dlansp.f lines 223-226
            for (j = 1; j < n; j++) {  // dlansp.f line 223: DO 110 J = 2, N
                dlassq(j, &AP[k], 1, &scale, &sum);  // dlansp.f line 224: CALL DLASSQ( J-1, AP( K ), 1, SCALE, SUM )
                k = k + (j + 1);  // dlansp.f line 225: K = K + J
            }
        } else {
            // dlansp.f lines 228-231
            for (j = 0; j < n - 1; j++) {  // dlansp.f line 228: DO 120 J = 1, N - 1
                dlassq(n - j - 1, &AP[k], 1, &scale, &sum);  // dlansp.f line 229: CALL DLASSQ( N-J, AP( K ), 1, SCALE, SUM )
                k = k + (n - j);  // dlansp.f line 230: K = K + N - J + 1
            }
        }
        sum = 2 * sum;  // dlansp.f line 233
        k = 0;  // dlansp.f line 234: K = 1 (0-based: k = 0)
        for (i = 0; i < n; i++) {  // dlansp.f line 235: DO 130 I = 1, N
            if (AP[k] != ZERO) {  // dlansp.f line 236
                absa = fabs(AP[k]);  // dlansp.f line 237
                if (scale < absa) {  // dlansp.f line 238
                    sum = ONE + sum * (scale / absa) * (scale / absa);  // dlansp.f line 239
                    scale = absa;  // dlansp.f line 240
                } else {
                    sum = sum + (absa / scale) * (absa / scale);  // dlansp.f line 242
                }
            }
            // dlansp.f lines 245-249
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                k = k + (i + 2);  // dlansp.f line 246: K = K + I + 1
            } else {
                k = k + (n - i);  // dlansp.f line 248: K = K + N - I + 1
            }
        }
        value = scale * sqrt(sum);  // dlansp.f line 251
    } else {
        value = ZERO;  // Default case (should not happen with valid input)
    }

    return value;  // dlansp.f line 254
}
