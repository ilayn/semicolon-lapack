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
    const int n,
    const f32* const restrict AP,
    f32* const restrict work)
{
    // slansp.f lines 129-130: Parameters
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    // slansp.f lines 133-134: Local Scalars
    int i, j, k;
    f32 absa, scale, sum, value;

    // slansp.f lines 148-252: Main logic
    if (n == 0) {
        // slansp.f line 149
        value = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        // slansp.f lines 152-173: Find max(abs(A(i,j)))
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // slansp.f lines 156-163
            k = 0;  // slansp.f line 156: K = 1 (0-based: k = 0)
            for (j = 0; j < n; j++) {  // slansp.f line 157: DO 20 J = 1, N
                for (i = k; i < k + j + 1; i++) {  // slansp.f line 158: DO 10 I = K, K + J - 1
                    sum = fabsf(AP[i]);  // slansp.f line 159
                    if (value < sum || sisnan(sum)) {  // slansp.f line 160
                        value = sum;
                    }
                }
                k = k + (j + 1);  // slansp.f line 162: K = K + J
            }
        } else {
            // slansp.f lines 165-172
            k = 0;  // slansp.f line 165: K = 1 (0-based: k = 0)
            for (j = 0; j < n; j++) {  // slansp.f line 166: DO 40 J = 1, N
                for (i = k; i < k + n - j; i++) {  // slansp.f line 167: DO 30 I = K, K + N - J
                    sum = fabsf(AP[i]);  // slansp.f line 168
                    if (value < sum || sisnan(sum)) {  // slansp.f line 169
                        value = sum;
                    }
                }
                k = k + (n - j);  // slansp.f line 171: K = K + N - J + 1
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' ||
               norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        // slansp.f lines 178-213: Find normI(A) (= norm1(A), since A is symmetric)
        value = ZERO;
        k = 0;  // slansp.f line 181: K = 1 (0-based: k = 0)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // slansp.f lines 183-197
            for (j = 0; j < n; j++) {  // slansp.f line 183: DO 60 J = 1, N
                sum = ZERO;  // slansp.f line 184
                for (i = 0; i < j; i++) {  // slansp.f line 185: DO 50 I = 1, J - 1
                    absa = fabsf(AP[k]);  // slansp.f line 186
                    sum = sum + absa;  // slansp.f line 187
                    work[i] = work[i] + absa;  // slansp.f line 188
                    k = k + 1;  // slansp.f line 189
                }
                work[j] = sum + fabsf(AP[k]);  // slansp.f line 191
                k = k + 1;  // slansp.f line 192
            }
            for (i = 0; i < n; i++) {  // slansp.f line 194: DO 70 I = 1, N
                sum = work[i];  // slansp.f line 195
                if (value < sum || sisnan(sum)) {  // slansp.f line 196
                    value = sum;
                }
            }
        } else {
            // slansp.f lines 199-212
            for (i = 0; i < n; i++) {  // slansp.f line 199: DO 80 I = 1, N
                work[i] = ZERO;  // slansp.f line 200
            }
            for (j = 0; j < n; j++) {  // slansp.f line 202: DO 100 J = 1, N
                sum = work[j] + fabsf(AP[k]);  // slansp.f line 203
                k = k + 1;  // slansp.f line 204
                for (i = j + 1; i < n; i++) {  // slansp.f line 205: DO 90 I = J + 1, N
                    absa = fabsf(AP[k]);  // slansp.f line 206
                    sum = sum + absa;  // slansp.f line 207
                    work[i] = work[i] + absa;  // slansp.f line 208
                    k = k + 1;  // slansp.f line 209
                }
                if (value < sum || sisnan(sum)) {  // slansp.f line 211
                    value = sum;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        // slansp.f lines 217-251: Find normF(A)
        scale = ZERO;  // slansp.f line 219
        sum = ONE;  // slansp.f line 220
        k = 1;  // slansp.f line 221: K = 2 (0-based: k = 1)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // slansp.f lines 223-226
            for (j = 1; j < n; j++) {  // slansp.f line 223: DO 110 J = 2, N
                slassq(j, &AP[k], 1, &scale, &sum);  // slansp.f line 224: CALL SLASSQ( J-1, AP( K ), 1, SCALE, SUM )
                k = k + (j + 1);  // slansp.f line 225: K = K + J
            }
        } else {
            // slansp.f lines 228-231
            for (j = 0; j < n - 1; j++) {  // slansp.f line 228: DO 120 J = 1, N - 1
                slassq(n - j - 1, &AP[k], 1, &scale, &sum);  // slansp.f line 229: CALL SLASSQ( N-J, AP( K ), 1, SCALE, SUM )
                k = k + (n - j);  // slansp.f line 230: K = K + N - J + 1
            }
        }
        sum = 2 * sum;  // slansp.f line 233
        k = 0;  // slansp.f line 234: K = 1 (0-based: k = 0)
        for (i = 0; i < n; i++) {  // slansp.f line 235: DO 130 I = 1, N
            if (AP[k] != ZERO) {  // slansp.f line 236
                absa = fabsf(AP[k]);  // slansp.f line 237
                if (scale < absa) {  // slansp.f line 238
                    sum = ONE + sum * (scale / absa) * (scale / absa);  // slansp.f line 239
                    scale = absa;  // slansp.f line 240
                } else {
                    sum = sum + (absa / scale) * (absa / scale);  // slansp.f line 242
                }
            }
            // slansp.f lines 245-249
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                k = k + (i + 2);  // slansp.f line 246: K = K + I + 1
            } else {
                k = k + (n - i);  // slansp.f line 248: K = K + N - I + 1
            }
        }
        value = scale * sqrtf(sum);  // slansp.f line 251
    } else {
        value = ZERO;  // Default case (should not happen with valid input)
    }

    return value;  // slansp.f line 254
}
