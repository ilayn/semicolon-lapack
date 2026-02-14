/**
 * @file zlansp.c
 * @brief ZLANSP returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the element of largest absolute value of a symmetric matrix supplied in packed form.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLANSP returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex symmetric matrix A, supplied in packed form.
 *
 * ZLANSP = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
f64 zlansp(
    const char* norm,
    const char* uplo,
    const int n,
    const c128* const restrict AP,
    f64* const restrict work)
{
    // zlansp.f lines 131-132: Parameters
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    // zlansp.f lines 135-136: Local Scalars
    int i, j, k;
    f64 absa, scale, sum, value;

    // zlansp.f lines 150-263: Main logic
    if (n == 0) {
        // zlansp.f line 150
        value = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        // zlansp.f lines 152-175: Find max(abs(A(i,j)))
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // zlansp.f lines 157-164
            k = 0;  // zlansp.f line 158: K = 1 (0-based: k = 0)
            for (j = 0; j < n; j++) {  // zlansp.f line 159: DO 20 J = 1, N
                for (i = k; i < k + j + 1; i++) {  // zlansp.f line 160: DO 10 I = K, K + J - 1
                    sum = cabs(AP[i]);  // zlansp.f line 161
                    if (value < sum || disnan(sum)) {  // zlansp.f line 162
                        value = sum;
                    }
                }
                k = k + (j + 1);  // zlansp.f line 164: K = K + J
            }
        } else {
            // zlansp.f lines 166-174
            k = 0;  // zlansp.f line 167: K = 1 (0-based: k = 0)
            for (j = 0; j < n; j++) {  // zlansp.f line 168: DO 40 J = 1, N
                for (i = k; i < k + n - j; i++) {  // zlansp.f line 169: DO 30 I = K, K + N - J
                    sum = cabs(AP[i]);  // zlansp.f line 170
                    if (value < sum || disnan(sum)) {  // zlansp.f line 171
                        value = sum;
                    }
                }
                k = k + (n - j);  // zlansp.f line 173: K = K + N - J + 1
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' ||
               norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        // zlansp.f lines 180-215: Find normI(A) (= norm1(A), since A is symmetric)
        value = ZERO;
        k = 0;  // zlansp.f line 183: K = 1 (0-based: k = 0)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // zlansp.f lines 184-199
            for (j = 0; j < n; j++) {  // zlansp.f line 185: DO 60 J = 1, N
                sum = ZERO;  // zlansp.f line 186
                for (i = 0; i < j; i++) {  // zlansp.f line 187: DO 50 I = 1, J - 1
                    absa = cabs(AP[k]);  // zlansp.f line 188
                    sum = sum + absa;  // zlansp.f line 189
                    work[i] = work[i] + absa;  // zlansp.f line 190
                    k = k + 1;  // zlansp.f line 191
                }
                work[j] = sum + cabs(AP[k]);  // zlansp.f line 193
                k = k + 1;  // zlansp.f line 194
            }
            for (i = 0; i < n; i++) {  // zlansp.f line 196: DO 70 I = 1, N
                sum = work[i];  // zlansp.f line 197
                if (value < sum || disnan(sum)) {  // zlansp.f line 198
                    value = sum;
                }
            }
        } else {
            // zlansp.f lines 200-215
            for (i = 0; i < n; i++) {  // zlansp.f line 201: DO 80 I = 1, N
                work[i] = ZERO;  // zlansp.f line 202
            }
            for (j = 0; j < n; j++) {  // zlansp.f line 204: DO 100 J = 1, N
                sum = work[j] + cabs(AP[k]);  // zlansp.f line 205
                k = k + 1;  // zlansp.f line 206
                for (i = j + 1; i < n; i++) {  // zlansp.f line 207: DO 90 I = J + 1, N
                    absa = cabs(AP[k]);  // zlansp.f line 208
                    sum = sum + absa;  // zlansp.f line 209
                    work[i] = work[i] + absa;  // zlansp.f line 210
                    k = k + 1;  // zlansp.f line 211
                }
                if (value < sum || disnan(sum)) {  // zlansp.f line 213
                    value = sum;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        // zlansp.f lines 217-262: Find normF(A)
        scale = ZERO;  // zlansp.f line 221
        sum = ONE;  // zlansp.f line 222
        k = 1;  // zlansp.f line 223: K = 2 (0-based: k = 1)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            // zlansp.f lines 224-228
            for (j = 1; j < n; j++) {  // zlansp.f line 225: DO 110 J = 2, N
                zlassq(j, &AP[k], 1, &scale, &sum);  // zlansp.f line 226: CALL ZLASSQ( J-1, AP( K ), 1, SCALE, SUM )
                k = k + (j + 1);  // zlansp.f line 227: K = K + J
            }
        } else {
            // zlansp.f lines 229-233
            for (j = 0; j < n - 1; j++) {  // zlansp.f line 230: DO 120 J = 1, N - 1
                zlassq(n - j - 1, &AP[k], 1, &scale, &sum);  // zlansp.f line 231: CALL ZLASSQ( N-J, AP( K ), 1, SCALE, SUM )
                k = k + (n - j);  // zlansp.f line 232: K = K + N - J + 1
            }
        }
        sum = 2 * sum;  // zlansp.f line 235
        k = 0;  // zlansp.f line 236: K = 1 (0-based: k = 0)
        for (i = 0; i < n; i++) {  // zlansp.f line 237: DO 130 I = 1, N
            if (creal(AP[k]) != ZERO) {  // zlansp.f line 238: IF( DBLE( AP( K ) ).NE.ZERO )
                absa = fabs(creal(AP[k]));  // zlansp.f line 239: ABSA = ABS( DBLE( AP( K ) ) )
                if (scale < absa) {  // zlansp.f line 240
                    sum = ONE + sum * (scale / absa) * (scale / absa);  // zlansp.f line 241
                    scale = absa;  // zlansp.f line 242
                } else {
                    sum = sum + (absa / scale) * (absa / scale);  // zlansp.f line 244
                }
            }
            if (cimag(AP[k]) != ZERO) {  // zlansp.f line 247: IF( DIMAG( AP( K ) ).NE.ZERO )
                absa = fabs(cimag(AP[k]));  // zlansp.f line 248: ABSA = ABS( DIMAG( AP( K ) ) )
                if (scale < absa) {  // zlansp.f line 249
                    sum = ONE + sum * (scale / absa) * (scale / absa);  // zlansp.f line 250
                    scale = absa;  // zlansp.f line 251
                } else {
                    sum = sum + (absa / scale) * (absa / scale);  // zlansp.f line 253
                }
            }
            // zlansp.f lines 256-260
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                k = k + (i + 2);  // zlansp.f line 257: K = K + I + 1
            } else {
                k = k + (n - i);  // zlansp.f line 259: K = K + N - I + 1
            }
        }
        value = scale * sqrt(sum);  // zlansp.f line 262
    } else {
        value = ZERO;  // Default case (should not happen with valid input)
    }

    return value;  // zlansp.f line 265
}
