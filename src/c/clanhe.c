/**
 * @file clanhe.c
 * @brief CLANHE returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element of
 *        a complex Hermitian matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CLANHE returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex hermitian matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (max column sum)
 *                  = 'I' or 'i': normI(A) (max row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 *                  Note: norm1(A) = normI(A) since A is hermitian.
 * @param[in] uplo  Specifies whether the upper or lower triangular part
 *                  of A is referenced:
 *                  = 'U': Upper triangular part of A is referenced
 *                  = 'L': Lower triangular part of A is referenced
 * @param[in] n     The order of the matrix A. n >= 0.
 *                  When n = 0, clanhe returns zero.
 * @param[in] A     Single complex array, dimension (lda, n).
 *                  The hermitian matrix A. Note that the imaginary parts
 *                  of the diagonal elements need not be set and are
 *                  assumed to be zero.
 * @param[in] lda   The leading dimension of A. lda >= max(n, 1).
 * @param[out] work Single precision array, dimension (max(1, lwork)).
 *                  where lwork >= n when norm = 'I' or '1' or 'O';
 *                  otherwise, work is not referenced.
 *
 * @return The computed norm value.
 */
f32 clanhe(
    const char* norm,
    const char* uplo,
    const int n,
    const c64* restrict A,
    const int lda,
    f32* restrict work)
{
    int i, j;
    f32 absa, scale, sum, value;

    if (n == 0) {
        return 0.0f;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))). */
        value = 0.0f;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                for (i = 0; i < j; i++) {
                    sum = cabsf(A[i + j * lda]);
                    if (value < sum || isnan(sum)) value = sum;
                }
                sum = fabsf(crealf(A[j + j * lda]));
                if (value < sum || isnan(sum)) value = sum;
            }
        } else {
            for (j = 0; j < n; j++) {
                sum = fabsf(crealf(A[j + j * lda]));
                if (value < sum || isnan(sum)) value = sum;
                for (i = j + 1; i < n; i++) {
                    sum = cabsf(A[i + j * lda]);
                    if (value < sum || isnan(sum)) value = sum;
                }
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' || norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find normI(A) ( = norm1(A), since A is hermitian). */
        value = 0.0f;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                sum = 0.0f;
                for (i = 0; i < j; i++) {
                    absa = cabsf(A[i + j * lda]);
                    sum += absa;
                    work[i] += absa;
                }
                work[j] = sum + fabsf(crealf(A[j + j * lda]));
            }
            for (i = 0; i < n; i++) {
                sum = work[i];
                if (value < sum || isnan(sum)) value = sum;
            }
        } else {
            for (i = 0; i < n; i++) {
                work[i] = 0.0f;
            }
            for (j = 0; j < n; j++) {
                sum = work[j] + fabsf(crealf(A[j + j * lda]));
                for (i = j + 1; i < n; i++) {
                    absa = cabsf(A[i + j * lda]);
                    sum += absa;
                    work[i] += absa;
                }
                if (value < sum || isnan(sum)) value = sum;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A). */
        scale = 0.0f;
        sum = 1.0f;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 1; j < n; j++) {
                classq(j, &A[0 + j * lda], 1, &scale, &sum);
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                classq(n - j - 1, &A[(j + 1) + j * lda], 1, &scale, &sum);
            }
        }
        sum = 2.0f * sum;
        for (i = 0; i < n; i++) {
            if (crealf(A[i + i * lda]) != 0.0f) {
                absa = fabsf(crealf(A[i + i * lda]));
                if (scale < absa) {
                    sum = 1.0f + sum * (scale / absa) * (scale / absa);
                    scale = absa;
                } else {
                    sum = sum + (absa / scale) * (absa / scale);
                }
            }
        }
        value = scale * sqrtf(sum);
    } else {
        value = 0.0f;
    }

    return value;
}
