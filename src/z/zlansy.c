/**
 * @file zlansy.c
 * @brief ZLANSY returns the value of the 1-norm, Frobenius norm,
 *        infinity-norm, or the largest absolute value of any element of
 *        a complex symmetric matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLANSY returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex symmetric matrix A.
 *
 * @param[in] norm  Specifies the value to be returned:
 *                  = 'M' or 'm': max(abs(A(i,j)))
 *                  = "1", 'O' or 'o': norm1(A) (max column sum)
 *                  = 'I' or 'i': normI(A) (max row sum)
 *                  = "F", "f", 'E' or 'e': normF(A) (Frobenius norm)
 *                  Note: norm1(A) = normI(A) since A is symmetric.
 * @param[in] uplo  Specifies whether the upper or lower triangular part
 *                  of A is referenced:
 *                  = 'U': Upper triangular part of A is referenced
 *                  = 'L': Lower triangular part of A is referenced
 * @param[in] n     The order of the matrix A. n >= 0.
 *                  When n = 0, zlansy returns zero.
 * @param[in] A     Double complex array, dimension (lda, n).
 *                  The symmetric matrix A.
 * @param[in] lda   The leading dimension of A. lda >= max(n, 1).
 * @param[out] work Double precision array, dimension (max(1, lwork)).
 *                  where lwork >= n when norm = 'I' or '1' or 'O';
 *                  otherwise, work is not referenced.
 *
 * @return The computed norm value.
 */
f64 zlansy(
    const char* norm,
    const char* uplo,
    const int n,
    const c128* restrict A,
    const int lda,
    f64* restrict work)
{
    int i, j;
    f64 absa, scale, sum, value, temp;

    /* Quick return if possible */
    if (n == 0) {
        return 0.0;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {
        /* Find max(abs(A(i,j))) */
        value = 0.0;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                for (i = 0; i <= j; i++) {
                    temp = cabs(A[i + j * lda]);
                    if (value < temp || isnan(temp)) {
                        value = temp;
                    }
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                for (i = j; i < n; i++) {
                    temp = cabs(A[i + j * lda]);
                    if (value < temp || isnan(temp)) {
                        value = temp;
                    }
                }
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' || norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        /* Find normI(A) ( = norm1(A), since A is symmetric) */
        value = 0.0;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                sum = 0.0;
                for (i = 0; i < j; i++) {
                    absa = cabs(A[i + j * lda]);
                    sum += absa;
                    work[i] += absa;
                }
                work[j] = sum + cabs(A[j + j * lda]);
            }
            for (i = 0; i < n; i++) {
                temp = work[i];
                if (value < temp || isnan(temp)) {
                    value = temp;
                }
            }
        } else {
            for (i = 0; i < n; i++) {
                work[i] = 0.0;
            }
            for (j = 0; j < n; j++) {
                sum = work[j] + cabs(A[j + j * lda]);
                for (i = j + 1; i < n; i++) {
                    absa = cabs(A[i + j * lda]);
                    sum += absa;
                    work[i] += absa;
                }
                if (value < sum || isnan(sum)) {
                    value = sum;
                }
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        /* Find normF(A) */
        scale = 0.0;
        sum = 1.0;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 1; j < n; j++) {
                zlassq(j, &A[0 + j * lda], 1, &scale, &sum);
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                zlassq(n - j - 1, &A[(j + 1) + j * lda], 1, &scale, &sum);
            }
        }
        sum = 2.0 * sum;
        zlassq(n, A, lda + 1, &scale, &sum);
        value = scale * sqrt(sum);
    } else {
        value = 0.0;
    }

    return value;
}
