/**
 * @file zlanhp.c
 * @brief ZLANHP returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the element of largest absolute value of a complex Hermitian matrix supplied in packed form.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLANHP returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex hermitian matrix A, supplied in packed form.
 *
 * ZLANHP = ( max(abs(A(i,j))), NORM = 'M' or 'm'
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
 * @param[in]     AP     The upper or lower triangle of the hermitian matrix A,
 *                       packed columnwise in a linear array.
 *                       Array of dimension (n*(n+1)/2).
 *                       Note that the imaginary parts of the diagonal elements
 *                       need not be set and are assumed to be zero.
 * @param[out]    work   Workspace array of dimension (max(1,lwork)),
 *                       where lwork >= n when norm = 'I' or '1' or 'O';
 *                       otherwise, work is not referenced.
 *
 * @return The computed norm value.
 */
double zlanhp(
    const char* norm,
    const char* uplo,
    const int n,
    const double complex* const restrict AP,
    double* const restrict work)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    int i, j, k;
    double absa, scale, sum, value;

    if (n == 0) {
        value = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {

        // Find max(abs(A(i,j))).

        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            k = 0;
            for (j = 0; j < n; j++) {
                // Fortran: DO 10 I = K + 1, K + J - 1 (off-diagonal elements)
                // 0-based: i = k to k + j - 1 (j elements before diagonal)
                for (i = k; i < k + j; i++) {
                    sum = cabs(AP[i]);
                    if (value < sum || disnan(sum)) value = sum;
                }
                // Fortran: K = K + J, then SUM = ABS( DBLE( AP( K ) ) )
                // 0-based: diagonal is at k + j
                sum = fabs(creal(AP[k + j]));
                if (value < sum || disnan(sum)) value = sum;
                k = k + (j + 1);
            }
        } else {
            k = 0;
            for (j = 0; j < n; j++) {
                // Fortran: SUM = ABS( DBLE( AP( K ) ) ) — diagonal first
                sum = fabs(creal(AP[k]));
                if (value < sum || disnan(sum)) value = sum;
                // Fortran: DO 30 I = K + 1, K + N - J (off-diagonal elements)
                // 0-based: i = k + 1 to k + n - j - 1
                for (i = k + 1; i < k + n - j; i++) {
                    sum = cabs(AP[i]);
                    if (value < sum || disnan(sum)) value = sum;
                }
                k = k + (n - j);
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' ||
               norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {

        // Find normI(A) ( = norm1(A), since A is hermitian).

        value = ZERO;
        k = 0;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                sum = ZERO;
                // Fortran: DO 50 I = 1, J - 1 (off-diagonal)
                for (i = 0; i < j; i++) {
                    absa = cabs(AP[k]);
                    sum = sum + absa;
                    work[i] = work[i] + absa;
                    k = k + 1;
                }
                // Fortran: WORK( J ) = SUM + ABS( DBLE( AP( K ) ) )
                work[j] = sum + fabs(creal(AP[k]));
                k = k + 1;
            }
            for (i = 0; i < n; i++) {
                sum = work[i];
                if (value < sum || disnan(sum)) value = sum;
            }
        } else {
            for (i = 0; i < n; i++) {
                work[i] = ZERO;
            }
            for (j = 0; j < n; j++) {
                // Fortran: SUM = WORK( J ) + ABS( DBLE( AP( K ) ) )
                sum = work[j] + fabs(creal(AP[k]));
                k = k + 1;
                // Fortran: DO 90 I = J + 1, N (off-diagonal)
                for (i = j + 1; i < n; i++) {
                    absa = cabs(AP[k]);
                    sum = sum + absa;
                    work[i] = work[i] + absa;
                    k = k + 1;
                }
                if (value < sum || disnan(sum)) value = sum;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {

        // Find normF(A).

        scale = ZERO;
        sum = ONE;
        k = 1;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 1; j < n; j++) {
                // Fortran: CALL ZLASSQ( J-1, AP( K ), 1, SCALE, SUM )
                zlassq(j, &AP[k], 1, &scale, &sum);
                k = k + (j + 1);
            }
        } else {
            for (j = 0; j < n - 1; j++) {
                // Fortran: CALL ZLASSQ( N-J, AP( K ), 1, SCALE, SUM )
                zlassq(n - j - 1, &AP[k], 1, &scale, &sum);
                k = k + (n - j);
            }
        }
        sum = 2 * sum;
        k = 0;
        for (i = 0; i < n; i++) {
            if (creal(AP[k]) != ZERO) {
                absa = fabs(creal(AP[k]));
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
        value = scale * sqrt(sum);
    } else {
        value = ZERO;
    }

    return value;
}
