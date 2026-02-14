/**
 * @file zlansb.c
 * @brief ZLANSB returns the value of the 1-norm, Frobenius norm, infinity norm, or max element of a symmetric band matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLANSB returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of an
 * n by n symmetric band matrix A, with k super-diagonals.
 *
 * @param[in]     norm   = 'M': max(abs(A(i,j)))
 *                        = '1', 'O': norm1(A) (maximum column sum)
 *                        = 'I': normI(A) (maximum row sum)
 *                        = 'F', 'E': normF(A) (Frobenius norm)
 * @param[in]     uplo   = 'U': Upper triangular part is supplied
 *                        = 'L': Lower triangular part is supplied
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     k      The number of super/sub-diagonals. k >= 0.
 * @param[in]     AB     The banded matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= k+1.
 * @param[out]    work   Workspace array of dimension (n) when norm = 'I' or '1' or 'O'.
 *
 * @return The computed norm value.
 */
double zlansb(
    const char* norm,
    const char* uplo,
    const int n,
    const int k,
    const double complex* const restrict AB,
    const int ldab,
    double* const restrict work)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    int i, j, l;
    double absa, scale, sum, value;

    if (n == 0) {
        value = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        // Find max(abs(A(i,j)))
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                int istart = (k + 1 - j - 1 > 0) ? k + 1 - j - 1 : 0;
                for (i = istart; i <= k; i++) {
                    sum = cabs(AB[i + j * ldab]);
                    if (value < sum || disnan(sum)) value = sum;
                }
            }
        } else {
            for (j = 0; j < n; j++) {
                int iend = (n - j < k + 1) ? n - j : k + 1;
                for (i = 0; i < iend; i++) {
                    sum = cabs(AB[i + j * ldab]);
                    if (value < sum || disnan(sum)) value = sum;
                }
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i' ||
               norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        // Find normI(A) (= norm1(A), since A is symmetric)
        value = ZERO;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                sum = ZERO;
                l = k - j;
                int istart = (j - k > 0) ? j - k : 0;
                for (i = istart; i < j; i++) {
                    absa = cabs(AB[l + i + j * ldab]);
                    sum = sum + absa;
                    work[i] = work[i] + absa;
                }
                work[j] = sum + cabs(AB[k + j * ldab]);
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
                sum = work[j] + cabs(AB[j * ldab]);
                l = -j;
                int iend = (n < j + k + 1) ? n : j + k + 1;
                for (i = j + 1; i < iend; i++) {
                    absa = cabs(AB[l + i + j * ldab]);
                    sum = sum + absa;
                    work[i] = work[i] + absa;
                }
                if (value < sum || disnan(sum)) value = sum;
            }
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' ||
               norm[0] == 'E' || norm[0] == 'e') {
        // Find normF(A)
        scale = ZERO;
        sum = ONE;
        if (k > 0) {
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                for (j = 1; j < n; j++) {
                    int len = (j < k) ? j : k;
                    int start = (k + 1 - j - 1 > 0) ? k + 1 - j - 1 : 0;
                    zlassq(len, &AB[start + j * ldab], 1, &scale, &sum);
                }
                l = k;
            } else {
                for (j = 0; j < n - 1; j++) {
                    int len = (n - j - 1 < k) ? n - j - 1 : k;
                    zlassq(len, &AB[1 + j * ldab], 1, &scale, &sum);
                }
                l = 0;
            }
            sum = 2 * sum;
        } else {
            l = 0;
        }
        zlassq(n, &AB[l], ldab, &scale, &sum);
        value = scale * sqrt(sum);
    } else {
        value = ZERO;
    }

    return value;
}
