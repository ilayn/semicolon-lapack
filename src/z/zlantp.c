/**
 * @file zlantp.c
 * @brief ZLANTP returns the value of the 1-norm, Frobenius norm, infinity norm, or max element of a packed triangular matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZLANTP returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * triangular matrix A, supplied in packed form.
 *
 * @param[in]     norm   = 'M': max(abs(A(i,j)))
 *                        = '1', 'O': norm1(A) (maximum column sum)
 *                        = 'I': normI(A) (maximum row sum)
 *                        = 'F', 'E': normF(A) (Frobenius norm)
 * @param[in]     uplo   = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     diag   = 'N': Non-unit triangular
 *                        = 'U': Unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     AP     The packed triangular matrix A. Array of dimension (n*(n+1)/2).
 * @param[out]    work   Workspace array of dimension (n) when norm = 'I'.
 *
 * @return The computed norm value.
 */
f64 zlantp(
    const char* norm,
    const char* uplo,
    const char* diag,
    const int n,
    const c128* const restrict AP,
    f64* const restrict work)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int udiag;
    int i, j, k;
    f64 scale, sum, value;

    if (n == 0) {
        value = ZERO;
    } else if (norm[0] == 'M' || norm[0] == 'm') {
        // Find max(abs(A(i,j)))
        k = 0;
        if (diag[0] == 'U' || diag[0] == 'u') {
            value = ONE;
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                for (j = 1; j <= n; j++) {
                    for (i = k; i < k + j - 1; i++) {
                        sum = cabs(AP[i]);
                        if (value < sum || disnan(sum)) value = sum;
                    }
                    k = k + j;
                }
            } else {
                for (j = 1; j <= n; j++) {
                    for (i = k + 1; i < k + n - j + 1; i++) {
                        sum = cabs(AP[i]);
                        if (value < sum || disnan(sum)) value = sum;
                    }
                    k = k + n - j + 1;
                }
            }
        } else {
            value = ZERO;
            if (uplo[0] == 'U' || uplo[0] == 'u') {
                for (j = 1; j <= n; j++) {
                    for (i = k; i < k + j; i++) {
                        sum = cabs(AP[i]);
                        if (value < sum || disnan(sum)) value = sum;
                    }
                    k = k + j;
                }
            } else {
                for (j = 1; j <= n; j++) {
                    for (i = k; i < k + n - j + 1; i++) {
                        sum = cabs(AP[i]);
                        if (value < sum || disnan(sum)) value = sum;
                    }
                    k = k + n - j + 1;
                }
            }
        }
    } else if (norm[0] == 'O' || norm[0] == 'o' || norm[0] == '1') {
        // Find norm1(A) (maximum column sum)
        value = ZERO;
        k = 0;
        udiag = (diag[0] == 'U' || diag[0] == 'u');
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 1; j <= n; j++) {
                if (udiag) {
                    sum = ONE;
                    for (i = k; i < k + j - 1; i++) {
                        sum = sum + cabs(AP[i]);
                    }
                } else {
                    sum = ZERO;
                    for (i = k; i < k + j; i++) {
                        sum = sum + cabs(AP[i]);
                    }
                }
                k = k + j;
                if (value < sum || disnan(sum)) value = sum;
            }
        } else {
            for (j = 1; j <= n; j++) {
                if (udiag) {
                    sum = ONE;
                    for (i = k + 1; i < k + n - j + 1; i++) {
                        sum = sum + cabs(AP[i]);
                    }
                } else {
                    sum = ZERO;
                    for (i = k; i < k + n - j + 1; i++) {
                        sum = sum + cabs(AP[i]);
                    }
                }
                k = k + n - j + 1;
                if (value < sum || disnan(sum)) value = sum;
            }
        }
    } else if (norm[0] == 'I' || norm[0] == 'i') {
        // Find normI(A) (maximum row sum)
        k = 0;
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            if (diag[0] == 'U' || diag[0] == 'u') {
                for (i = 0; i < n; i++) {
                    work[i] = ONE;
                }
                for (j = 1; j <= n; j++) {
                    for (i = 0; i < j - 1; i++) {
                        work[i] = work[i] + cabs(AP[k]);
                        k = k + 1;
                    }
                    k = k + 1;
                }
            } else {
                for (i = 0; i < n; i++) {
                    work[i] = ZERO;
                }
                for (j = 1; j <= n; j++) {
                    for (i = 0; i < j; i++) {
                        work[i] = work[i] + cabs(AP[k]);
                        k = k + 1;
                    }
                }
            }
        } else {
            if (diag[0] == 'U' || diag[0] == 'u') {
                for (i = 0; i < n; i++) {
                    work[i] = ONE;
                }
                for (j = 1; j <= n; j++) {
                    k = k + 1;
                    for (i = j; i < n; i++) {
                        work[i] = work[i] + cabs(AP[k]);
                        k = k + 1;
                    }
                }
            } else {
                for (i = 0; i < n; i++) {
                    work[i] = ZERO;
                }
                for (j = 1; j <= n; j++) {
                    for (i = j - 1; i < n; i++) {
                        work[i] = work[i] + cabs(AP[k]);
                        k = k + 1;
                    }
                }
            }
        }
        value = ZERO;
        for (i = 0; i < n; i++) {
            sum = work[i];
            if (value < sum || disnan(sum)) value = sum;
        }
    } else if (norm[0] == 'F' || norm[0] == 'f' || norm[0] == 'E' || norm[0] == 'e') {
        // Find normF(A) (Frobenius norm)
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            if (diag[0] == 'U' || diag[0] == 'u') {
                scale = ONE;
                sum = n;
                k = 1;
                for (j = 2; j <= n; j++) {
                    zlassq(j - 1, &AP[k], 1, &scale, &sum);
                    k = k + j;
                }
            } else {
                scale = ZERO;
                sum = ONE;
                k = 0;
                for (j = 1; j <= n; j++) {
                    zlassq(j, &AP[k], 1, &scale, &sum);
                    k = k + j;
                }
            }
        } else {
            if (diag[0] == 'U' || diag[0] == 'u') {
                scale = ONE;
                sum = n;
                k = 1;
                for (j = 1; j <= n - 1; j++) {
                    zlassq(n - j, &AP[k], 1, &scale, &sum);
                    k = k + n - j + 1;
                }
            } else {
                scale = ZERO;
                sum = ONE;
                k = 0;
                for (j = 1; j <= n; j++) {
                    zlassq(n - j + 1, &AP[k], 1, &scale, &sum);
                    k = k + n - j + 1;
                }
            }
        }
        value = scale * sqrt(sum);
    } else {
        value = ZERO;
    }

    return value;
}
