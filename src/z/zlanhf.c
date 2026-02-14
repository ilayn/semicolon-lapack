/**
 * @file zlanhf.c
 * @brief ZLANHF returns the value of the 1-norm, or the Frobenius norm, or
 *        the infinity norm, or the element of largest absolute value of a
 *        complex Hermitian matrix in RFP format.
 */

#include "semicolon_lapack_complex_double.h"
#include <math.h>
#include <complex.h>

/**
 * ZLANHF returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * complex Hermitian matrix A in RFP format.
 *
 * @param[in] norm
 *          Specifies the value to be returned in ZLANHF as described
 *          above.
 *          = 'M' or 'm': max(abs(A(i,j)))
 *          = '1', 'O' or 'o': norm1(A)
 *          = 'I' or 'i': normI(A)
 *          = 'F', 'f', 'E' or 'e': normF(A)
 *
 * @param[in] transr
 *          = 'N':  RFP format is Normal;
 *          = 'C':  RFP format is Conjugate-transposed.
 *
 * @param[in] uplo
 *          = 'U': RFP A came from an upper triangular matrix;
 *          = 'L': RFP A came from a lower triangular matrix.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0. When n = 0, ZLANHF is
 *          set to zero.
 *
 * @param[in] A
 *          Complex*16 array, dimension ( n*(n+1)/2 );
 *          On entry, the matrix A in RFP Format.
 *
 * @param[out] work
 *          Double precision array, dimension (MAX(1,LWORK)),
 *          where LWORK >= n when NORM = 'I' or '1' or 'O'; otherwise,
 *          WORK is not referenced.
 *
 * @return The norm value.
 */
double zlanhf(
    const char* norm,
    const char* transr,
    const char* uplo,
    const int n,
    const double complex* const restrict A,
    double* const restrict work)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    int i, j, ifm, ilu, noe, n1, k, l, lda;
    double scale, s, value, aa, temp;

    if (n == 0) {
        return ZERO;
    } else if (n == 1) {
        return fabs(creal(A[0]));
    }

    noe = 1;
    if (n % 2 == 0) {
        noe = 0;
    }

    ifm = 1;
    if (transr[0] == 'C' || transr[0] == 'c') {
        ifm = 0;
    }

    ilu = 1;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        ilu = 0;
    }

    if (ifm == 1) {
        if (noe == 1) {
            lda = n;
        } else {
            lda = n + 1;
        }
    } else {
        lda = (n + 1) / 2;
    }

    if (norm[0] == 'M' || norm[0] == 'm') {

        k = (n + 1) / 2;
        value = ZERO;
        if (noe == 1) {
            if (ifm == 1) {
                if (ilu == 1) {
                    j = 0;
                    temp = fabs(creal(A[j + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    for (i = 1; i <= n - 1; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    for (j = 1; j <= k - 1; j++) {
                        for (i = 0; i <= j - 2; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = j - 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = j;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = j + 1; i <= n - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                } else {
                    for (j = 0; j <= k - 2; j++) {
                        for (i = 0; i <= k + j - 2; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = k + j - 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = i + 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = k + j + 1; i <= n - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                    for (i = 0; i <= n - 2; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    temp = fabs(creal(A[i + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                }
            } else {
                if (ilu == 1) {
                    for (j = 0; j <= k - 2; j++) {
                        for (i = 0; i <= j - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = j;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = j + 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = j + 2; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                    j = k - 1;
                    for (i = 0; i <= k - 2; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    i = k - 1;
                    temp = fabs(creal(A[i + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    for (j = k; j <= n - 1; j++) {
                        for (i = 0; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                } else {
                    for (j = 0; j <= k - 2; j++) {
                        for (i = 0; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                    j = k - 1;
                    temp = fabs(creal(A[0 + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    for (i = 1; i <= k - 1; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    for (j = k; j <= n - 1; j++) {
                        for (i = 0; i <= j - k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = j - k;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = j - k + 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = j - k + 2; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                }
            }
        } else {
            if (ifm == 1) {
                if (ilu == 1) {
                    j = 0;
                    temp = fabs(creal(A[j + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    temp = fabs(creal(A[j + 1 + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    for (i = 2; i <= n; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    for (j = 1; j <= k - 1; j++) {
                        for (i = 0; i <= j - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = j;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = j + 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = j + 2; i <= n; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                } else {
                    for (j = 0; j <= k - 2; j++) {
                        for (i = 0; i <= k + j - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = k + j;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = i + 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = k + j + 2; i <= n; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                    for (i = 0; i <= n - 2; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    temp = fabs(creal(A[i + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    i = n;
                    temp = fabs(creal(A[i + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                }
            } else {
                if (ilu == 1) {
                    j = 0;
                    temp = fabs(creal(A[j + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    for (i = 1; i <= k - 1; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    for (j = 1; j <= k - 1; j++) {
                        for (i = 0; i <= j - 2; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = j - 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = j;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = j + 1; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                    j = k;
                    for (i = 0; i <= k - 2; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    i = k - 1;
                    temp = fabs(creal(A[i + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    for (j = k + 1; j <= n; j++) {
                        for (i = 0; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                } else {
                    for (j = 0; j <= k - 1; j++) {
                        for (i = 0; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                    j = k;
                    temp = fabs(creal(A[0 + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                    for (i = 1; i <= k - 1; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    for (j = k + 1; j <= n - 1; j++) {
                        for (i = 0; i <= j - k - 2; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                        i = j - k - 1;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        i = j - k;
                        temp = fabs(creal(A[i + j * lda]));
                        if (value < temp || disnan(temp))
                            value = temp;
                        for (i = j - k + 1; i <= k - 1; i++) {
                            temp = cabs(A[i + j * lda]);
                            if (value < temp || disnan(temp))
                                value = temp;
                        }
                    }
                    j = n;
                    for (i = 0; i <= k - 2; i++) {
                        temp = cabs(A[i + j * lda]);
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                    i = k - 1;
                    temp = fabs(creal(A[i + j * lda]));
                    if (value < temp || disnan(temp))
                        value = temp;
                }
            }
        }

    } else if ((norm[0] == 'I' || norm[0] == 'i') ||
               (norm[0] == 'O' || norm[0] == 'o') ||
               (norm[0] == '1')) {

        if (ifm == 1) {
            k = n / 2;
            if (noe == 1) {
                if (ilu == 0) {
                    for (i = 0; i <= k - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = 0; j <= k; j++) {
                        s = ZERO;
                        for (i = 0; i <= k + j - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[i] = work[i] + aa;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        work[j + k] = s + aa;
                        if (i == k + k)
                            goto L10;
                        i = i + 1;
                        aa = fabs(creal(A[i + j * lda]));
                        work[j] = work[j] + aa;
                        s = ZERO;
                        for (l = j + 1; l <= k - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
L10:
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                } else {
                    k = k + 1;
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = k - 1; j >= 0; j--) {
                        s = ZERO;
                        for (i = 0; i <= j - 2; i++) {
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[i + k] = work[i + k] + aa;
                        }
                        if (j > 0) {
                            aa = fabs(creal(A[i + j * lda]));
                            s = s + aa;
                            work[i + k] = work[i + k] + s;
                            i = i + 1;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        work[j] = aa;
                        s = ZERO;
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                }
            } else {
                if (ilu == 0) {
                    for (i = 0; i <= k - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = 0; j <= k - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= k + j - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[i] = work[i] + aa;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        work[j + k] = s + aa;
                        i = i + 1;
                        aa = fabs(creal(A[i + j * lda]));
                        work[j] = work[j] + aa;
                        s = ZERO;
                        for (l = j + 1; l <= k - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                } else {
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = k - 1; j >= 0; j--) {
                        s = ZERO;
                        for (i = 0; i <= j - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[i + k] = work[i + k] + aa;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        s = s + aa;
                        work[i + k] = work[i + k] + s;
                        i = i + 1;
                        aa = fabs(creal(A[i + j * lda]));
                        work[j] = aa;
                        s = ZERO;
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                }
            }
        } else {
            k = n / 2;
            if (noe == 1) {
                if (ilu == 0) {
                    n1 = k;
                    k = k + 1;
                    for (i = n1; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = 0; j <= n1 - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= k - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i + n1] = work[i + n1] + aa;
                            s = s + aa;
                        }
                        work[j] = s;
                    }
                    s = fabs(creal(A[0 + j * lda]));
                    for (i = 1; i <= k - 1; i++) {
                        aa = cabs(A[i + j * lda]);
                        work[i + n1] = work[i + n1] + aa;
                        s = s + aa;
                    }
                    work[j] = work[j] + s;
                    for (j = k; j <= n - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - k - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        s = s + aa;
                        work[j - k] = work[j - k] + s;
                        i = i + 1;
                        s = fabs(creal(A[i + j * lda]));
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            work[l] = work[l] + aa;
                            s = s + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                } else {
                    k = k + 1;
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = 0; j <= k - 2; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        s = s + aa;
                        work[j] = s;
                        i = i + 1;
                        aa = fabs(creal(A[i + j * lda]));
                        s = aa;
                        for (l = k + j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[k + j] = work[k + j] + s;
                    }
                    s = ZERO;
                    for (i = 0; i <= k - 2; i++) {
                        aa = cabs(A[i + j * lda]);
                        work[i] = work[i] + aa;
                        s = s + aa;
                    }
                    aa = fabs(creal(A[i + j * lda]));
                    s = s + aa;
                    work[i] = s;
                    for (j = k; j <= n - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= k - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                }
            } else {
                if (ilu == 0) {
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = 0; j <= k - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= k - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i + k] = work[i + k] + aa;
                            s = s + aa;
                        }
                        work[j] = s;
                    }
                    aa = fabs(creal(A[0 + j * lda]));
                    s = aa;
                    for (i = 1; i <= k - 1; i++) {
                        aa = cabs(A[i + j * lda]);
                        work[i + k] = work[i + k] + aa;
                        s = s + aa;
                    }
                    work[j] = work[j] + s;
                    for (j = k + 1; j <= n - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - 2 - k; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        s = s + aa;
                        work[j - k - 1] = work[j - k - 1] + s;
                        i = i + 1;
                        aa = fabs(creal(A[i + j * lda]));
                        s = aa;
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            work[l] = work[l] + aa;
                            s = s + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    s = ZERO;
                    for (i = 0; i <= k - 2; i++) {
                        aa = cabs(A[i + j * lda]);
                        work[i] = work[i] + aa;
                        s = s + aa;
                    }
                    aa = fabs(creal(A[i + j * lda]));
                    s = s + aa;
                    work[i] = work[i] + s;
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                } else {
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    s = fabs(creal(A[0]));
                    for (i = 1; i <= k - 1; i++) {
                        aa = cabs(A[i]);
                        work[i + k] = work[i + k] + aa;
                        s = s + aa;
                    }
                    work[k] = work[k] + s;
                    for (j = 1; j <= k - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - 2; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(creal(A[i + j * lda]));
                        s = s + aa;
                        work[j - 1] = s;
                        i = i + 1;
                        aa = fabs(creal(A[i + j * lda]));
                        s = aa;
                        for (l = k + j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = cabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[k + j] = work[k + j] + s;
                    }
                    s = ZERO;
                    for (i = 0; i <= k - 2; i++) {
                        aa = cabs(A[i + j * lda]);
                        work[i] = work[i] + aa;
                        s = s + aa;
                    }
                    aa = fabs(creal(A[i + j * lda]));
                    s = s + aa;
                    work[i] = s;
                    for (j = k + 1; j <= n; j++) {
                        s = ZERO;
                        for (i = 0; i <= k - 1; i++) {
                            aa = cabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        work[j - 1] = work[j - 1] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp))
                            value = temp;
                    }
                }
            }
        }

    } else if ((norm[0] == 'F' || norm[0] == 'f') ||
               (norm[0] == 'E' || norm[0] == 'e')) {

        k = (n + 1) / 2;
        scale = ZERO;
        s = ONE;
        if (noe == 1) {
            if (ifm == 1) {
                if (ilu == 0) {
                    for (j = 0; j <= k - 3; j++) {
                        zlassq(k - j - 2, &A[k + j + 1 + j * lda], 1,
                               &scale, &s);
                    }
                    for (j = 0; j <= k - 1; j++) {
                        zlassq(k + j - 1, &A[0 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    l = k - 1;
                    for (i = 0; i <= k - 2; i++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                    aa = creal(A[l]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                } else {
                    for (j = 0; j <= k - 1; j++) {
                        zlassq(n - j - 1, &A[j + 1 + j * lda], 1,
                               &scale, &s);
                    }
                    for (j = 1; j <= k - 2; j++) {
                        zlassq(j, &A[0 + (1 + j) * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    aa = creal(A[0]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                    l = lda;
                    for (i = 1; i <= k - 1; i++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                }
            } else {
                if (ilu == 0) {
                    for (j = 1; j <= k - 2; j++) {
                        zlassq(j, &A[0 + (k + j) * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        zlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        zlassq(k - j - 1, &A[j + 1 + (j + k - 1) * lda], 1,
                               &scale, &s);
                    }
                    s = s + s;
                    l = 0 + k * lda - lda;
                    aa = creal(A[l]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                    l = l + lda;
                    for (j = k; j <= n - 1; j++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                } else {
                    for (j = 1; j <= k - 1; j++) {
                        zlassq(j, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = k; j <= n - 1; j++) {
                        zlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 3; j++) {
                        zlassq(k - j - 2, &A[j + 2 + j * lda], 1,
                               &scale, &s);
                    }
                    s = s + s;
                    l = 0;
                    for (i = 0; i <= k - 2; i++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                    aa = creal(A[l]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                }
            }
        } else {
            if (ifm == 1) {
                if (ilu == 0) {
                    for (j = 0; j <= k - 2; j++) {
                        zlassq(k - j - 1, &A[k + j + 2 + j * lda], 1,
                               &scale, &s);
                    }
                    for (j = 0; j <= k - 1; j++) {
                        zlassq(k + j, &A[0 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    l = k;
                    for (i = 0; i <= k - 1; i++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                } else {
                    for (j = 0; j <= k - 1; j++) {
                        zlassq(n - j - 1, &A[j + 2 + j * lda], 1,
                               &scale, &s);
                    }
                    for (j = 1; j <= k - 1; j++) {
                        zlassq(j, &A[0 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    l = 0;
                    for (i = 0; i <= k - 1; i++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                }
            } else {
                if (ilu == 0) {
                    for (j = 1; j <= k - 1; j++) {
                        zlassq(j, &A[0 + (k + 1 + j) * lda], 1,
                               &scale, &s);
                    }
                    for (j = 0; j <= k - 1; j++) {
                        zlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        zlassq(k - j - 1, &A[j + 1 + (j + k) * lda], 1,
                               &scale, &s);
                    }
                    s = s + s;
                    l = 0 + k * lda;
                    aa = creal(A[l]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                    l = l + lda;
                    for (j = k + 1; j <= n - 1; j++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                    aa = creal(A[l]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                } else {
                    for (j = 1; j <= k - 1; j++) {
                        zlassq(j, &A[0 + (j + 1) * lda], 1, &scale, &s);
                    }
                    for (j = k + 1; j <= n; j++) {
                        zlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        zlassq(k - j - 1, &A[j + 1 + j * lda], 1,
                               &scale, &s);
                    }
                    s = s + s;
                    l = 0;
                    aa = creal(A[l]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                    l = lda;
                    for (i = 0; i <= k - 2; i++) {
                        aa = creal(A[l]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        aa = creal(A[l + 1]);
                        if (aa != ZERO) {
                            if (scale < aa) {
                                s = ONE + s * (scale / aa) * (scale / aa);
                                scale = aa;
                            } else {
                                s = s + (aa / scale) * (aa / scale);
                            }
                        }
                        l = l + lda + 1;
                    }
                    aa = creal(A[l]);
                    if (aa != ZERO) {
                        if (scale < aa) {
                            s = ONE + s * (scale / aa) * (scale / aa);
                            scale = aa;
                        } else {
                            s = s + (aa / scale) * (aa / scale);
                        }
                    }
                }
            }
        }
        value = scale * sqrt(s);
    } else {
        value = ZERO;
    }

    return value;
}
