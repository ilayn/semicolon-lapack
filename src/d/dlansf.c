/**
 * @file dlansf.c
 * @brief DLANSF returns the value of the 1-norm, or the Frobenius norm, or the infinity norm, or the element of largest absolute value of a symmetric matrix in RFP format.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * DLANSF returns the value of the one norm, or the Frobenius norm, or
 * the infinity norm, or the element of largest absolute value of a
 * real symmetric matrix A in RFP format.
 *
 * @param[in] norm
 *          Specifies the value to be returned in DLANSF as described
 *          above.
 *          = 'M' or 'm': max(abs(A(i,j)))
 *          = '1', 'O' or 'o': norm1(A)
 *          = 'I' or 'i': normI(A)
 *          = 'F', 'f', 'E' or 'e': normF(A)
 *
 * @param[in] transr
 *          = 'N':  RFP format is Normal;
 *          = 'T':  RFP format is Transpose.
 *
 * @param[in] uplo
 *          = 'U': RFP A came from an upper triangular matrix;
 *          = 'L': RFP A came from a lower triangular matrix.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0. When n = 0, DLANSF is
 *          set to zero.
 *
 * @param[in] A
 *          Double precision array, dimension ( n*(n+1)/2 );
 *          On entry, the upper (if UPLO = 'U') or lower (if UPLO = 'L')
 *          part of the symmetric matrix A stored in RFP format.
 *
 * @param[out] work
 *          Double precision array, dimension (MAX(1,LWORK)),
 *          where LWORK >= n when NORM = 'I' or '1' or 'O'; otherwise,
 *          WORK is not referenced.
 *
 * @return The norm value.
 */
f64 dlansf(
    const char* norm,
    const char* transr,
    const char* uplo,
    const int n,
    const f64* const restrict A,
    f64* const restrict work)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int i, j, ifm, ilu, noe, n1, k, l, lda;
    f64 scale, s, value, aa, temp;

    if (n == 0) {
        return ZERO;
    } else if (n == 1) {
        return fabs(A[0]);
    }

    noe = 1;
    if (n % 2 == 0) {
        noe = 0;
    }

    ifm = 1;
    if (transr[0] == 'T' || transr[0] == 't') {
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
                for (j = 0; j <= k - 1; j++) {
                    for (i = 0; i <= n - 1; i++) {
                        temp = fabs(A[i + j * lda]);
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
                }
            } else {
                for (j = 0; j <= n - 1; j++) {
                    for (i = 0; i <= k - 1; i++) {
                        temp = fabs(A[i + j * lda]);
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
                }
            }
        } else {
            if (ifm == 1) {
                for (j = 0; j <= k - 1; j++) {
                    for (i = 0; i <= n; i++) {
                        temp = fabs(A[i + j * lda]);
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
                }
            } else {
                for (j = 0; j <= n; j++) {
                    for (i = 0; i <= k - 1; i++) {
                        temp = fabs(A[i + j * lda]);
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
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
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[i] = work[i] + aa;
                        }
                        aa = fabs(A[i + j * lda]);
                        work[j + k] = s + aa;
                        if (i == k + k) {
                            goto L10;
                        }
                        i = i + 1;
                        aa = fabs(A[i + j * lda]);
                        work[j] = work[j] + aa;
                        s = ZERO;
                        for (l = j + 1; l <= k - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
L10:
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
                } else {
                    k = k + 1;
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = k - 1; j >= 0; j--) {
                        s = ZERO;
                        for (i = 0; i <= j - 2; i++) {
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[i + k] = work[i + k] + aa;
                        }
                        if (j > 0) {
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[i + k] = work[i + k] + s;
                            i = i + 1;
                        }
                        aa = fabs(A[i + j * lda]);
                        work[j] = aa;
                        s = ZERO;
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
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
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[i] = work[i] + aa;
                        }
                        aa = fabs(A[i + j * lda]);
                        work[j + k] = s + aa;
                        i = i + 1;
                        aa = fabs(A[i + j * lda]);
                        work[j] = work[j] + aa;
                        s = ZERO;
                        for (l = j + 1; l <= k - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
                } else {
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = k - 1; j >= 0; j--) {
                        s = ZERO;
                        for (i = 0; i <= j - 1; i++) {
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[i + k] = work[i + k] + aa;
                        }
                        aa = fabs(A[i + j * lda]);
                        s = s + aa;
                        work[i + k] = work[i + k] + s;
                        i = i + 1;
                        aa = fabs(A[i + j * lda]);
                        work[j] = aa;
                        s = ZERO;
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
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
                            aa = fabs(A[i + j * lda]);
                            work[i + n1] = work[i + n1] + aa;
                            s = s + aa;
                        }
                        work[j] = s;
                    }
                    s = fabs(A[0 + j * lda]);
                    for (i = 1; i <= k - 1; i++) {
                        aa = fabs(A[i + j * lda]);
                        work[i + n1] = work[i + n1] + aa;
                        s = s + aa;
                    }
                    work[j] = work[j] + s;
                    for (j = k; j <= n - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - k - 1; i++) {
                            aa = fabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(A[i + j * lda]);
                        s = s + aa;
                        work[j - k] = work[j - k] + s;
                        i = i + 1;
                        s = fabs(A[i + j * lda]);
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            work[l] = work[l] + aa;
                            s = s + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
                } else {
                    k = k + 1;
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    for (j = 0; j <= k - 2; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - 1; i++) {
                            aa = fabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(A[i + j * lda]);
                        s = s + aa;
                        work[j] = s;
                        i = i + 1;
                        aa = fabs(A[i + j * lda]);
                        s = aa;
                        for (l = k + j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[k + j] = work[k + j] + s;
                    }
                    s = ZERO;
                    for (i = 0; i <= k - 2; i++) {
                        aa = fabs(A[i + j * lda]);
                        work[i] = work[i] + aa;
                        s = s + aa;
                    }
                    aa = fabs(A[i + j * lda]);
                    s = s + aa;
                    work[i] = s;
                    for (j = k; j <= n - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= k - 1; i++) {
                            aa = fabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
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
                            aa = fabs(A[i + j * lda]);
                            work[i + k] = work[i + k] + aa;
                            s = s + aa;
                        }
                        work[j] = s;
                    }
                    aa = fabs(A[0 + j * lda]);
                    s = aa;
                    for (i = 1; i <= k - 1; i++) {
                        aa = fabs(A[i + j * lda]);
                        work[i + k] = work[i + k] + aa;
                        s = s + aa;
                    }
                    work[j] = work[j] + s;
                    for (j = k + 1; j <= n - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - 2 - k; i++) {
                            aa = fabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(A[i + j * lda]);
                        s = s + aa;
                        work[j - k - 1] = work[j - k - 1] + s;
                        i = i + 1;
                        aa = fabs(A[i + j * lda]);
                        s = aa;
                        for (l = j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            work[l] = work[l] + aa;
                            s = s + aa;
                        }
                        work[j] = work[j] + s;
                    }
                    s = ZERO;
                    for (i = 0; i <= k - 2; i++) {
                        aa = fabs(A[i + j * lda]);
                        work[i] = work[i] + aa;
                        s = s + aa;
                    }
                    aa = fabs(A[i + j * lda]);
                    s = s + aa;
                    work[i] = work[i] + s;
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
                    }
                } else {
                    for (i = k; i <= n - 1; i++) {
                        work[i] = ZERO;
                    }
                    s = fabs(A[0]);
                    for (i = 1; i <= k - 1; i++) {
                        aa = fabs(A[i]);
                        work[i + k] = work[i + k] + aa;
                        s = s + aa;
                    }
                    work[k] = work[k] + s;
                    for (j = 1; j <= k - 1; j++) {
                        s = ZERO;
                        for (i = 0; i <= j - 2; i++) {
                            aa = fabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        aa = fabs(A[i + j * lda]);
                        s = s + aa;
                        work[j - 1] = s;
                        i = i + 1;
                        aa = fabs(A[i + j * lda]);
                        s = aa;
                        for (l = k + j + 1; l <= n - 1; l++) {
                            i = i + 1;
                            aa = fabs(A[i + j * lda]);
                            s = s + aa;
                            work[l] = work[l] + aa;
                        }
                        work[k + j] = work[k + j] + s;
                    }
                    s = ZERO;
                    for (i = 0; i <= k - 2; i++) {
                        aa = fabs(A[i + j * lda]);
                        work[i] = work[i] + aa;
                        s = s + aa;
                    }
                    aa = fabs(A[i + j * lda]);
                    s = s + aa;
                    work[i] = s;
                    for (j = k + 1; j <= n; j++) {
                        s = ZERO;
                        for (i = 0; i <= k - 1; i++) {
                            aa = fabs(A[i + j * lda]);
                            work[i] = work[i] + aa;
                            s = s + aa;
                        }
                        work[j - 1] = work[j - 1] + s;
                    }
                    value = work[0];
                    for (i = 1; i <= n - 1; i++) {
                        temp = work[i];
                        if (value < temp || disnan(temp)) {
                            value = temp;
                        }
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
                        dlassq(k - j - 2, &A[k + j + 1 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 1; j++) {
                        dlassq(k + j - 1, &A[0 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k - 1, &A[k], lda + 1, &scale, &s);
                    dlassq(k, &A[k - 1], lda + 1, &scale, &s);
                } else {
                    for (j = 0; j <= k - 1; j++) {
                        dlassq(n - j - 1, &A[j + 1 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        dlassq(j, &A[0 + (1 + j) * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k, &A[0], lda + 1, &scale, &s);
                    dlassq(k - 1, &A[0 + lda], lda + 1, &scale, &s);
                }
            } else {
                if (ilu == 0) {
                    for (j = 1; j <= k - 2; j++) {
                        dlassq(j, &A[0 + (k + j) * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        dlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        dlassq(k - j - 1, &A[j + 1 + (j + k - 1) * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k - 1, &A[0 + k * lda], lda + 1, &scale, &s);
                    dlassq(k, &A[0 + (k - 1) * lda], lda + 1, &scale, &s);
                } else {
                    for (j = 1; j <= k - 1; j++) {
                        dlassq(j, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = k; j <= n - 1; j++) {
                        dlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 3; j++) {
                        dlassq(k - j - 2, &A[j + 2 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k, &A[0], lda + 1, &scale, &s);
                    dlassq(k - 1, &A[1], lda + 1, &scale, &s);
                }
            }
        } else {
            if (ifm == 1) {
                if (ilu == 0) {
                    for (j = 0; j <= k - 2; j++) {
                        dlassq(k - j - 1, &A[k + j + 2 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 1; j++) {
                        dlassq(k + j, &A[0 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k, &A[k + 1], lda + 1, &scale, &s);
                    dlassq(k, &A[k], lda + 1, &scale, &s);
                } else {
                    for (j = 0; j <= k - 1; j++) {
                        dlassq(n - j - 1, &A[j + 2 + j * lda], 1, &scale, &s);
                    }
                    for (j = 1; j <= k - 1; j++) {
                        dlassq(j, &A[0 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k, &A[1], lda + 1, &scale, &s);
                    dlassq(k, &A[0], lda + 1, &scale, &s);
                }
            } else {
                if (ilu == 0) {
                    for (j = 1; j <= k - 1; j++) {
                        dlassq(j, &A[0 + (k + 1 + j) * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 1; j++) {
                        dlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        dlassq(k - j - 1, &A[j + 1 + (j + k) * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k, &A[0 + (k + 1) * lda], lda + 1, &scale, &s);
                    dlassq(k, &A[0 + k * lda], lda + 1, &scale, &s);
                } else {
                    for (j = 1; j <= k - 1; j++) {
                        dlassq(j, &A[0 + (j + 1) * lda], 1, &scale, &s);
                    }
                    for (j = k + 1; j <= n; j++) {
                        dlassq(k, &A[0 + j * lda], 1, &scale, &s);
                    }
                    for (j = 0; j <= k - 2; j++) {
                        dlassq(k - j - 1, &A[j + 1 + j * lda], 1, &scale, &s);
                    }
                    s = s + s;
                    dlassq(k, &A[lda], lda + 1, &scale, &s);
                    dlassq(k, &A[0], lda + 1, &scale, &s);
                }
            }
        }
        value = scale * sqrt(s);
    } else {
        value = ZERO;
    }

    return value;
}
