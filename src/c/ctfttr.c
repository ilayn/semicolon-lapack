/**
 * @file ctfttr.c
 * @brief CTFTTR copies a triangular matrix from rectangular full packed format (TF) to standard full format (TR).
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CTFTTR copies a triangular matrix A from rectangular full packed
 * format (TF) to standard full format (TR).
 *
 * @param[in] transr
 *          = 'N':  ARF is in Normal format;
 *          = 'C':  ARF is in Conjugate-transpose format.
 *
 * @param[in] uplo
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 * @param[in] n
 *          The order of the matrices ARF and A. n >= 0.
 *
 * @param[in] ARF
 *          Single complex array, dimension (n*(n+1)/2).
 *          On entry, the upper (if UPLO = 'U') or lower (if UPLO = 'L')
 *          matrix A in RFP format. See the "Notes" below for more
 *          details.
 *
 * @param[out] A
 *          Single complex array, dimension (lda,n)
 *          On exit, the triangular matrix A.  If UPLO = 'U', the
 *          leading n-by-n upper triangular part of the array A contains
 *          the upper triangular matrix, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading n-by-n lower triangular part of the array A contains
 *          the lower triangular matrix, and the strictly upper
 *          triangular part of A is not referenced.
 *
 * @param[in] lda
 *          The leading dimension of the array A.  lda >= max(1,n).
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void ctfttr(
    const char* transr,
    const char* uplo,
    const INT n,
    const c64* restrict ARF,
    c64* restrict A,
    const INT lda,
    INT* info)
{
    INT lower, nisodd, normaltransr;
    INT n1, n2, k, nt, nx2, np1x2;
    INT i, j, l, ij;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    if (!normaltransr && !(transr[0] == 'C' || transr[0] == 'c')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("CTFTTR", -(*info));
        return;
    }

    if (n <= 1) {
        if (n == 1) {
            if (normaltransr) {
                A[0] = ARF[0];
            } else {
                A[0] = conjf(ARF[0]);
            }
        }
        return;
    }

    nt = n * (n + 1) / 2;

    if (lower) {
        n2 = n / 2;
        n1 = n - n2;
    } else {
        n1 = n / 2;
        n2 = n - n1;
    }

    if (n % 2 == 0) {
        k = n / 2;
        nisodd = 0;
        if (!lower) {
            np1x2 = n + n + 2;
        }
    } else {
        nisodd = 1;
        if (!lower) {
            nx2 = n + n;
        }
    }

    if (nisodd) {

        if (normaltransr) {

            if (lower) {

                ij = 0;
                for (j = 0; j <= n2; j++) {
                    for (i = n1; i <= n2 + j; i++) {
                        A[(n2 + j) + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                    for (i = j; i <= n - 1; i++) {
                        A[i + j * lda] = ARF[ij];
                        ij++;
                    }
                }

            } else {

                ij = nt - n;
                for (j = n - 1; j >= n1; j--) {
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = ARF[ij];
                        ij++;
                    }
                    for (l = j - n1; l <= n1 - 1; l++) {
                        A[(j - n1) + l * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                    ij -= nx2;
                }

            }

        } else {

            if (lower) {

                ij = 0;
                for (j = 0; j <= n2 - 1; j++) {
                    for (i = 0; i <= j; i++) {
                        A[j + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                    for (i = n1 + j; i <= n - 1; i++) {
                        A[i + (n1 + j) * lda] = ARF[ij];
                        ij++;
                    }
                }
                for (j = n2; j <= n - 1; j++) {
                    for (i = 0; i <= n1 - 1; i++) {
                        A[j + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                }

            } else {

                ij = 0;
                for (j = 0; j <= n1; j++) {
                    for (i = n1; i <= n - 1; i++) {
                        A[j + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                }
                for (j = 0; j <= n1 - 1; j++) {
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = ARF[ij];
                        ij++;
                    }
                    for (l = n2 + j; l <= n - 1; l++) {
                        A[(n2 + j) + l * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                }

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                ij = 0;
                for (j = 0; j <= k - 1; j++) {
                    for (i = k; i <= k + j; i++) {
                        A[(k + j) + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                    for (i = j; i <= n - 1; i++) {
                        A[i + j * lda] = ARF[ij];
                        ij++;
                    }
                }

            } else {

                ij = nt - n - 1;
                for (j = n - 1; j >= k; j--) {
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = ARF[ij];
                        ij++;
                    }
                    for (l = j - k; l <= k - 1; l++) {
                        A[(j - k) + l * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                    ij -= np1x2;
                }

            }

        } else {

            if (lower) {

                ij = 0;
                j = k;
                for (i = k; i <= n - 1; i++) {
                    A[i + j * lda] = ARF[ij];
                    ij++;
                }
                for (j = 0; j <= k - 2; j++) {
                    for (i = 0; i <= j; i++) {
                        A[j + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                    for (i = k + 1 + j; i <= n - 1; i++) {
                        A[i + (k + 1 + j) * lda] = ARF[ij];
                        ij++;
                    }
                }
                for (j = k - 1; j <= n - 1; j++) {
                    for (i = 0; i <= k - 1; i++) {
                        A[j + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                }

            } else {

                ij = 0;
                for (j = 0; j <= k; j++) {
                    for (i = k; i <= n - 1; i++) {
                        A[j + i * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                }
                for (j = 0; j <= k - 2; j++) {
                    for (i = 0; i <= j; i++) {
                        A[i + j * lda] = ARF[ij];
                        ij++;
                    }
                    for (l = k + 1 + j; l <= n - 1; l++) {
                        A[(k + 1 + j) + l * lda] = conjf(ARF[ij]);
                        ij++;
                    }
                }
                for (i = 0; i <= j; i++) {
                    A[i + j * lda] = ARF[ij];
                    ij++;
                }

            }

        }

    }
}
