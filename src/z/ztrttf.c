/**
 * @file ztrttf.c
 * @brief ZTRTTF copies a triangular matrix from standard full format (TR) to rectangular full packed format (TF).
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZTRTTF copies a triangular matrix A from standard full format (TR)
 * to rectangular full packed format (TF).
 *
 * @param[in] transr
 *          = 'N':  ARF in Normal mode is wanted;
 *          = 'C':  ARF in Conjugate Transpose mode is wanted.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] A
 *          Double complex array, dimension (lda,n).
 *          On entry, the triangular matrix A.  If UPLO = 'U', the
 *          leading n-by-n upper triangular part of the array A contains
 *          the upper triangular matrix, and the strictly lower
 *          triangular part of A is not referenced.  If UPLO = 'L', the
 *          leading n-by-n lower triangular part of the array A contains
 *          the lower triangular matrix, and the strictly upper
 *          triangular part of A is not referenced.
 *
 * @param[in] lda
 *          The leading dimension of the matrix A.  lda >= max(1,n).
 *
 * @param[out] ARF
 *          Double complex array, dimension (n*(n+1)/2).
 *          On exit, the triangular matrix A in RFP format.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void ztrttf(
    const char* transr,
    const char* uplo,
    const INT n,
    const c128* restrict A,
    const INT lda,
    c128* restrict ARF,
    INT* info)
{
    INT lower, nisodd, normaltransr;
    INT i, j, k, l, n1, n2, nt, nx2, np1x2;
    INT ij;

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
        *info = -5;
    }
    if (*info != 0) {
        xerbla("ZTRTTF", -(*info));
        return;
    }

    if (n <= 1) {
        if (n == 1) {
            if (normaltransr) {
                ARF[0] = A[0];
            } else {
                ARF[0] = conj(A[0]);
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
                        ARF[ij] = conj(A[(n2 + j) + i * lda]);
                        ij++;
                    }
                    for (i = j; i <= n - 1; i++) {
                        ARF[ij] = A[i + j * lda];
                        ij++;
                    }
                }

            } else {

                ij = nt - n;
                for (j = n - 1; j >= n1; j--) {
                    for (i = 0; i <= j; i++) {
                        ARF[ij] = A[i + j * lda];
                        ij++;
                    }
                    for (l = j - n1; l <= n1 - 1; l++) {
                        ARF[ij] = conj(A[(j - n1) + l * lda]);
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
                        ARF[ij] = conj(A[j + i * lda]);
                        ij++;
                    }
                    for (i = n1 + j; i <= n - 1; i++) {
                        ARF[ij] = A[i + (n1 + j) * lda];
                        ij++;
                    }
                }
                for (j = n2; j <= n - 1; j++) {
                    for (i = 0; i <= n1 - 1; i++) {
                        ARF[ij] = conj(A[j + i * lda]);
                        ij++;
                    }
                }

            } else {

                ij = 0;
                for (j = 0; j <= n1; j++) {
                    for (i = n1; i <= n - 1; i++) {
                        ARF[ij] = conj(A[j + i * lda]);
                        ij++;
                    }
                }
                for (j = 0; j <= n1 - 1; j++) {
                    for (i = 0; i <= j; i++) {
                        ARF[ij] = A[i + j * lda];
                        ij++;
                    }
                    for (l = n2 + j; l <= n - 1; l++) {
                        ARF[ij] = conj(A[(n2 + j) + l * lda]);
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
                        ARF[ij] = conj(A[(k + j) + i * lda]);
                        ij++;
                    }
                    for (i = j; i <= n - 1; i++) {
                        ARF[ij] = A[i + j * lda];
                        ij++;
                    }
                }

            } else {

                ij = nt - n - 1;
                for (j = n - 1; j >= k; j--) {
                    for (i = 0; i <= j; i++) {
                        ARF[ij] = A[i + j * lda];
                        ij++;
                    }
                    for (l = j - k; l <= k - 1; l++) {
                        ARF[ij] = conj(A[(j - k) + l * lda]);
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
                    ARF[ij] = A[i + j * lda];
                    ij++;
                }
                for (j = 0; j <= k - 2; j++) {
                    for (i = 0; i <= j; i++) {
                        ARF[ij] = conj(A[j + i * lda]);
                        ij++;
                    }
                    for (i = k + 1 + j; i <= n - 1; i++) {
                        ARF[ij] = A[i + (k + 1 + j) * lda];
                        ij++;
                    }
                }
                for (j = k - 1; j <= n - 1; j++) {
                    for (i = 0; i <= k - 1; i++) {
                        ARF[ij] = conj(A[j + i * lda]);
                        ij++;
                    }
                }

            } else {

                ij = 0;
                for (j = 0; j <= k; j++) {
                    for (i = k; i <= n - 1; i++) {
                        ARF[ij] = conj(A[j + i * lda]);
                        ij++;
                    }
                }
                for (j = 0; j <= k - 2; j++) {
                    for (i = 0; i <= j; i++) {
                        ARF[ij] = A[i + j * lda];
                        ij++;
                    }
                    for (l = k + 1 + j; l <= n - 1; l++) {
                        ARF[ij] = conj(A[(k + 1 + j) + l * lda]);
                        ij++;
                    }
                }
                for (i = 0; i <= j; i++) {
                    ARF[ij] = A[i + j * lda];
                    ij++;
                }

            }

        }

    }
}
