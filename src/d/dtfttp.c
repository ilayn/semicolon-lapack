/**
 * @file dtfttp.c
 * @brief DTFTTP copies a triangular matrix from rectangular full packed format (TF) to standard packed format (TP).
 */

#include "semicolon_lapack_double.h"

/**
 * DTFTTP copies a triangular matrix A from rectangular full packed
 * format (TF) to standard packed format (TP).
 *
 * @param[in] transr
 *          = 'N':  ARF is in Normal format;
 *          = 'T':  ARF is in Transpose format;
 *
 * @param[in] uplo
 *          = 'U':  A is upper triangular;
 *          = 'L':  A is lower triangular.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] ARF
 *          Double precision array, dimension ( n*(n+1)/2 ),
 *          On entry, the upper or lower triangular matrix A stored in
 *          RFP format. For a further discussion see Notes below.
 *
 * @param[out] AP
 *          Double precision array, dimension ( n*(n+1)/2 ),
 *          On exit, the upper or lower triangular matrix A, packed
 *          columnwise in a linear array. The j-th column of A is stored
 *          in the array AP as follows:
 *          if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for 1<=i<=j;
 *          if UPLO = 'L', AP(i + (j-1)*(2n-j)/2) = A(i,j) for j<=i<=n.
 *
 * @param[out] info
 *          = 0:  successful exit
 *          < 0:  if info = -i, the i-th argument had an illegal value
 */
void dtfttp(
    const char* transr,
    const char* uplo,
    const int n,
    const double* const restrict ARF,
    double* const restrict AP,
    int* info)
{
    int lower, nisodd, normaltransr;
    int n1, n2, k;
    int i, j, ij;
    int ijp, jp, lda, js;

    *info = 0;
    normaltransr = (transr[0] == 'N' || transr[0] == 'n');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    if (!normaltransr && !(transr[0] == 'T' || transr[0] == 't')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    }
    if (*info != 0) {
        xerbla("DTFTTP", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        /* LAPACK has identical branches */
        AP[0] = ARF[0];
        return;
    }

    (void)(n * (n + 1) / 2);  /* nt computed in Fortran but unused */

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
        lda = n + 1;
    } else {
        nisodd = 1;
        lda = n;
    }

    if (!normaltransr) {
        lda = (n + 1) / 2;
    }

    if (nisodd) {

        if (normaltransr) {

            if (lower) {

                ijp = 0;
                jp = 0;
                for (j = 0; j <= n2; j++) {
                    for (i = j; i <= n - 1; i++) {
                        ij = i + jp;
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    jp += lda;
                }
                for (i = 0; i <= n2 - 1; i++) {
                    for (j = 1 + i; j <= n2; j++) {
                        ij = i + j * lda;
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                }

            } else {

                ijp = 0;
                for (j = 0; j <= n1 - 1; j++) {
                    ij = n2 + j;
                    for (i = 0; i <= j; i++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                        ij += lda;
                    }
                }
                js = 0;
                for (j = n1; j <= n - 1; j++) {
                    ij = js;
                    for (ij = js; ij <= js + j; ij++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    js += lda;
                }

            }

        } else {

            if (lower) {

                ijp = 0;
                for (i = 0; i <= n2; i++) {
                    for (ij = i * (lda + 1); ij <= n * lda - 1; ij += lda) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                }
                js = 1;
                for (j = 0; j <= n2 - 1; j++) {
                    for (ij = js; ij <= js + n2 - j - 1; ij++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    js += lda + 1;
                }

            } else {

                ijp = 0;
                js = n2 * lda;
                for (j = 0; j <= n1 - 1; j++) {
                    for (ij = js; ij <= js + j; ij++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    js += lda;
                }
                for (i = 0; i <= n1; i++) {
                    for (ij = i; ij <= i + (n1 + i) * lda; ij += lda) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                }

            }

        }

    } else {

        if (normaltransr) {

            if (lower) {

                ijp = 0;
                jp = 0;
                for (j = 0; j <= k - 1; j++) {
                    for (i = j; i <= n - 1; i++) {
                        ij = 1 + i + jp;
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    jp += lda;
                }
                for (i = 0; i <= k - 1; i++) {
                    for (j = i; j <= k - 1; j++) {
                        ij = i + j * lda;
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                }

            } else {

                ijp = 0;
                for (j = 0; j <= k - 1; j++) {
                    ij = k + 1 + j;
                    for (i = 0; i <= j; i++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                        ij += lda;
                    }
                }
                js = 0;
                for (j = k; j <= n - 1; j++) {
                    ij = js;
                    for (ij = js; ij <= js + j; ij++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    js += lda;
                }

            }

        } else {

            if (lower) {

                ijp = 0;
                for (i = 0; i <= k - 1; i++) {
                    for (ij = i + (i + 1) * lda; ij <= (n + 1) * lda - 1; ij += lda) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                }
                js = 0;
                for (j = 0; j <= k - 1; j++) {
                    for (ij = js; ij <= js + k - j - 1; ij++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    js += lda + 1;
                }

            } else {

                ijp = 0;
                js = (k + 1) * lda;
                for (j = 0; j <= k - 1; j++) {
                    for (ij = js; ij <= js + j; ij++) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                    js += lda;
                }
                for (i = 0; i <= k - 1; i++) {
                    for (ij = i; ij <= i + (k + i) * lda; ij += lda) {
                        AP[ijp] = ARF[ij];
                        ijp++;
                    }
                }

            }

        }

    }
}
