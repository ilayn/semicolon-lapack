/**
 * @file zsytrf_aa.c
 * @brief ZSYTRF_AA computes the factorization of a complex symmetric matrix using Aasen's algorithm.
 */

#include <cblas.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZSYTRF_AA computes the factorization of a complex symmetric matrix A
 * using the Aasen's algorithm. The form of the factorization is
 *
 *    A = U**T*T*U  or  A = L*T*L**T
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and T is a complex symmetric tridiagonal matrix.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double complex array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, the tridiagonal matrix is stored in the diagonals
 *          and the subdiagonals of A just below (or above) the diagonals,
 *          and L is stored below (or above) the subdiagonals.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          On exit, it contains the details of the interchanges.
 *
 * @param[out] work
 *          Double complex array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The length of work.
 *          lwork >= 1, if n <= 1, and lwork >= 2*n, otherwise.
 *          For optimum performance lwork >= n*(1+nb).
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value.
 */
void zsytrf_aa(
    const char* uplo,
    const int n,
    double complex* const restrict A,
    const int lda,
    int* restrict ipiv,
    double complex* restrict work,
    const int lwork,
    int* info)
{
    int upper, lquery;
    int j, lwkmin, lwkopt;
    int nb, mj, nj, k1, k2, j1, j2, j3, jb;
    double complex alpha;

    const double complex ONE = CMPLX(1.0, 0.0);
    const double complex NEG_ONE = CMPLX(-1.0, 0.0);

    nb = lapack_get_nb("SYTRF");

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    if (n <= 1) {
        lwkmin = 1;
        lwkopt = 1;
    } else {
        lwkmin = 2 * n;
        lwkopt = (nb + 1) * n;
    }

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (lwork < lwkmin && !lquery) {
        *info = -7;
    }

    if (*info == 0) {
        work[0] = (double complex)lwkopt;
    }

    if (*info != 0) {
        xerbla("ZSYTRF_AA", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }
    ipiv[0] = 0;
    if (n == 1) {
        return;
    }

    if (lwork < (1 + nb) * n) {
        nb = (lwork - n) / n;
    }

    if (upper) {

        cblas_zcopy(n, &A[0 + 0 * lda], lda, &work[0], 1);

        j = 0;
        while (j < n) {

            j1 = j;
            jb = ((n - j1) < nb) ? (n - j1) : nb;
            k1 = ((1 > j) ? 1 : j) - j - 1;

            zlasyf_aa(uplo, 2 - k1 - 1, n - j, jb,
                      &A[((0 > j - 1) ? 0 : j - 1) + j * lda], lda,
                      &ipiv[j], work, n, &work[n * nb]);

            for (j2 = j + 1; j2 < ((n < j + jb + 1) ? n : j + jb + 1); j2++) {
                ipiv[j2] = ipiv[j2] + j;
                if ((j2 != ipiv[j2]) && ((j1 - k1) > 2)) {
                    cblas_zswap(j1 - k1 - 2, &A[0 + j2 * lda], 1,
                                &A[0 + ipiv[j2] * lda], 1);
                }
            }
            j = j + jb;

            if (j < n) {

                if (j1 > 0 || jb > 1) {

                    alpha = A[(j - 1) + j * lda];
                    A[(j - 1) + j * lda] = ONE;
                    cblas_zcopy(n - j, &A[(j - 2) + j * lda], lda,
                                &work[(j - j1) + jb * n], 1);
                    cblas_zscal(n - j, &alpha, &work[(j - j1) + jb * n], 1);

                    if (j1 > 0) {
                        k2 = 1;
                    } else {
                        k2 = 0;
                        jb = jb - 1;
                    }

                    for (j2 = j; j2 < n; j2 += nb) {
                        nj = ((nb < n - j2) ? nb : n - j2);

                        j3 = j2;
                        for (mj = nj - 1; mj >= 1; mj--) {
                            cblas_zgemv(CblasColMajor, CblasNoTrans, mj, jb + 1,
                                        &NEG_ONE, &work[(j3 - j1) + (k1 + 1) * n], n,
                                        &A[(j1 - k2) + j3 * lda], 1,
                                        &ONE, &A[j3 + j3 * lda], lda);
                            j3 = j3 + 1;
                        }

                        cblas_zgemm(CblasColMajor, CblasTrans, CblasTrans,
                                    nj, n - j3, jb + 1,
                                    &NEG_ONE, &A[(j1 - k2) + j2 * lda], lda,
                                    &work[(j3 - j1) + (k1 + 1) * n], n,
                                    &ONE, &A[j2 + j3 * lda], lda);
                    }

                    A[(j - 1) + j * lda] = alpha;
                }

                cblas_zcopy(n - j, &A[j + j * lda], lda, &work[0], 1);
            }
        }

    } else {

        cblas_zcopy(n, &A[0 + 0 * lda], 1, &work[0], 1);

        j = 0;
        while (j < n) {

            j1 = j;
            jb = ((n - j1) < nb) ? (n - j1) : nb;
            k1 = ((1 > j) ? 1 : j) - j - 1;

            zlasyf_aa(uplo, 2 - k1 - 1, n - j, jb,
                      &A[j + ((0 > j - 1) ? 0 : j - 1) * lda], lda,
                      &ipiv[j], work, n, &work[n * nb]);

            for (j2 = j + 1; j2 < ((n < j + jb + 1) ? n : j + jb + 1); j2++) {
                ipiv[j2] = ipiv[j2] + j;
                if ((j2 != ipiv[j2]) && ((j1 - k1) > 2)) {
                    cblas_zswap(j1 - k1 - 2, &A[j2 + 0 * lda], lda,
                                &A[ipiv[j2] + 0 * lda], lda);
                }
            }
            j = j + jb;

            if (j < n) {

                if (j1 > 0 || jb > 1) {

                    alpha = A[j + (j - 1) * lda];
                    A[j + (j - 1) * lda] = ONE;
                    cblas_zcopy(n - j, &A[j + (j - 2) * lda], 1,
                                &work[(j - j1) + jb * n], 1);
                    cblas_zscal(n - j, &alpha, &work[(j - j1) + jb * n], 1);

                    if (j1 > 0) {
                        k2 = 1;
                    } else {
                        k2 = 0;
                        jb = jb - 1;
                    }

                    for (j2 = j; j2 < n; j2 += nb) {
                        nj = ((nb < n - j2) ? nb : n - j2);

                        j3 = j2;
                        for (mj = nj - 1; mj >= 1; mj--) {
                            cblas_zgemv(CblasColMajor, CblasNoTrans, mj, jb + 1,
                                        &NEG_ONE, &work[(j3 - j1) + (k1 + 1) * n], n,
                                        &A[j3 + (j1 - k2) * lda], lda,
                                        &ONE, &A[j3 + j3 * lda], 1);
                            j3 = j3 + 1;
                        }

                        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    n - j3, nj, jb + 1,
                                    &NEG_ONE, &work[(j3 - j1) + (k1 + 1) * n], n,
                                    &A[j2 + (j1 - k2) * lda], lda,
                                    &ONE, &A[j3 + j2 * lda], lda);
                    }

                    A[j + (j - 1) * lda] = alpha;
                }

                cblas_zcopy(n - j, &A[j + j * lda], 1, &work[0], 1);
            }
        }
    }

    work[0] = (double complex)lwkopt;
}
