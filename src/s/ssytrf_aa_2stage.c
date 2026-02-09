/**
 * @file ssytrf_aa_2stage.c
 * @brief SSYTRF_AA_2STAGE computes the factorization of a real symmetric matrix using Aasen's 2-stage algorithm.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SSYTRF_AA_2STAGE computes the factorization of a real symmetric matrix A
 * using the Aasen's algorithm. The form of the factorization is
 *
 *    A = U**T*T*U  or  A = L*T*L**T
 *
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and T is a symmetric band matrix with the
 * bandwidth of NB (NB is internally selected and stored in TB[0], and T is
 * LU factorized with partial pivoting).
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
 *          Double precision array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, L is stored below (or above) the subdiagonal blocks.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] TB
 *          Double precision array, dimension (max(1, ltb)).
 *          On exit, details of the LU factorization of the band matrix.
 *
 * @param[in] ltb
 *          The size of the array TB. ltb >= max(1, 4*n).
 *          If ltb = -1, then a workspace query is assumed.
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          On exit, details of the interchanges.
 *
 * @param[out] ipiv2
 *          Integer array, dimension (n).
 *          On exit, details of the interchanges in T.
 *
 * @param[out] work
 *          Double precision workspace of size (max(1, lwork)).
 *
 * @param[in] lwork
 *          The size of work. lwork >= max(1, n).
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *          = 0:  successful exit
 *          < 0:  if info = -i, the i-th argument had an illegal value.
 *          > 0:  if info = i, band LU factorization failed on i-th column
 */
void ssytrf_aa_2stage(
    const char* uplo,
    const int n,
    float* const restrict A,
    const int lda,
    float* restrict TB,
    const int ltb,
    int* restrict ipiv,
    int* restrict ipiv2,
    float* restrict work,
    const int lwork,
    int* info)
{
    int upper, tquery, wquery;
    int i, j, k, i1, i2, td;
    int ldtb, nb, kb, jb, nt, iinfo;
    float piv;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    wquery = (lwork == -1);
    tquery = (ltb == -1);

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    } else if (ltb < (1 > 4 * n ? 1 : 4 * n) && !tquery) {
        *info = -6;
    } else if (lwork < (1 > n ? 1 : n) && !wquery) {
        *info = -10;
    }

    if (*info != 0) {
        xerbla("SSYTRF_AA_2STAGE", -(*info));
        return;
    }

    nb = lapack_get_nb("SYTRF");

    if (*info == 0) {
        if (tquery) {
            TB[0] = (float)((1 > (3 * nb + 1) * n) ? 1 : (3 * nb + 1) * n);
        }
        if (wquery) {
            work[0] = (float)((1 > n * nb) ? 1 : n * nb);
        }
    }
    if (tquery || wquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    ldtb = ltb / n;
    if (ldtb < 3 * nb + 1) {
        nb = (ldtb - 1) / 3;
    }
    if (lwork < nb * n) {
        nb = lwork / n;
    }

    nt = (n + nb - 1) / nb;
    td = 2 * nb;
    kb = (nb < n) ? nb : n;

    for (j = 0; j < kb; j++) {
        ipiv[j] = j;
    }

    TB[0] = (float)nb;

    if (upper) {

        for (j = 0; j < nt; j++) {

            kb = (nb < n - j * nb) ? nb : n - j * nb;
            for (i = 1; i <= j - 1; i++) {
                if (i == 1) {
                    if (i == j - 1) {
                        jb = nb + kb;
                    } else {
                        jb = 2 * nb;
                    }
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                nb, kb, jb,
                                1.0f, &TB[td + (i * nb) * ldtb], ldtb - 1,
                                &A[(i - 1) * nb + j * nb * lda], lda,
                                0.0f, &work[i * nb], n);
                } else {
                    if (i == j - 1) {
                        jb = 2 * nb + kb;
                    } else {
                        jb = 3 * nb;
                    }
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                nb, kb, jb,
                                1.0f, &TB[td + nb + ((i - 1) * nb) * ldtb], ldtb - 1,
                                &A[(i - 2) * nb + j * nb * lda], lda,
                                0.0f, &work[i * nb], n);
                }
            }

            slacpy("Upper", kb, kb, &A[j * nb + j * nb * lda], lda,
                   &TB[td + (j * nb) * ldtb], ldtb - 1);
            if (j > 1) {
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            kb, kb, (j - 1) * nb,
                            -1.0f, &A[0 + j * nb * lda], lda,
                            &work[nb], n,
                            1.0f, &TB[td + (j * nb) * ldtb], ldtb - 1);
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            kb, nb, kb,
                            1.0f, &A[(j - 1) * nb + j * nb * lda], lda,
                            &TB[td + nb + ((j - 1) * nb) * ldtb], ldtb - 1,
                            0.0f, &work[0], n);
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            kb, kb, nb,
                            -1.0f, &work[0], n,
                            &A[(j - 2) * nb + j * nb * lda], lda,
                            1.0f, &TB[td + (j * nb) * ldtb], ldtb - 1);
            }
            if (j > 0) {
                ssygst(1, "Upper", kb,
                       &TB[td + (j * nb) * ldtb], ldtb - 1,
                       &A[(j - 1) * nb + j * nb * lda], lda, &iinfo);
            }

            for (i = 0; i < kb; i++) {
                for (k = i + 1; k < kb; k++) {
                    TB[td + (k - i) + (j * nb + i) * ldtb] =
                        TB[td - (k - i) + (j * nb + k) * ldtb];
                }
            }

            if (j < nt - 1) {
                if (j > 0) {

                    if (j == 1) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    kb, kb, kb,
                                    1.0f, &TB[td + (j * nb) * ldtb], ldtb - 1,
                                    &A[(j - 1) * nb + j * nb * lda], lda,
                                    0.0f, &work[j * nb], n);
                    } else {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    kb, kb, nb + kb,
                                    1.0f, &TB[td + nb + ((j - 1) * nb) * ldtb], ldtb - 1,
                                    &A[(j - 2) * nb + j * nb * lda], lda,
                                    0.0f, &work[j * nb], n);
                    }

                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                                nb, n - (j + 1) * nb, j * nb,
                                -1.0f, &work[nb], n,
                                &A[0 + (j + 1) * nb * lda], lda,
                                1.0f, &A[j * nb + (j + 1) * nb * lda], lda);
                }

                for (k = 0; k < nb; k++) {
                    cblas_scopy(n - (j + 1) * nb,
                                &A[j * nb + k + (j + 1) * nb * lda], lda,
                                &work[0 + k * n], 1);
                }

                sgetrf(n - (j + 1) * nb, nb,
                       work, n,
                       &ipiv[(j + 1) * nb], &iinfo);

                for (k = 0; k < nb; k++) {
                    cblas_scopy(n - (j + 1) * nb,
                                &work[0 + k * n], 1,
                                &A[j * nb + k + (j + 1) * nb * lda], lda);
                }

                kb = (nb < n - (j + 1) * nb) ? nb : n - (j + 1) * nb;
                slaset("Full", kb, nb, 0.0f, 0.0f,
                       &TB[td + nb + (j * nb) * ldtb], ldtb - 1);
                slacpy("Upper", kb, nb,
                       work, n,
                       &TB[td + nb + (j * nb) * ldtb], ldtb - 1);
                if (j > 0) {
                    cblas_strsm(CblasColMajor, CblasRight, CblasUpper,
                                CblasNoTrans, CblasUnit, kb, nb, 1.0f,
                                &A[(j - 1) * nb + j * nb * lda], lda,
                                &TB[td + nb + (j * nb) * ldtb], ldtb - 1);
                }

                for (k = 0; k < nb; k++) {
                    for (i = 0; i < kb; i++) {
                        TB[td - nb + k - i + (j * nb + nb + i) * ldtb] =
                            TB[td + nb + i - k + (j * nb + k) * ldtb];
                    }
                }
                slaset("Lower", kb, nb, 0.0f, 1.0f,
                       &A[j * nb + (j + 1) * nb * lda], lda);

                for (k = 0; k < kb; k++) {
                    ipiv[(j + 1) * nb + k] = ipiv[(j + 1) * nb + k] + (j + 1) * nb;

                    i1 = (j + 1) * nb + k;
                    i2 = ipiv[(j + 1) * nb + k];
                    if (i1 != i2) {
                        cblas_sswap(k, &A[(j + 1) * nb + i1 * lda], 1,
                                    &A[(j + 1) * nb + i2 * lda], 1);
                        if (i2 > i1 + 1) {
                            cblas_sswap(i2 - i1 - 1, &A[i1 + (i1 + 1) * lda], lda,
                                        &A[i1 + 1 + i2 * lda], 1);
                        }
                        if (i2 < n - 1) {
                            cblas_sswap(n - i2 - 1, &A[i1 + (i2 + 1) * lda], lda,
                                        &A[i2 + (i2 + 1) * lda], lda);
                        }
                        piv = A[i1 + i1 * lda];
                        A[i1 + i1 * lda] = A[i2 + i2 * lda];
                        A[i2 + i2 * lda] = piv;
                        if (j > 0) {
                            cblas_sswap(j * nb, &A[0 + i1 * lda], 1,
                                        &A[0 + i2 * lda], 1);
                        }
                    }
                }
            }
        }

    } else {

        for (j = 0; j < nt; j++) {

            kb = (nb < n - j * nb) ? nb : n - j * nb;
            for (i = 1; i <= j - 1; i++) {
                if (i == 1) {
                    if (i == j - 1) {
                        jb = nb + kb;
                    } else {
                        jb = 2 * nb;
                    }
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                nb, kb, jb,
                                1.0f, &TB[td + (i * nb) * ldtb], ldtb - 1,
                                &A[j * nb + (i - 1) * nb * lda], lda,
                                0.0f, &work[i * nb], n);
                } else {
                    if (i == j - 1) {
                        jb = 2 * nb + kb;
                    } else {
                        jb = 3 * nb;
                    }
                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                nb, kb, jb,
                                1.0f, &TB[td + nb + ((i - 1) * nb) * ldtb], ldtb - 1,
                                &A[j * nb + (i - 2) * nb * lda], lda,
                                0.0f, &work[i * nb], n);
                }
            }

            slacpy("Lower", kb, kb, &A[j * nb + j * nb * lda], lda,
                   &TB[td + (j * nb) * ldtb], ldtb - 1);
            if (j > 1) {
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            kb, kb, (j - 1) * nb,
                            -1.0f, &A[j * nb + 0 * lda], lda,
                            &work[nb], n,
                            1.0f, &TB[td + (j * nb) * ldtb], ldtb - 1);
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                            kb, nb, kb,
                            1.0f, &A[j * nb + (j - 1) * nb * lda], lda,
                            &TB[td + nb + ((j - 1) * nb) * ldtb], ldtb - 1,
                            0.0f, &work[0], n);
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                            kb, kb, nb,
                            -1.0f, &work[0], n,
                            &A[j * nb + (j - 2) * nb * lda], lda,
                            1.0f, &TB[td + (j * nb) * ldtb], ldtb - 1);
            }
            if (j > 0) {
                ssygst(1, "Lower", kb,
                       &TB[td + (j * nb) * ldtb], ldtb - 1,
                       &A[j * nb + (j - 1) * nb * lda], lda, &iinfo);
            }

            for (i = 0; i < kb; i++) {
                for (k = i + 1; k < kb; k++) {
                    TB[td - (k - i) + (j * nb + k) * ldtb] =
                        TB[td + (k - i) + (j * nb + i) * ldtb];
                }
            }

            if (j < nt - 1) {
                if (j > 0) {

                    if (j == 1) {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    kb, kb, kb,
                                    1.0f, &TB[td + (j * nb) * ldtb], ldtb - 1,
                                    &A[j * nb + (j - 1) * nb * lda], lda,
                                    0.0f, &work[j * nb], n);
                    } else {
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                                    kb, kb, nb + kb,
                                    1.0f, &TB[td + nb + ((j - 1) * nb) * ldtb], ldtb - 1,
                                    &A[j * nb + (j - 2) * nb * lda], lda,
                                    0.0f, &work[j * nb], n);
                    }

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                n - (j + 1) * nb, nb, j * nb,
                                -1.0f, &A[(j + 1) * nb + 0 * lda], lda,
                                &work[nb], n,
                                1.0f, &A[(j + 1) * nb + j * nb * lda], lda);
                }

                sgetrf(n - (j + 1) * nb, nb,
                       &A[(j + 1) * nb + j * nb * lda], lda,
                       &ipiv[(j + 1) * nb], &iinfo);

                kb = (nb < n - (j + 1) * nb) ? nb : n - (j + 1) * nb;
                slaset("Full", kb, nb, 0.0f, 0.0f,
                       &TB[td + nb + (j * nb) * ldtb], ldtb - 1);
                slacpy("Upper", kb, nb,
                       &A[(j + 1) * nb + j * nb * lda], lda,
                       &TB[td + nb + (j * nb) * ldtb], ldtb - 1);
                if (j > 0) {
                    cblas_strsm(CblasColMajor, CblasRight, CblasLower,
                                CblasTrans, CblasUnit, kb, nb, 1.0f,
                                &A[j * nb + (j - 1) * nb * lda], lda,
                                &TB[td + nb + (j * nb) * ldtb], ldtb - 1);
                }

                for (k = 0; k < nb; k++) {
                    for (i = 0; i < kb; i++) {
                        TB[td - nb + k - i + (j * nb + nb + i) * ldtb] =
                            TB[td + nb + i - k + (j * nb + k) * ldtb];
                    }
                }
                slaset("Upper", kb, nb, 0.0f, 1.0f,
                       &A[(j + 1) * nb + j * nb * lda], lda);

                for (k = 0; k < kb; k++) {
                    ipiv[(j + 1) * nb + k] = ipiv[(j + 1) * nb + k] + (j + 1) * nb;

                    i1 = (j + 1) * nb + k;
                    i2 = ipiv[(j + 1) * nb + k];
                    if (i1 != i2) {
                        cblas_sswap(k, &A[i1 + (j + 1) * nb * lda], lda,
                                    &A[i2 + (j + 1) * nb * lda], lda);
                        if (i2 > i1 + 1) {
                            cblas_sswap(i2 - i1 - 1, &A[i1 + 1 + i1 * lda], 1,
                                        &A[i2 + (i1 + 1) * lda], lda);
                        }
                        if (i2 < n - 1) {
                            cblas_sswap(n - i2 - 1, &A[i2 + 1 + i1 * lda], 1,
                                        &A[i2 + 1 + i2 * lda], 1);
                        }
                        piv = A[i1 + i1 * lda];
                        A[i1 + i1 * lda] = A[i2 + i2 * lda];
                        A[i2 + i2 * lda] = piv;
                        if (j > 0) {
                            cblas_sswap(j * nb, &A[i1 + 0 * lda], lda,
                                        &A[i2 + 0 * lda], lda);
                        }
                    }
                }
            }
        }
    }

    sgbtrf(n, n, nb, nb, TB, ldtb, ipiv2, info);
}
