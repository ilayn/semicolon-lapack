/**
 * @file ssytri_3x.c
 * @brief SSYTRI_3X computes the inverse of a real symmetric indefinite matrix using the factorization computed by SSYTRF_RK or DSYTRF_BK (blocked algorithm).
 */

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSYTRI_3X computes the inverse of a real symmetric indefinite
 * matrix A using the factorization computed by SSYTRF_RK or DSYTRF_BK:
 *
 *     A = P*U*D*(U**T)*(P**T) or A = P*L*D*(L**T)*(P**T),
 *
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *          Specifies whether the details of the factorization are
 *          stored as an upper or lower triangular matrix.
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, diagonal of the block diagonal matrix D and
 *          factors U or L as computed by SSYTRF_RK and DSYTRF_BK.
 *          On exit, if info = 0, the symmetric inverse of the original
 *          matrix.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[in] E
 *          Double precision array, dimension (n).
 *          Contains the superdiagonal (or subdiagonal) elements of the
 *          symmetric block diagonal matrix D.
 *
 * @param[in] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D.
 *
 * @param[out] work
 *          Double precision array, dimension (n+nb+1, nb+3).
 *
 * @param[in] nb
 *          Block size.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) = 0; the matrix is singular.
 */
void ssytri_3x(
    const char* uplo,
    const int n,
    float* const restrict A,
    const int lda,
    const float* restrict E,
    const int* restrict ipiv,
    float* restrict work,
    const int nb,
    int* info)
{
    const float ONE = 1.0f;
    const float ZERO = 0.0f;

    int upper;
    int cut, i, icount, invd, ip, k, nnb, j, u11;
    float ak, akkp1, akp1, d, t, u01_i_j, u01_ip1_j, u11_i_j, u11_ip1_j;
    int ldwork = n + nb + 1;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -4;
    }

    if (*info != 0) {
        xerbla("SSYTRI_3X", -(*info));
        return;
    }
    if (n == 0) {
        return;
    }

    for (k = 0; k < n; k++) {
        work[k + 0 * ldwork] = E[k];
    }

    if (upper) {

        for (*info = n - 1; *info >= 0; (*info)--) {
            if (ipiv[*info] > 0 && A[*info + (*info) * lda] == ZERO) {
                (*info)++;
                return;
            }
        }
    } else {

        for (*info = 0; *info < n; (*info)++) {
            if (ipiv[*info] > 0 && A[*info + (*info) * lda] == ZERO) {
                (*info)++;
                return;
            }
        }
    }

    *info = 0;

    u11 = n;

    invd = nb + 1;

    if (upper) {

        strtri(uplo, "U", n, A, lda, info);

        k = 0;
        while (k < n) {
            if (ipiv[k] > 0) {
                work[k + invd * ldwork] = ONE / A[k + k * lda];
                work[k + (invd + 1) * ldwork] = ZERO;
            } else {
                t = work[k + 1 + 0 * ldwork];
                ak = A[k + k * lda] / t;
                akp1 = A[k + 1 + (k + 1) * lda] / t;
                akkp1 = work[k + 1 + 0 * ldwork] / t;
                d = t * (ak * akp1 - ONE);
                work[k + invd * ldwork] = akp1 / d;
                work[k + 1 + (invd + 1) * ldwork] = ak / d;
                work[k + (invd + 1) * ldwork] = -akkp1 / d;
                work[k + 1 + invd * ldwork] = work[k + (invd + 1) * ldwork];
                k = k + 1;
            }
            k = k + 1;
        }

        cut = n;
        while (cut > 0) {
            nnb = nb;
            if (cut <= nnb) {
                nnb = cut;
            } else {
                icount = 0;
                for (i = cut - nnb; i < cut; i++) {
                    if (ipiv[i] < 0) {
                        icount = icount + 1;
                    }
                }
                if (icount % 2 == 1) {
                    nnb = nnb + 1;
                }
            }

            cut = cut - nnb;

            for (i = 0; i < cut; i++) {
                for (j = 0; j < nnb; j++) {
                    work[i + j * ldwork] = A[i + (cut + j) * lda];
                }
            }

            for (i = 0; i < nnb; i++) {
                work[u11 + i + i * ldwork] = ONE;
                for (j = 0; j < i; j++) {
                    work[u11 + i + j * ldwork] = ZERO;
                }
                for (j = i + 1; j < nnb; j++) {
                    work[u11 + i + j * ldwork] = A[cut + i + (cut + j) * lda];
                }
            }

            i = 0;
            while (i < cut) {
                if (ipiv[i] > 0) {
                    for (j = 0; j < nnb; j++) {
                        work[i + j * ldwork] = work[i + invd * ldwork] * work[i + j * ldwork];
                    }
                } else {
                    for (j = 0; j < nnb; j++) {
                        u01_i_j = work[i + j * ldwork];
                        u01_ip1_j = work[i + 1 + j * ldwork];
                        work[i + j * ldwork] = work[i + invd * ldwork] * u01_i_j
                                             + work[i + (invd + 1) * ldwork] * u01_ip1_j;
                        work[i + 1 + j * ldwork] = work[i + 1 + invd * ldwork] * u01_i_j
                                                 + work[i + 1 + (invd + 1) * ldwork] * u01_ip1_j;
                    }
                    i = i + 1;
                }
                i = i + 1;
            }

            i = 0;
            while (i < nnb) {
                if (ipiv[cut + i] > 0) {
                    for (j = i; j < nnb; j++) {
                        work[u11 + i + j * ldwork] = work[cut + i + invd * ldwork] * work[u11 + i + j * ldwork];
                    }
                } else {
                    for (j = i; j < nnb; j++) {
                        u11_i_j = work[u11 + i + j * ldwork];
                        u11_ip1_j = work[u11 + i + 1 + j * ldwork];
                        work[u11 + i + j * ldwork] = work[cut + i + invd * ldwork] * work[u11 + i + j * ldwork]
                                                   + work[cut + i + (invd + 1) * ldwork] * work[u11 + i + 1 + j * ldwork];
                        work[u11 + i + 1 + j * ldwork] = work[cut + i + 1 + invd * ldwork] * u11_i_j
                                                       + work[cut + i + 1 + (invd + 1) * ldwork] * u11_ip1_j;
                    }
                    i = i + 1;
                }
                i = i + 1;
            }

            cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                        nnb, nnb, ONE, &A[cut + (cut) * lda], lda, &work[u11 + 0 * ldwork], ldwork);

            for (i = 0; i < nnb; i++) {
                for (j = i; j < nnb; j++) {
                    A[cut + i + (cut + j) * lda] = work[u11 + i + j * ldwork];
                }
            }

            cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                        nnb, nnb, cut, ONE, &A[0 + cut * lda], lda,
                        work, ldwork, ZERO, &work[u11 + 0 * ldwork], ldwork);

            for (i = 0; i < nnb; i++) {
                for (j = i; j < nnb; j++) {
                    A[cut + i + (cut + j) * lda] = A[cut + i + (cut + j) * lda] + work[u11 + i + j * ldwork];
                }
            }

            cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasUnit,
                        cut, nnb, ONE, A, lda, work, ldwork);

            for (i = 0; i < cut; i++) {
                for (j = 0; j < nnb; j++) {
                    A[i + (cut + j) * lda] = work[i + j * ldwork];
                }
            }
        }

        for (i = 0; i < n; i++) {
            ip = abs(ipiv[i]) - 1;
            if (ip != i) {
                if (i < ip) {
                    ssyswapr(uplo, n, A, lda, i, ip);
                }
                if (i > ip) {
                    ssyswapr(uplo, n, A, lda, ip, i);
                }
            }
        }

    } else {

        strtri(uplo, "U", n, A, lda, info);

        k = n - 1;
        while (k >= 0) {
            if (ipiv[k] > 0) {
                work[k + invd * ldwork] = ONE / A[k + k * lda];
                work[k + (invd + 1) * ldwork] = ZERO;
            } else {
                t = work[k - 1 + 0 * ldwork];
                ak = A[k - 1 + (k - 1) * lda] / t;
                akp1 = A[k + k * lda] / t;
                akkp1 = work[k - 1 + 0 * ldwork] / t;
                d = t * (ak * akp1 - ONE);
                work[k - 1 + invd * ldwork] = akp1 / d;
                work[k + invd * ldwork] = ak / d;
                work[k + (invd + 1) * ldwork] = -akkp1 / d;
                work[k - 1 + (invd + 1) * ldwork] = work[k + (invd + 1) * ldwork];
                k = k - 1;
            }
            k = k - 1;
        }

        cut = 0;
        while (cut < n) {
            nnb = nb;
            if (cut + nnb > n) {
                nnb = n - cut;
            } else {
                icount = 0;
                for (i = cut; i < cut + nnb; i++) {
                    if (ipiv[i] < 0) {
                        icount = icount + 1;
                    }
                }
                if (icount % 2 == 1) {
                    nnb = nnb + 1;
                }
            }

            for (i = 0; i < n - cut - nnb; i++) {
                for (j = 0; j < nnb; j++) {
                    work[i + j * ldwork] = A[cut + nnb + i + (cut + j) * lda];
                }
            }

            for (i = 0; i < nnb; i++) {
                work[u11 + i + i * ldwork] = ONE;
                for (j = i + 1; j < nnb; j++) {
                    work[u11 + i + j * ldwork] = ZERO;
                }
                for (j = 0; j < i; j++) {
                    work[u11 + i + j * ldwork] = A[cut + i + (cut + j) * lda];
                }
            }

            i = n - cut - nnb - 1;
            while (i >= 0) {
                if (ipiv[cut + nnb + i] > 0) {
                    for (j = 0; j < nnb; j++) {
                        work[i + j * ldwork] = work[cut + nnb + i + invd * ldwork] * work[i + j * ldwork];
                    }
                } else {
                    for (j = 0; j < nnb; j++) {
                        u01_i_j = work[i + j * ldwork];
                        u01_ip1_j = work[i - 1 + j * ldwork];
                        work[i + j * ldwork] = work[cut + nnb + i + invd * ldwork] * u01_i_j
                                             + work[cut + nnb + i + (invd + 1) * ldwork] * u01_ip1_j;
                        work[i - 1 + j * ldwork] = work[cut + nnb + i - 1 + (invd + 1) * ldwork] * u01_i_j
                                                 + work[cut + nnb + i - 1 + invd * ldwork] * u01_ip1_j;
                    }
                    i = i - 1;
                }
                i = i - 1;
            }

            i = nnb - 1;
            while (i >= 0) {
                if (ipiv[cut + i] > 0) {
                    for (j = 0; j < nnb; j++) {
                        work[u11 + i + j * ldwork] = work[cut + i + invd * ldwork] * work[u11 + i + j * ldwork];
                    }
                } else {
                    for (j = 0; j < nnb; j++) {
                        u11_i_j = work[u11 + i + j * ldwork];
                        u11_ip1_j = work[u11 + i - 1 + j * ldwork];
                        work[u11 + i + j * ldwork] = work[cut + i + invd * ldwork] * work[u11 + i + j * ldwork]
                                                   + work[cut + i + (invd + 1) * ldwork] * u11_ip1_j;
                        work[u11 + i - 1 + j * ldwork] = work[cut + i - 1 + (invd + 1) * ldwork] * u11_i_j
                                                       + work[cut + i - 1 + invd * ldwork] * u11_ip1_j;
                    }
                    i = i - 1;
                }
                i = i - 1;
            }

            cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                        nnb, nnb, ONE, &A[cut + cut * lda], lda, &work[u11 + 0 * ldwork], ldwork);

            for (i = 0; i < nnb; i++) {
                for (j = 0; j <= i; j++) {
                    A[cut + i + (cut + j) * lda] = work[u11 + i + j * ldwork];
                }
            }

            if (cut + nnb < n) {

                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            nnb, nnb, n - nnb - cut, ONE,
                            &A[cut + nnb + cut * lda], lda, work, ldwork,
                            ZERO, &work[u11 + 0 * ldwork], ldwork);

                for (i = 0; i < nnb; i++) {
                    for (j = 0; j <= i; j++) {
                        A[cut + i + (cut + j) * lda] = A[cut + i + (cut + j) * lda] + work[u11 + i + j * ldwork];
                    }
                }

                cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                            n - nnb - cut, nnb, ONE,
                            &A[cut + nnb + (cut + nnb) * lda], lda, work, ldwork);

                for (i = 0; i < n - cut - nnb; i++) {
                    for (j = 0; j < nnb; j++) {
                        A[cut + nnb + i + (cut + j) * lda] = work[i + j * ldwork];
                    }
                }

            } else {

                for (i = 0; i < nnb; i++) {
                    for (j = 0; j <= i; j++) {
                        A[cut + i + (cut + j) * lda] = work[u11 + i + j * ldwork];
                    }
                }
            }

            cut = cut + nnb;
        }

        for (i = n - 1; i >= 0; i--) {
            ip = abs(ipiv[i]) - 1;
            if (ip != i) {
                if (i < ip) {
                    ssyswapr(uplo, n, A, lda, i, ip);
                }
                if (i > ip) {
                    ssyswapr(uplo, n, A, lda, ip, i);
                }
            }
        }
    }
}
