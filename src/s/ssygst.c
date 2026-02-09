/**
 * @file ssygst.c
 * @brief SSYGST reduces a symmetric-definite generalized eigenproblem to standard form.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <cblas.h>

/**
 * SSYGST reduces a real symmetric-definite generalized eigenproblem to standard form.
 *
 * If ITYPE = 1, the problem is A*x = lambda*B*x,
 * and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
 *
 * If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 * B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
 *
 * B must have been previously factorized as U**T*U or L*L**T by SPOTRF.
 *
 * @param[in]     itype = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T);
 *                      = 2 or 3: compute U*A*U**T or L**T*A*L.
 * @param[in]     uplo  = 'U': Upper triangle stored, B = U**T*U;
 *                      = 'L': Lower triangle stored, B = L*L**T.
 * @param[in]     n     The order of the matrices A and B. n >= 0.
 * @param[in,out] A     On entry, the symmetric matrix A. On exit, the transformed matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     B     The triangular factor from Cholesky factorization of B.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[out]    info  = 0: successful exit; < 0: if -i, the i-th argument was illegal.
 */
void ssygst(
    const int itype,
    const char* uplo,
    const int n,
    float* restrict A,
    const int lda,
    const float* restrict B,
    const int ldb,
    int* info)
{
    const float ONE = 1.0f;
    const float HALF = 0.5f;
    int upper;
    int k, kb, nb;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("SSYGST", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    nb = lapack_get_nb("SYGST");

    if (nb <= 1 || nb >= n) {
        ssygs2(itype, uplo, n, A, lda, B, ldb, info);
    } else {
        if (itype == 1) {
            if (upper) {
                /* Compute inv(U**T)*A*inv(U) */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    ssygs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);

                    if (k + kb < n) {
                        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                                    CblasNonUnit, kb, n - k - kb, ONE,
                                    &B[k + k * ldb], ldb,
                                    &A[k + (k + kb) * lda], lda);
                        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
                                    kb, n - k - kb, -HALF,
                                    &A[k + k * lda], lda,
                                    &B[k + (k + kb) * ldb], ldb, ONE,
                                    &A[k + (k + kb) * lda], lda);
                        cblas_ssyr2k(CblasColMajor, CblasUpper, CblasTrans,
                                     n - k - kb, kb, -ONE,
                                     &A[k + (k + kb) * lda], lda,
                                     &B[k + (k + kb) * ldb], ldb, ONE,
                                     &A[(k + kb) + (k + kb) * lda], lda);
                        cblas_ssymm(CblasColMajor, CblasLeft, CblasUpper,
                                    kb, n - k - kb, -HALF,
                                    &A[k + k * lda], lda,
                                    &B[k + (k + kb) * ldb], ldb, ONE,
                                    &A[k + (k + kb) * lda], lda);
                        cblas_strsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                                    CblasNonUnit, kb, n - k - kb, ONE,
                                    &B[(k + kb) + (k + kb) * ldb], ldb,
                                    &A[k + (k + kb) * lda], lda);
                    }
                }
            } else {
                /* Compute inv(L)*A*inv(L**T) */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    ssygs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);

                    if (k + kb < n) {
                        cblas_strsm(CblasColMajor, CblasRight, CblasLower, CblasTrans,
                                    CblasNonUnit, n - k - kb, kb, ONE,
                                    &B[k + k * ldb], ldb,
                                    &A[(k + kb) + k * lda], lda);
                        cblas_ssymm(CblasColMajor, CblasRight, CblasLower,
                                    n - k - kb, kb, -HALF,
                                    &A[k + k * lda], lda,
                                    &B[(k + kb) + k * ldb], ldb, ONE,
                                    &A[(k + kb) + k * lda], lda);
                        cblas_ssyr2k(CblasColMajor, CblasLower, CblasNoTrans,
                                     n - k - kb, kb, -ONE,
                                     &A[(k + kb) + k * lda], lda,
                                     &B[(k + kb) + k * ldb], ldb, ONE,
                                     &A[(k + kb) + (k + kb) * lda], lda);
                        cblas_ssymm(CblasColMajor, CblasRight, CblasLower,
                                    n - k - kb, kb, -HALF,
                                    &A[k + k * lda], lda,
                                    &B[(k + kb) + k * ldb], ldb, ONE,
                                    &A[(k + kb) + k * lda], lda);
                        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                                    CblasNonUnit, n - k - kb, kb, ONE,
                                    &B[(k + kb) + (k + kb) * ldb], ldb,
                                    &A[(k + kb) + k * lda], lda);
                    }
                }
            }
        } else {
            if (upper) {
                /* Compute U*A*U**T */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                                CblasNonUnit, k, kb, ONE, B, ldb, &A[0 + k * lda], lda);
                    cblas_ssymm(CblasColMajor, CblasRight, CblasUpper,
                                k, kb, HALF,
                                &A[k + k * lda], lda,
                                &B[0 + k * ldb], ldb, ONE,
                                &A[0 + k * lda], lda);
                    cblas_ssyr2k(CblasColMajor, CblasUpper, CblasNoTrans,
                                 k, kb, ONE,
                                 &A[0 + k * lda], lda,
                                 &B[0 + k * ldb], ldb, ONE, A, lda);
                    cblas_ssymm(CblasColMajor, CblasRight, CblasUpper,
                                k, kb, HALF,
                                &A[k + k * lda], lda,
                                &B[0 + k * ldb], ldb, ONE,
                                &A[0 + k * lda], lda);
                    cblas_strmm(CblasColMajor, CblasRight, CblasUpper, CblasTrans,
                                CblasNonUnit, k, kb, ONE,
                                &B[k + k * ldb], ldb, &A[0 + k * lda], lda);

                    ssygs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);
                }
            } else {
                /* Compute L**T*A*L */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    cblas_strmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                                CblasNonUnit, kb, k, ONE, B, ldb, &A[k + 0 * lda], lda);
                    cblas_ssymm(CblasColMajor, CblasLeft, CblasLower,
                                kb, k, HALF,
                                &A[k + k * lda], lda,
                                &B[k + 0 * ldb], ldb, ONE,
                                &A[k + 0 * lda], lda);
                    cblas_ssyr2k(CblasColMajor, CblasLower, CblasTrans,
                                 k, kb, ONE,
                                 &A[k + 0 * lda], lda,
                                 &B[k + 0 * ldb], ldb, ONE, A, lda);
                    cblas_ssymm(CblasColMajor, CblasLeft, CblasLower,
                                kb, k, HALF,
                                &A[k + k * lda], lda,
                                &B[k + 0 * ldb], ldb, ONE,
                                &A[k + 0 * lda], lda);
                    cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                                CblasNonUnit, kb, k, ONE,
                                &B[k + k * ldb], ldb, &A[k + 0 * lda], lda);

                    ssygs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);
                }
            }
        }
    }
}
