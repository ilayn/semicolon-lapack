/**
 * @file chegst.c
 * @brief CHEGST reduces a Hermitian-definite generalized eigenproblem to standard form.
 */

#include "semicolon_lapack_complex_single.h"
#include "lapack_tuning.h"
#include <complex.h>
#include <cblas.h>

/**
 * CHEGST reduces a complex Hermitian-definite generalized eigenproblem
 * to standard form.
 *
 * If ITYPE = 1, the problem is A*x = lambda*B*x,
 * and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
 *
 * If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 * B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H*A*L.
 *
 * B must have been previously factorized as U**H*U or L*L**H by CPOTRF.
 *
 * @param[in]     itype = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H);
 *                      = 2 or 3: compute U*A*U**H or L**H*A*L.
 * @param[in]     uplo  = 'U': Upper triangle stored, B = U**H*U;
 *                      = 'L': Lower triangle stored, B = L*L**H.
 * @param[in]     n     The order of the matrices A and B. n >= 0.
 * @param[in,out] A     On entry, the Hermitian matrix A. On exit, the transformed matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in,out] B     The triangular factor from Cholesky factorization of B.
 *                      B is modified by the routine but restored on exit.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit; < 0: if -i, the i-th argument was illegal.
 */
void chegst(
    const int itype,
    const char* uplo,
    const int n,
    c64* restrict A,
    const int lda,
    c64* restrict B,
    const int ldb,
    int* info)
{
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 HALF = CMPLXF(0.5f, 0.0f);
    const c64 NEG_HALF = CMPLXF(-0.5f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);
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
        xerbla("CHEGST", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    nb = lapack_get_nb("SYGST");

    if (nb <= 1 || nb >= n) {
        chegs2(itype, uplo, n, A, lda, B, ldb, info);
    } else {
        if (itype == 1) {
            if (upper) {
                /* Compute inv(U**H)*A*inv(U) */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    chegs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);

                    if (k + kb < n) {
                        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                                    CblasNonUnit, kb, n - k - kb, &CONE,
                                    &B[k + k * ldb], ldb,
                                    &A[k + (k + kb) * lda], lda);
                        cblas_chemm(CblasColMajor, CblasLeft, CblasUpper,
                                    kb, n - k - kb, &NEG_HALF,
                                    &A[k + k * lda], lda,
                                    &B[k + (k + kb) * ldb], ldb, &CONE,
                                    &A[k + (k + kb) * lda], lda);
                        cblas_cher2k(CblasColMajor, CblasUpper, CblasConjTrans,
                                     n - k - kb, kb, &NEG_CONE,
                                     &A[k + (k + kb) * lda], lda,
                                     &B[k + (k + kb) * ldb], ldb, ONE,
                                     &A[(k + kb) + (k + kb) * lda], lda);
                        cblas_chemm(CblasColMajor, CblasLeft, CblasUpper,
                                    kb, n - k - kb, &NEG_HALF,
                                    &A[k + k * lda], lda,
                                    &B[k + (k + kb) * ldb], ldb, &CONE,
                                    &A[k + (k + kb) * lda], lda);
                        cblas_ctrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans,
                                    CblasNonUnit, kb, n - k - kb, &CONE,
                                    &B[(k + kb) + (k + kb) * ldb], ldb,
                                    &A[k + (k + kb) * lda], lda);
                    }
                }
            } else {
                /* Compute inv(L)*A*inv(L**H) */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    chegs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);

                    if (k + kb < n) {
                        cblas_ctrsm(CblasColMajor, CblasRight, CblasLower, CblasConjTrans,
                                    CblasNonUnit, n - k - kb, kb, &CONE,
                                    &B[k + k * ldb], ldb,
                                    &A[(k + kb) + k * lda], lda);
                        cblas_chemm(CblasColMajor, CblasRight, CblasLower,
                                    n - k - kb, kb, &NEG_HALF,
                                    &A[k + k * lda], lda,
                                    &B[(k + kb) + k * ldb], ldb, &CONE,
                                    &A[(k + kb) + k * lda], lda);
                        cblas_cher2k(CblasColMajor, CblasLower, CblasNoTrans,
                                     n - k - kb, kb, &NEG_CONE,
                                     &A[(k + kb) + k * lda], lda,
                                     &B[(k + kb) + k * ldb], ldb, ONE,
                                     &A[(k + kb) + (k + kb) * lda], lda);
                        cblas_chemm(CblasColMajor, CblasRight, CblasLower,
                                    n - k - kb, kb, &NEG_HALF,
                                    &A[k + k * lda], lda,
                                    &B[(k + kb) + k * ldb], ldb, &CONE,
                                    &A[(k + kb) + k * lda], lda);
                        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                                    CblasNonUnit, n - k - kb, kb, &CONE,
                                    &B[(k + kb) + (k + kb) * ldb], ldb,
                                    &A[(k + kb) + k * lda], lda);
                    }
                }
            }
        } else {
            if (upper) {
                /* Compute U*A*U**H */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                                CblasNonUnit, k, kb, &CONE, B, ldb, &A[0 + k * lda], lda);
                    cblas_chemm(CblasColMajor, CblasRight, CblasUpper,
                                k, kb, &HALF,
                                &A[k + k * lda], lda,
                                &B[0 + k * ldb], ldb, &CONE,
                                &A[0 + k * lda], lda);
                    cblas_cher2k(CblasColMajor, CblasUpper, CblasNoTrans,
                                 k, kb, &CONE,
                                 &A[0 + k * lda], lda,
                                 &B[0 + k * ldb], ldb, ONE, A, lda);
                    cblas_chemm(CblasColMajor, CblasRight, CblasUpper,
                                k, kb, &HALF,
                                &A[k + k * lda], lda,
                                &B[0 + k * ldb], ldb, &CONE,
                                &A[0 + k * lda], lda);
                    cblas_ctrmm(CblasColMajor, CblasRight, CblasUpper, CblasConjTrans,
                                CblasNonUnit, k, kb, &CONE,
                                &B[k + k * ldb], ldb, &A[0 + k * lda], lda);

                    chegs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);
                }
            } else {
                /* Compute L**H*A*L */
                for (k = 0; k < n; k += nb) {
                    kb = (n - k < nb) ? n - k : nb;

                    cblas_ctrmm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans,
                                CblasNonUnit, kb, k, &CONE, B, ldb, &A[k + 0 * lda], lda);
                    cblas_chemm(CblasColMajor, CblasLeft, CblasLower,
                                kb, k, &HALF,
                                &A[k + k * lda], lda,
                                &B[k + 0 * ldb], ldb, &CONE,
                                &A[k + 0 * lda], lda);
                    cblas_cher2k(CblasColMajor, CblasLower, CblasConjTrans,
                                 k, kb, &CONE,
                                 &A[k + 0 * lda], lda,
                                 &B[k + 0 * ldb], ldb, ONE, A, lda);
                    cblas_chemm(CblasColMajor, CblasLeft, CblasLower,
                                kb, k, &HALF,
                                &A[k + k * lda], lda,
                                &B[k + 0 * ldb], ldb, &CONE,
                                &A[k + 0 * lda], lda);
                    cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                                CblasNonUnit, kb, k, &CONE,
                                &B[k + k * ldb], ldb, &A[k + 0 * lda], lda);

                    chegs2(itype, uplo, kb, &A[k + k * lda], lda,
                           &B[k + k * ldb], ldb, info);
                }
            }
        }
    }
}
