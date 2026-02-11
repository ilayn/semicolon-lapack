/**
 * @file dsygs2.c
 * @brief DSYGS2 reduces a symmetric-definite generalized eigenproblem
 *        to standard form (unblocked algorithm).
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>

/**
 * DSYGS2 reduces a real symmetric-definite generalized eigenproblem
 * to standard form.
 *
 * If ITYPE = 1, the problem is A*x = lambda*B*x,
 * and A is overwritten by inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T)
 *
 * If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 * B*A*x = lambda*x, and A is overwritten by U*A*U**T or L**T*A*L.
 *
 * B must have been previously factorized as U**T*U or L*L**T by DPOTRF.
 *
 * @param[in]     itype = 1: compute inv(U**T)*A*inv(U) or inv(L)*A*inv(L**T);
 *                      = 2 or 3: compute U*A*U**T or L**T*A*L.
 * @param[in]     uplo  = 'U': Upper triangular; = 'L': Lower triangular
 * @param[in]     n     The order of the matrices A and B. n >= 0.
 * @param[in,out] A     On entry, the symmetric matrix A. On exit, the transformed matrix.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[in]     B     The triangular factor from Cholesky factorization of B.
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit; < 0: if -i, the i-th argument was illegal.
 */
void dsygs2(
    const int itype,
    const char* uplo,
    const int n,
    double* restrict A,
    const int lda,
    const double* restrict B,
    const int ldb,
    int* info)
{
    const double ONE = 1.0;
    const double HALF = 0.5;
    int upper;
    int k;
    double akk, bkk, ct;

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
        xerbla("DSYGS2", -(*info));
        return;
    }

    if (itype == 1) {
        if (upper) {
            /* Compute inv(U**T)*A*inv(U) */
            for (k = 0; k < n; k++) {
                akk = A[k + k * lda];
                bkk = B[k + k * ldb];
                akk = akk / (bkk * bkk);
                A[k + k * lda] = akk;

                if (k < n - 1) {
                    cblas_dscal(n - k - 1, ONE / bkk, &A[k + (k + 1) * lda], lda);
                    ct = -HALF * akk;
                    cblas_daxpy(n - k - 1, ct, &B[k + (k + 1) * ldb], ldb,
                                &A[k + (k + 1) * lda], lda);
                    cblas_dsyr2(CblasColMajor, CblasUpper, n - k - 1, -ONE,
                                &A[k + (k + 1) * lda], lda,
                                &B[k + (k + 1) * ldb], ldb,
                                &A[(k + 1) + (k + 1) * lda], lda);
                    cblas_daxpy(n - k - 1, ct, &B[k + (k + 1) * ldb], ldb,
                                &A[k + (k + 1) * lda], lda);
                    cblas_dtrsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                                n - k - 1, &B[(k + 1) + (k + 1) * ldb], ldb,
                                &A[k + (k + 1) * lda], lda);
                }
            }
        } else {
            /* Compute inv(L)*A*inv(L**T) */
            for (k = 0; k < n; k++) {
                akk = A[k + k * lda];
                bkk = B[k + k * ldb];
                akk = akk / (bkk * bkk);
                A[k + k * lda] = akk;

                if (k < n - 1) {
                    cblas_dscal(n - k - 1, ONE / bkk, &A[(k + 1) + k * lda], 1);
                    ct = -HALF * akk;
                    cblas_daxpy(n - k - 1, ct, &B[(k + 1) + k * ldb], 1,
                                &A[(k + 1) + k * lda], 1);
                    cblas_dsyr2(CblasColMajor, CblasLower, n - k - 1, -ONE,
                                &A[(k + 1) + k * lda], 1,
                                &B[(k + 1) + k * ldb], 1,
                                &A[(k + 1) + (k + 1) * lda], lda);
                    cblas_daxpy(n - k - 1, ct, &B[(k + 1) + k * ldb], 1,
                                &A[(k + 1) + k * lda], 1);
                    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                                n - k - 1, &B[(k + 1) + (k + 1) * ldb], ldb,
                                &A[(k + 1) + k * lda], 1);
                }
            }
        }
    } else {
        if (upper) {
            /* Compute U*A*U**T */
            for (k = 0; k < n; k++) {
                akk = A[k + k * lda];
                bkk = B[k + k * ldb];

                cblas_dtrmv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                            k, B, ldb, &A[0 + k * lda], 1);
                ct = HALF * akk;
                cblas_daxpy(k, ct, &B[0 + k * ldb], 1, &A[0 + k * lda], 1);
                cblas_dsyr2(CblasColMajor, CblasUpper, k, ONE,
                            &A[0 + k * lda], 1, &B[0 + k * ldb], 1, A, lda);
                cblas_daxpy(k, ct, &B[0 + k * ldb], 1, &A[0 + k * lda], 1);
                cblas_dscal(k, bkk, &A[0 + k * lda], 1);
                A[k + k * lda] = akk * bkk * bkk;
            }
        } else {
            /* Compute L**T*A*L */
            for (k = 0; k < n; k++) {
                akk = A[k + k * lda];
                bkk = B[k + k * ldb];

                cblas_dtrmv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                            k, B, ldb, &A[k + 0 * lda], lda);
                ct = HALF * akk;
                cblas_daxpy(k, ct, &B[k + 0 * ldb], ldb, &A[k + 0 * lda], lda);
                cblas_dsyr2(CblasColMajor, CblasLower, k, ONE,
                            &A[k + 0 * lda], lda, &B[k + 0 * ldb], ldb, A, lda);
                cblas_daxpy(k, ct, &B[k + 0 * ldb], ldb, &A[k + 0 * lda], lda);
                cblas_dscal(k, bkk, &A[k + 0 * lda], lda);
                A[k + k * lda] = akk * bkk * bkk;
            }
        }
    }
}
