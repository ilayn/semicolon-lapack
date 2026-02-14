/**
 * @file zhegs2.c
 * @brief ZHEGS2 reduces a Hermitian-definite generalized eigenproblem to standard form.
 */
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZHEGS2 reduces a complex Hermitian-definite generalized
 * eigenproblem to standard form.
 *
 * If ITYPE = 1, the problem is A*x = lambda*B*x,
 * and A is overwritten by inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H)
 *
 * If ITYPE = 2 or 3, the problem is A*B*x = lambda*x or
 * B*A*x = lambda*x, and A is overwritten by U*A*U**H or L**H *A*L.
 *
 * B must have been previously factorized as U**H *U or L*L**H by ZPOTRF.
 *
 * @param[in]     itype = 1: compute inv(U**H)*A*inv(U) or inv(L)*A*inv(L**H);
 *                      = 2 or 3: compute U*A*U**H or L**H *A*L.
 * @param[in]     uplo  Specifies whether the upper or lower triangular part of the
 *                      Hermitian matrix A is stored, and how B has been factorized.
 *                      = 'U':  Upper triangular
 *                      = 'L':  Lower triangular
 * @param[in]     n     The order of the matrices A and B.  N >= 0.
 * @param[in,out] A     Complex*16 array, dimension (LDA,N).
 *                      On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
 *                      n by n upper triangular part of A contains the upper
 *                      triangular part of the matrix A, and the strictly lower
 *                      triangular part of A is not referenced.  If UPLO = 'L', the
 *                      leading n by n lower triangular part of A contains the lower
 *                      triangular part of the matrix A, and the strictly upper
 *                      triangular part of A is not referenced.
 *                      On exit, if INFO = 0, the transformed matrix, stored in the
 *                      same format as A.
 * @param[in]     lda   The leading dimension of the array A.  LDA >= max(1,N).
 * @param[in,out] B     Complex*16 array, dimension (LDB,N).
 *                      The triangular factor from the Cholesky factorization of B,
 *                      as returned by ZPOTRF.
 *                      B is modified by the routine but restored on exit.
 * @param[in]     ldb   The leading dimension of the array B.  LDB >= max(1,N).
 * @param[out]    info  = 0:  successful exit.
 *                      < 0:  if INFO = -i, the i-th argument had an illegal value.
 */
void zhegs2(
    const int itype,
    const char* uplo,
    const int n,
    c128* const restrict A,
    const int lda,
    c128* const restrict B,
    const int ldb,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 HALF = 0.5;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 NEG_CONE = CMPLX(-1.0, 0.0);

    int upper;
    int k;
    f64 akk, bkk;
    c128 ct;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("ZHEGS2", -(*info));
        return;
    }

    if (itype == 1) {
        if (upper) {
            //
            // Compute inv(U**H)*A*inv(U)
            //
            for (k = 0; k < n; k++) {
                //
                // Update the upper triangle of A(k:n,k:n)
                //
                akk = creal(A[k + k * lda]);
                bkk = creal(B[k + k * ldb]);
                akk = akk / (bkk * bkk);
                A[k + k * lda] = CMPLX(akk, 0.0);
                if (k < n - 1) {
                    cblas_zdscal(n - k - 1, ONE / bkk, &A[k + (k + 1) * lda], lda);
                    ct = CMPLX(-HALF * akk, 0.0);
                    zlacgv(n - k - 1, &A[k + (k + 1) * lda], lda);
                    zlacgv(n - k - 1, &B[k + (k + 1) * ldb], ldb);
                    cblas_zaxpy(n - k - 1, &ct, &B[k + (k + 1) * ldb], ldb,
                                &A[k + (k + 1) * lda], lda);
                    cblas_zher2(CblasColMajor, CblasUpper, n - k - 1,
                                &NEG_CONE, &A[k + (k + 1) * lda], lda,
                                &B[k + (k + 1) * ldb], ldb,
                                &A[(k + 1) + (k + 1) * lda], lda);
                    cblas_zaxpy(n - k - 1, &ct, &B[k + (k + 1) * ldb], ldb,
                                &A[k + (k + 1) * lda], lda);
                    zlacgv(n - k - 1, &B[k + (k + 1) * ldb], ldb);
                    cblas_ztrsv(CblasColMajor, CblasUpper, CblasConjTrans,
                                CblasNonUnit,
                                n - k - 1, &B[(k + 1) + (k + 1) * ldb], ldb,
                                &A[k + (k + 1) * lda], lda);
                    zlacgv(n - k - 1, &A[k + (k + 1) * lda], lda);
                }
            }
        } else {
            //
            // Compute inv(L)*A*inv(L**H)
            //
            for (k = 0; k < n; k++) {
                //
                // Update the lower triangle of A(k:n,k:n)
                //
                akk = creal(A[k + k * lda]);
                bkk = creal(B[k + k * ldb]);
                akk = akk / (bkk * bkk);
                A[k + k * lda] = CMPLX(akk, 0.0);
                if (k < n - 1) {
                    cblas_zdscal(n - k - 1, ONE / bkk, &A[(k + 1) + k * lda], 1);
                    ct = CMPLX(-HALF * akk, 0.0);
                    cblas_zaxpy(n - k - 1, &ct, &B[(k + 1) + k * ldb], 1,
                                &A[(k + 1) + k * lda], 1);
                    cblas_zher2(CblasColMajor, CblasLower, n - k - 1,
                                &NEG_CONE, &A[(k + 1) + k * lda], 1,
                                &B[(k + 1) + k * ldb], 1,
                                &A[(k + 1) + (k + 1) * lda], lda);
                    cblas_zaxpy(n - k - 1, &ct, &B[(k + 1) + k * ldb], 1,
                                &A[(k + 1) + k * lda], 1);
                    cblas_ztrsv(CblasColMajor, CblasLower, CblasNoTrans,
                                CblasNonUnit, n - k - 1,
                                &B[(k + 1) + (k + 1) * ldb], ldb,
                                &A[(k + 1) + k * lda], 1);
                }
            }
        }
    } else {
        if (upper) {
            //
            // Compute U*A*U**H
            //
            for (k = 0; k < n; k++) {
                //
                // Update the upper triangle of A(1:k,1:k)
                //
                akk = creal(A[k + k * lda]);
                bkk = creal(B[k + k * ldb]);
                cblas_ztrmv(CblasColMajor, CblasUpper, CblasNoTrans,
                            CblasNonUnit, k, B, ldb, &A[k * lda], 1);
                ct = CMPLX(HALF * akk, 0.0);
                cblas_zaxpy(k, &ct, &B[k * ldb], 1, &A[k * lda], 1);
                cblas_zher2(CblasColMajor, CblasUpper, k,
                            &CONE, &A[k * lda], 1, &B[k * ldb], 1,
                            A, lda);
                cblas_zaxpy(k, &ct, &B[k * ldb], 1, &A[k * lda], 1);
                cblas_zdscal(k, bkk, &A[k * lda], 1);
                A[k + k * lda] = CMPLX(akk * bkk * bkk, 0.0);
            }
        } else {
            //
            // Compute L**H *A*L
            //
            for (k = 0; k < n; k++) {
                //
                // Update the lower triangle of A(1:k,1:k)
                //
                akk = creal(A[k + k * lda]);
                bkk = creal(B[k + k * ldb]);
                zlacgv(k, &A[k], lda);
                cblas_ztrmv(CblasColMajor, CblasLower, CblasConjTrans,
                            CblasNonUnit, k, B, ldb, &A[k], lda);
                ct = CMPLX(HALF * akk, 0.0);
                zlacgv(k, &B[k], ldb);
                cblas_zaxpy(k, &ct, &B[k], ldb, &A[k], lda);
                cblas_zher2(CblasColMajor, CblasLower, k,
                            &CONE, &A[k], lda, &B[k], ldb,
                            A, lda);
                cblas_zaxpy(k, &ct, &B[k], ldb, &A[k], lda);
                zlacgv(k, &B[k], ldb);
                cblas_zdscal(k, bkk, &A[k], lda);
                zlacgv(k, &A[k], lda);
                A[k + k * lda] = CMPLX(akk * bkk * bkk, 0.0);
            }
        }
    }
}
