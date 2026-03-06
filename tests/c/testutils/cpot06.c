/**
 * @file cpot06.c
 * @brief CPOT06 computes the residual for a solution of a Hermitian positive definite system.
 */

#include <math.h>
#include <float.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CPOT06 computes the residual for a solution of a system of linear
 * equations  A*x = b:
 *    RESID = norm(B - A*X,inf) / ( norm(A,inf) * norm(X,inf) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part of the
 *                        Hermitian matrix A is stored:
 *                        = "U":  Upper triangular
 *                        = "L":  Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A.  n >= 0.
 * @param[in]     nrhs    The number of columns of B, the matrix of right hand sides.
 *                        nrhs >= 0.
 * @param[in]     A       Complex*16 array, dimension (lda, n).
 *                        The original n x n Hermitian matrix A.
 * @param[in]     lda     The leading dimension of the array A.  lda >= max(1,n).
 * @param[in]     X       Complex*16 array, dimension (ldx, nrhs).
 *                        The computed solution vectors for the system.
 * @param[in]     ldx     The leading dimension of the array X.  ldx >= max(1,n).
 * @param[in,out] B       Complex*16 array, dimension (ldb, nrhs).
 *                        On entry, the right hand side vectors.
 *                        On exit, B is overwritten with the difference B - A*X.
 * @param[in]     ldb     The leading dimension of the array B.  ldb >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   The maximum over the number of right hand sides of
 *                        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void cpot06(const char* uplo, const INT n, const INT nrhs,
            const c64* A, const INT lda, const c64* X, const INT ldx,
            c64* B, const INT ldb, f32* rwork, f32* resid)
{
    (void)rwork;  /* unused in this implementation */

    f32 anorm, bnorm, xnorm, eps;

    /* Quick exit if n = 0 or nrhs = 0 */
    if (n <= 0 || nrhs == 0) {
        *resid = 0.0f;
        return;
    }

    /* Get machine epsilon */
    eps = FLT_EPSILON;

    /* Compute infinity norm of A (Hermitian) */
    anorm = 0.0f;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Upper triangular stored */
        for (INT i = 0; i < n; i++) {
            f32 row_sum = 0.0f;
            for (INT j = 0; j < i; j++) {
                row_sum += cabs1f(A[j + i * lda]);
            }
            for (INT j = i; j < n; j++) {
                row_sum += cabs1f(A[i + j * lda]);
            }
            if (row_sum > anorm) anorm = row_sum;
        }
    } else {
        /* Lower triangular stored */
        for (INT i = 0; i < n; i++) {
            f32 row_sum = 0.0f;
            for (INT j = 0; j <= i; j++) {
                row_sum += cabs1f(A[i + j * lda]);
            }
            for (INT j = i + 1; j < n; j++) {
                row_sum += cabs1f(A[j + i * lda]);
            }
            if (row_sum > anorm) anorm = row_sum;
        }
    }

    /* Exit with resid = 1/eps if anorm = 0 */
    if (anorm <= 0.0f) {
        *resid = 1.0f / eps;
        return;
    }

    /* Compute B - A*X and store in B using ZHEMM */
    /* B := -A*X + B */
    CBLAS_UPLO uplo_cblas = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    cblas_chemm(CblasColMajor, CblasLeft, uplo_cblas,
                n, nrhs, &NEG_ONE, A, lda, X, ldx, &ONE, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / (norm(A) * norm(X) * eps) */
    *resid = 0.0f;
    for (INT j = 0; j < nrhs; j++) {
        /* Compute infinity norm of column j of B */
        INT idx = cblas_icamax(n, &B[j * ldb], 1);
        bnorm = cabs1f(B[idx + j * ldb]);

        /* Compute infinity norm of column j of X */
        idx = cblas_icamax(n, &X[j * ldx], 1);
        xnorm = cabs1f(X[idx + j * ldx]);

        if (xnorm <= 0.0f) {
            *resid = 1.0f / eps;
        } else {
            f32 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) *resid = ratio;
        }
    }
}
