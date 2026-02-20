/**
 * @file dpot06.c
 * @brief DPOT06 computes the residual for a solution of a symmetric positive definite system.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "verify.h"

/**
 * DPOT06 computes the residual for a solution of a system of linear
 * equations  A*x = b:
 *    RESID = norm(B - A*X,inf) / ( norm(A,inf) * norm(X,inf) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part of the
 *                        symmetric matrix A is stored:
 *                        = "U":  Upper triangular
 *                        = "L":  Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A.  n >= 0.
 * @param[in]     nrhs    The number of columns of B, the matrix of right hand sides.
 *                        nrhs >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original n x n symmetric matrix A.
 * @param[in]     lda     The leading dimension of the array A.  lda >= max(1,n).
 * @param[in]     X       Double precision array, dimension (ldx, nrhs).
 *                        The computed solution vectors for the system.
 * @param[in]     ldx     The leading dimension of the array X.  ldx >= max(1,n).
 * @param[in,out] B       Double precision array, dimension (ldb, nrhs).
 *                        On entry, the right hand side vectors.
 *                        On exit, B is overwritten with the difference B - A*X.
 * @param[in]     ldb     The leading dimension of the array B.  ldb >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   The maximum over the number of right hand sides of
 *                        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void dpot06(const char* uplo, const int n, const int nrhs,
            const f64* A, const int lda, const f64* X, const int ldx,
            f64* B, const int ldb, f64* rwork, f64* resid)
{
    (void)rwork;  /* unused in this implementation */

    f64 anorm, bnorm, xnorm, eps;

    /* Quick exit if n = 0 or nrhs = 0 */
    if (n <= 0 || nrhs == 0) {
        *resid = 0.0;
        return;
    }

    /* Get machine epsilon */
    eps = DBL_EPSILON;

    /* Compute infinity norm of A (symmetric) */
    /* For symmetric matrices, ||A||_inf = max over rows of sum of |a_ij| */
    /* Since only upper or lower is stored, we need to account for both parts */
    anorm = 0.0;
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Upper triangular stored */
        for (int i = 0; i < n; i++) {
            f64 row_sum = 0.0;
            /* Elements in column j >= i are stored (upper triangle) */
            for (int j = 0; j < i; j++) {
                /* A[j,i] is in upper part but corresponds to A[i,j] */
                row_sum += fabs(A[j + i * lda]);
            }
            for (int j = i; j < n; j++) {
                /* A[i,j] is stored */
                row_sum += fabs(A[i + j * lda]);
            }
            if (row_sum > anorm) anorm = row_sum;
        }
    } else {
        /* Lower triangular stored */
        for (int i = 0; i < n; i++) {
            f64 row_sum = 0.0;
            /* Elements in column j <= i are stored (lower triangle) */
            for (int j = 0; j <= i; j++) {
                /* A[i,j] is stored */
                row_sum += fabs(A[i + j * lda]);
            }
            for (int j = i + 1; j < n; j++) {
                /* A[j,i] is in lower part but corresponds to A[i,j] */
                row_sum += fabs(A[j + i * lda]);
            }
            if (row_sum > anorm) anorm = row_sum;
        }
    }

    /* Exit with resid = 1/eps if anorm = 0 */
    if (anorm <= 0.0) {
        *resid = 1.0 / eps;
        return;
    }

    /* Compute B - A*X and store in B using DSYMM */
    /* B := -A*X + B */
    CBLAS_UPLO uplo_cblas = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    cblas_dsymm(CblasColMajor, CblasLeft, uplo_cblas,
                n, nrhs, -1.0, A, lda, X, ldx, 1.0, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / (norm(A) * norm(X) * eps) */
    *resid = 0.0;
    for (int j = 0; j < nrhs; j++) {
        /* Compute infinity norm of column j of B */
        int idx = cblas_idamax(n, &B[j * ldb], 1);
        bnorm = fabs(B[idx + j * ldb]);

        /* Compute infinity norm of column j of X */
        idx = cblas_idamax(n, &X[j * ldx], 1);
        xnorm = fabs(X[idx + j * ldx]);

        if (xnorm <= 0.0) {
            *resid = 1.0 / eps;
        } else {
            f64 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) *resid = ratio;
        }
    }
}
