/**
 * @file sppt02.c
 * @brief SPPT02 computes the residual in the solution of a symmetric system with packed storage.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

extern f32 slamch(const char* cmach);
extern f32 slansp(const char* norm, const char* uplo, const int n,
                     const f32* const restrict AP,
                     f32* const restrict work);

/**
 * SPPT02 computes the residual in the solution of a symmetric system
 * of linear equations  A*x = b  when packed storage is used for the
 * coefficient matrix.  The ratio computed is
 *
 *    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS),
 *
 * where EPS is the machine precision.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The number of rows and columns of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of columns of B, the matrix of right hand sides.
 *
 * @param[in] A
 *          The original symmetric matrix A, stored as a packed
 *          triangular matrix. Dimension (n*(n+1)/2).
 *
 * @param[in] X
 *          The computed solution vectors for the system of linear
 *          equations. Dimension (ldx, nrhs).
 *
 * @param[in] ldx
 *          The leading dimension of the array X. ldx >= max(1, n).
 *
 * @param[in,out] B
 *          On entry, the right hand side vectors for the system of
 *          linear equations.
 *          On exit, B is overwritten with the difference B - A*X.
 *          Dimension (ldb, nrhs).
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] rwork
 *          Workspace of dimension (n).
 *
 * @param[out] resid
 *          The maximum over the number of right hand sides of
 *          norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void sppt02(const char* uplo, const int n, const int nrhs,
            const f32* const restrict A,
            const f32* const restrict X, const int ldx,
            f32* const restrict B, const int ldb,
            f32* const restrict rwork,
            f32* resid)
{
    int j;
    f32 anorm, bnorm, eps, xnorm;
    CBLAS_UPLO cblas_uplo;

    if (n <= 0 || nrhs <= 0) {
        *resid = 0.0f;
        return;
    }

    eps = slamch("E");
    anorm = slansp("1", uplo, n, A, rwork);
    if (anorm <= 0.0f) {
        *resid = 1.0f / eps;
        return;
    }

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    for (j = 0; j < nrhs; j++) {
        cblas_sspmv(CblasColMajor, cblas_uplo, n, -1.0f, A,
                    &X[j * ldx], 1, 1.0f, &B[j * ldb], 1);
    }

    *resid = 0.0f;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_sasum(n, &B[j * ldb], 1);
        xnorm = cblas_sasum(n, &X[j * ldx], 1);
        if (xnorm <= 0.0f) {
            *resid = 1.0f / eps;
        } else {
            f32 tmp = ((bnorm / anorm) / xnorm) / eps;
            if (tmp > *resid) {
                *resid = tmp;
            }
        }
    }
}
