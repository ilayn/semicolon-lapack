/**
 * @file dppt02.c
 * @brief DPPT02 computes the residual in the solution of a symmetric system with packed storage.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

extern double dlamch(const char* cmach);
extern double dlansp(const char* norm, const char* uplo, const int n,
                     const double* const restrict AP,
                     double* const restrict work);

/**
 * DPPT02 computes the residual in the solution of a symmetric system
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
void dppt02(const char* uplo, const int n, const int nrhs,
            const double* const restrict A,
            const double* const restrict X, const int ldx,
            double* const restrict B, const int ldb,
            double* const restrict rwork,
            double* resid)
{
    int j;
    double anorm, bnorm, eps, xnorm;
    CBLAS_UPLO cblas_uplo;

    if (n <= 0 || nrhs <= 0) {
        *resid = 0.0;
        return;
    }

    eps = dlamch("E");
    anorm = dlansp("1", uplo, n, A, rwork);
    if (anorm <= 0.0) {
        *resid = 1.0 / eps;
        return;
    }

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    for (j = 0; j < nrhs; j++) {
        cblas_dspmv(CblasColMajor, cblas_uplo, n, -1.0, A,
                    &X[j * ldx], 1, 1.0, &B[j * ldb], 1);
    }

    *resid = 0.0;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_dasum(n, &B[j * ldb], 1);
        xnorm = cblas_dasum(n, &X[j * ldx], 1);
        if (xnorm <= 0.0) {
            *resid = 1.0 / eps;
        } else {
            double tmp = ((bnorm / anorm) / xnorm) / eps;
            if (tmp > *resid) {
                *resid = tmp;
            }
        }
    }
}
