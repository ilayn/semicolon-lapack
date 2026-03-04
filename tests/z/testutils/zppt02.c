/**
 * @file zppt02.c
 * @brief ZPPT02 computes the residual in the solution of a Hermitian system with packed storage.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZPPT02 computes the residual in the solution of a Hermitian system
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
 *          The original Hermitian matrix A, stored as a packed
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
void zppt02(const char* uplo, const INT n, const INT nrhs,
            const c128* const restrict A,
            const c128* const restrict X, const INT ldx,
            c128* const restrict B, const INT ldb,
            f64* const restrict rwork,
            f64* resid)
{
    INT j;
    f64 anorm, bnorm, eps, xnorm;
    CBLAS_UPLO cblas_uplo;

    const c128 CNEGONE = CMPLX(-1.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    if (n <= 0 || nrhs <= 0) {
        *resid = 0.0;
        return;
    }

    eps = dlamch("E");
    anorm = zlanhp("1", uplo, n, A, rwork);
    if (anorm <= 0.0) {
        *resid = 1.0 / eps;
        return;
    }

    cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    for (j = 0; j < nrhs; j++) {
        cblas_zhpmv(CblasColMajor, cblas_uplo, n, &CNEGONE, A,
                    &X[j * ldx], 1, &CONE, &B[j * ldb], 1);
    }

    *resid = 0.0;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_dzasum(n, &B[j * ldb], 1);
        xnorm = cblas_dzasum(n, &X[j * ldx], 1);
        if (xnorm <= 0.0) {
            *resid = 1.0 / eps;
        } else {
            f64 tmp = ((bnorm / anorm) / xnorm) / eps;
            if (tmp > *resid) {
                *resid = tmp;
            }
        }
    }
}
