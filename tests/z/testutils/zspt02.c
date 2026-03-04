/**
 * @file zspt02.c
 * @brief ZSPT02 computes the residual for the solution of a complex symmetric
 *        system of linear equations A*x = b when packed storage is used.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZSPT02 computes the residual in the solution of a complex symmetric
 * system of linear equations  A*x = b  when packed storage is used for
 * the coefficient matrix.  The ratio computed is
 *
 *    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS).
 *
 * where EPS is the machine precision.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the complex symmetric matrix A is stored:
 *                       = 'U': Upper triangular
 *                       = 'L': Lower triangular
 * @param[in]     n      The number of rows and columns of the matrix A.  n >= 0.
 * @param[in]     nrhs   The number of columns of B, the matrix of right hand
 *                       sides.  nrhs >= 0.
 * @param[in]     A      Complex*16 array, dimension (n*(n+1)/2).
 *                       The original complex symmetric matrix A, stored as a
 *                       packed triangular matrix.
 * @param[in]     X      Complex*16 array, dimension (ldx, nrhs).
 *                       The computed solution vectors for the system of linear
 *                       equations.
 * @param[in]     ldx    The leading dimension of the array X.  ldx >= max(1,n).
 * @param[in,out] B      Complex*16 array, dimension (ldb, nrhs).
 *                       On entry, the right hand side vectors for the system of
 *                       linear equations.
 *                       On exit, B is overwritten with the difference B - A*X.
 * @param[in]     ldb    The leading dimension of the array B.  ldb >= max(1,n).
 * @param[out]    rwork  Double precision array, dimension (n).
 * @param[out]    resid  The maximum over the number of right hand sides of
 *                       norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void zspt02(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c128* const restrict A,
    const c128* const restrict X,
    const INT ldx,
    c128* const restrict B,
    const INT ldb,
    f64* const restrict rwork,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT j;
    f64 anorm, bnorm, eps, xnorm;

    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    anorm = zlansp("1", uplo, n, A, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    for (j = 0; j < nrhs; j++) {
        zspmv(uplo, n, CNEGONE, A, &X[j * ldx], 1, CONE, &B[j * ldb], 1);
    }

    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_dzasum(n, &B[j * ldb], 1);
        xnorm = cblas_dzasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f64 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }
}
