/**
 * @file dptt02.c
 * @brief DPTT02 computes the residual for the solution to a symmetric
 *        tridiagonal system of equations.
 *
 * Port of LAPACK's TESTING/LIN/dptt02.f to C.
 */

#include "semicolon_lapack_double.h"
#include "verify.h"
#include <cblas.h>
#include <math.h>

/**
 * DPTT02 computes the residual for the solution to a symmetric
 * tridiagonal system of equations:
 *    RESID = norm(B - A*X) / (norm(A) * norm(X) * EPS),
 * where EPS is the machine epsilon.
 *
 * @param[in]     n     The order of the matrix A.
 * @param[in]     nrhs  The number of right hand sides.
 * @param[in]     D     Diagonal elements of A (n).
 * @param[in]     E     Subdiagonal elements of A (n-1).
 * @param[in]     X     Solution matrix (ldx x nrhs).
 * @param[in]     ldx   Leading dimension of X.
 * @param[in,out] B     On entry, the right hand side. On exit, overwritten
 *                      with B - A*X.
 * @param[in]     ldb   Leading dimension of B.
 * @param[out]    resid The residual.
 */
void dptt02(
    const int n,
    const int nrhs,
    const f64* const restrict D,
    const f64* const restrict E,
    const f64* const restrict X,
    const int ldx,
    f64* const restrict B,
    const int ldb,
    f64* resid)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    /* Quick return if possible */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Compute the 1-norm of the tridiagonal matrix A. */
    f64 anorm = dlanst("1", n, D, E);

    /* Exit with RESID = 1/EPS if ANORM = 0. */
    f64 eps = dlamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute B - A*X. */
    dlaptm(n, nrhs, -ONE, D, E, X, ldx, ONE, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / ( norm(A) * norm(X) * EPS ). */
    *resid = ZERO;
    for (int j = 0; j < nrhs; j++) {
        f64 bnorm = cblas_dasum(n, &B[j * ldb], 1);
        f64 xnorm = cblas_dasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            *resid = fmax(*resid, ((bnorm / anorm) / xnorm) / eps);
        }
    }
}
