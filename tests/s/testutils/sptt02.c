/**
 * @file sptt02.c
 * @brief SPTT02 computes the residual for the solution to a symmetric
 *        tridiagonal system of equations.
 *
 * Port of LAPACK's TESTING/LIN/sptt02.f to C.
 */

#include "semicolon_lapack_single.h"
#include "verify.h"
#include <cblas.h>
#include <math.h>

/**
 * SPTT02 computes the residual for the solution to a symmetric
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
void sptt02(
    const int n,
    const int nrhs,
    const f32* const restrict D,
    const f32* const restrict E,
    const f32* const restrict X,
    const int ldx,
    f32* const restrict B,
    const int ldb,
    f32* resid)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    /* Quick return if possible */
    if (n <= 0) {
        *resid = ZERO;
        return;
    }

    /* Compute the 1-norm of the tridiagonal matrix A. */
    f32 anorm = slanst("1", n, D, E);

    /* Exit with RESID = 1/EPS if ANORM = 0. */
    f32 eps = slamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute B - A*X. */
    slaptm(n, nrhs, -ONE, D, E, X, ldx, ONE, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / ( norm(A) * norm(X) * EPS ). */
    *resid = ZERO;
    for (int j = 0; j < nrhs; j++) {
        f32 bnorm = cblas_sasum(n, &B[j * ldb], 1);
        f32 xnorm = cblas_sasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            *resid = fmaxf(*resid, ((bnorm / anorm) / xnorm) / eps);
        }
    }
}
