/**
 * @file zptt02.c
 * @brief ZPTT02 computes the residual for the solution to a Hermitian
 *        tridiagonal system of equations.
 *
 * Port of LAPACK's TESTING/LIN/zptt02.f to C.
 */

#include "semicolon_lapack_complex_double.h"
#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * ZPTT02 computes the residual for the solution to a Hermitian
 * tridiagonal system of equations:
 *    RESID = norm(B - A*X) / (norm(A) * norm(X) * EPS),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo  Specifies whether the superdiagonal or the subdiagonal
 *                      of the tridiagonal matrix A is stored.
 *                      = 'U': E is the superdiagonal of A
 *                      = 'L': E is the subdiagonal of A
 * @param[in]     n     The order of the matrix A.
 * @param[in]     nrhs  The number of right hand sides.
 * @param[in]     D     Diagonal elements of A (n). Real.
 * @param[in]     E     Off-diagonal elements of A (n-1). Complex.
 * @param[in]     X     Solution matrix (ldx x nrhs). Complex.
 * @param[in]     ldx   Leading dimension of X.
 * @param[in,out] B     On entry, the right hand side. On exit, overwritten
 *                      with B - A*X. Complex.
 * @param[in]     ldb   Leading dimension of B.
 * @param[out]    resid The residual.
 */
void zptt02(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* const restrict D,
    const c128* const restrict E,
    const c128* const restrict X,
    const INT ldx,
    c128* const restrict B,
    const INT ldb,
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
    f64 anorm = zlanht("1", n, D, E);

    /* Exit with RESID = 1/EPS if ANORM = 0. */
    f64 eps = dlamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute B - A*X. */
    zlaptm(uplo, n, nrhs, -ONE, D, E, X, ldx, ONE, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / ( norm(A) * norm(X) * EPS ). */
    *resid = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        f64 bnorm = cblas_dzasum(n, &B[j * ldb], 1);
        f64 xnorm = cblas_dzasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            *resid = fmax(*resid, ((bnorm / anorm) / xnorm) / eps);
        }
    }
}
