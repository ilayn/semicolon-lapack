/**
 * @file cptt02.c
 * @brief CPTT02 computes the residual for the solution to a Hermitian
 *        tridiagonal system of equations.
 *
 * Port of LAPACK's TESTING/LIN/cptt02.f to C.
 */

#include "semicolon_lapack_complex_single.h"
#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * CPTT02 computes the residual for the solution to a Hermitian
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
void cptt02(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f32* const restrict D,
    const c64* const restrict E,
    const c64* const restrict X,
    const INT ldx,
    c64* const restrict B,
    const INT ldb,
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
    f32 anorm = clanht("1", n, D, E);

    /* Exit with RESID = 1/EPS if ANORM = 0. */
    f32 eps = slamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute B - A*X. */
    claptm(uplo, n, nrhs, -ONE, D, E, X, ldx, ONE, B, ldb);

    /* Compute the maximum over the number of right hand sides of
       norm(B - A*X) / ( norm(A) * norm(X) * EPS ). */
    *resid = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        f32 bnorm = cblas_scasum(n, &B[j * ldb], 1);
        f32 xnorm = cblas_scasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            *resid = fmaxf(*resid, ((bnorm / anorm) / xnorm) / eps);
        }
    }
}
