/**
 * @file cgtt02.c
 * @brief CGTT02 computes the residual for the solution to a tridiagonal
 *        system of equations.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CGTT02 computes the residual for the solution to a tridiagonal
 * system of equations:
 *    RESID = norm(B - op(A)*X) / (norm(op(A)) * norm(X) * EPS),
 * where EPS is the machine epsilon.
 *
 * @param[in]     trans Specifies the form of the residual.
 *                      = 'N': B - A * X    (No transpose)
 *                      = 'T': B - A**T * X (Transpose)
 *                      = 'C': B - A**H * X (Conjugate transpose)
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     DL    The (n-1) sub-diagonal elements of A. Array of dimension (n-1).
 * @param[in]     D     The diagonal elements of A. Array of dimension (n).
 * @param[in]     DU    The (n-1) super-diagonal elements of A. Array of dimension (n-1).
 * @param[in]     X     The computed solution vectors X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of X. ldx >= max(1, n).
 * @param[in,out] B     On entry, the right hand side vectors.
 *                      On exit, B is overwritten with the difference B - op(A)*X.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[out]    resid The maximum residual over all right hand sides:
 *                      norm(B - op(A)*X) / (norm(op(A)) * norm(X) * EPS)
 */
void cgtt02(const char* trans, const INT n, const INT nrhs,
            const c64* DL, const c64* D, const c64* DU,
            const c64* X, const INT ldx,
            c64* B, const INT ldb,
            f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;

    INT j;
    f32 anorm, bnorm, eps, xnorm;

    /* Quick exit if n = 0 or nrhs = 0 */
    *resid = ZERO;
    if (n <= 0 || nrhs == 0) {
        return;
    }

    /* Compute the maximum over the number of right hand sides of
       norm(B - op(A)*X) / ( norm(op(A)) * norm(X) * EPS ). */
    if (trans[0] == 'N' || trans[0] == 'n') {
        anorm = clangt("1", n, DL, D, DU);
    } else {
        anorm = clangt("I", n, DL, D, DU);
    }

    /* Exit with resid = 1/EPS if anorm = 0. */
    eps = slamch("Epsilon");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute B - op(A)*X and store in B. */
    clagtm(trans, n, nrhs, NEG_ONE, DL, D, DU, X, ldx, ONE, B, ldb);

    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_scasum(n, B + j * ldb, 1);
        xnorm = cblas_scasum(n, X + j * ldx, 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f32 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }
}
