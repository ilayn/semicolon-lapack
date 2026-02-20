/**
 * @file sgtt02.c
 * @brief SGTT02 computes the residual for the solution to a tridiagonal
 *        system of equations.
 */

#include <math.h>
#include "verify.h"
#include <cblas.h>

/* Forward declarations */
extern f32 slamch(const char* cmach);
extern f32 slangt(const char* norm, const int n,
                     const f32* const restrict DL,
                     const f32* const restrict D,
                     const f32* const restrict DU);
extern void slagtm(const char* trans, const int n, const int nrhs,
                   const f32 alpha,
                   const f32* const restrict DL,
                   const f32* const restrict D,
                   const f32* const restrict DU,
                   const f32* const restrict X, const int ldx,
                   const f32 beta,
                   f32* const restrict B, const int ldb);

/**
 * SGTT02 computes the residual for the solution to a tridiagonal
 * system of equations:
 *    RESID = norm(B - op(A)*X) / (norm(op(A)) * norm(X) * EPS),
 * where EPS is the machine epsilon.
 * The norm used is the 1-norm.
 *
 * @param[in]     trans Specifies the form of the residual.
 *                      = 'N': B - A * X    (No transpose)
 *                      = 'T': B - A**T * X (Transpose)
 *                      = 'C': B - A**H * X (Conjugate transpose = Transpose)
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
void sgtt02(
    const char* trans,
    const int n,
    const int nrhs,
    const f32 * const restrict DL,
    const f32 * const restrict D,
    const f32 * const restrict DU,
    const f32 * const restrict X,
    const int ldx,
    f32 * const restrict B,
    const int ldb,
    f32 *resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 NEG_ONE = -1.0f;

    int j;
    f32 anorm, bnorm, eps, xnorm;

    /* Quick exit if n = 0 or nrhs = 0 */
    *resid = ZERO;
    if (n <= 0 || nrhs == 0) {
        return;
    }

    /* Compute the norm of op(A) */
    if (trans[0] == 'N' || trans[0] == 'n') {
        anorm = slangt("1", n, DL, D, DU);
    } else {
        anorm = slangt("I", n, DL, D, DU);
    }

    /* Exit with resid = 1/EPS if anorm = 0 */
    eps = slamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute B - op(A)*X and store in B */
    slagtm(trans, n, nrhs, NEG_ONE, DL, D, DU, X, ldx, ONE, B, ldb);

    /* Compute the maximum residual over all right hand sides */
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_sasum(n, B + j * ldb, 1);
        xnorm = cblas_sasum(n, X + j * ldx, 1);
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
