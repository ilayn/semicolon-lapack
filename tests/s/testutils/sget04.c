/**
 * @file sget04.c
 * @brief SGET04 computes the difference between a computed solution and the
 *        true solution to a system of linear equations.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern f32 slamch(const char* cmach);

/**
 * SGET04 computes the difference between a computed solution and the
 * true solution to a system of linear equations.
 *
 *    RESID = ( norm(X-XACT) * RCOND ) / ( norm(XACT) * EPS ),
 * where RCOND is the reciprocal of the condition number and EPS is the
 * machine epsilon.
 *
 * @param[in]     n       The number of rows of the matrices X and XACT.
 *                        n >= 0.
 * @param[in]     nrhs    The number of columns of the matrices X and XACT.
 *                        nrhs >= 0.
 * @param[in]     X       Double precision array, dimension (ldx, nrhs).
 *                        The computed solution vectors. Each vector is stored
 *                        as a column of the matrix X.
 * @param[in]     ldx     The leading dimension of the array X. ldx >= max(1,n).
 * @param[in]     XACT    Double precision array, dimension (ldxact, nrhs).
 *                        The exact solution vectors. Each vector is stored as
 *                        a column of the matrix XACT.
 * @param[in]     ldxact  The leading dimension of the array XACT.
 *                        ldxact >= max(1,n).
 * @param[in]     rcond   The reciprocal of the condition number of the
 *                        coefficient matrix in the system of equations.
 * @param[out]    resid   The maximum over the nrhs solution vectors of
 *                        ( norm(X-XACT) * RCOND ) / ( norm(XACT) * EPS )
 */
void sget04(
    const int n,
    const int nrhs,
    const f32 * const restrict X,
    const int ldx,
    const f32 * const restrict XACT,
    const int ldxact,
    const f32 rcond,
    f32 *resid)
{
    const f32 ZERO = 0.0f;

    int i, ix, j;
    f32 diffnm, eps, xnorm;

    // Quick exit if n = 0 or nrhs = 0
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    // Exit with RESID = 1/EPS if RCOND is invalid
    eps = slamch("E");
    if (rcond < ZERO) {
        *resid = 1.0f / eps;
        return;
    }

    // Compute the maximum of
    //    norm(X - XACT) / ( norm(XACT) * EPS )
    // over all the vectors X and XACT using infinity-norm
    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        // Find index of element with maximum absolute value in XACT column j
        // Note: cblas_idamax returns 0-based index
        ix = cblas_isamax(n, &XACT[j * ldxact], 1);
        xnorm = fabsf(XACT[ix + j * ldxact]);

        // Compute infinity-norm of X(:,j) - XACT(:,j)
        diffnm = ZERO;
        for (i = 0; i < n; i++) {
            f32 diff = fabsf(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (diff > diffnm) {
                diffnm = diff;
            }
        }

        if (xnorm <= ZERO) {
            if (diffnm > ZERO) {
                *resid = 1.0f / eps;
            }
        } else {
            f32 ratio = (diffnm / xnorm) * rcond;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }

    if (*resid * eps < 1.0f) {
        *resid = *resid / eps;
    }
}
