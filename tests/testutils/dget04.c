/**
 * @file dget04.c
 * @brief DGET04 computes the difference between a computed solution and the
 *        true solution to a system of linear equations.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern double dlamch(const char* cmach);

/**
 * DGET04 computes the difference between a computed solution and the
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
void dget04(
    const int n,
    const int nrhs,
    const double * const restrict X,
    const int ldx,
    const double * const restrict XACT,
    const int ldxact,
    const double rcond,
    double *resid)
{
    const double ZERO = 0.0;

    int i, ix, j;
    double diffnm, eps, xnorm;

    // Quick exit if n = 0 or nrhs = 0
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    // Exit with RESID = 1/EPS if RCOND is invalid
    eps = dlamch("E");
    if (rcond < ZERO) {
        *resid = 1.0 / eps;
        return;
    }

    // Compute the maximum of
    //    norm(X - XACT) / ( norm(XACT) * EPS )
    // over all the vectors X and XACT using infinity-norm
    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        // Find index of element with maximum absolute value in XACT column j
        // Note: cblas_idamax returns 0-based index
        ix = cblas_idamax(n, &XACT[j * ldxact], 1);
        xnorm = fabs(XACT[ix + j * ldxact]);

        // Compute infinity-norm of X(:,j) - XACT(:,j)
        diffnm = ZERO;
        for (i = 0; i < n; i++) {
            double diff = fabs(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (diff > diffnm) {
                diffnm = diff;
            }
        }

        if (xnorm <= ZERO) {
            if (diffnm > ZERO) {
                *resid = 1.0 / eps;
            }
        } else {
            double ratio = (diffnm / xnorm) * rcond;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }

    if (*resid * eps < 1.0) {
        *resid = *resid / eps;
    }
}
