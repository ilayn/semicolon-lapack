/**
 * @file dptt05.c
 * @brief DPTT05 tests the error bounds from iterative refinement for
 *        symmetric tridiagonal systems.
 *
 * Port of LAPACK's TESTING/LIN/dptt05.f to C.
 */

#include "semicolon_lapack_double.h"
#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * DPTT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * symmetric tridiagonal matrix of order n.
 *
 * RESLTS[0] = test of the error bound
 *           = norm(X - XACT) / ( norm(X) * FERR )
 *
 * A large value is returned if this ratio is not less than one.
 *
 * RESLTS[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / ( NZ*EPS + (*) ), where
 *             (*) = NZ*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
 *             and NZ = max. number of nonzeros in any row of A, plus 1
 *
 * @param[in]     n       Order of the matrix A.
 * @param[in]     nrhs    Number of right hand sides.
 * @param[in]     D       Diagonal elements (n).
 * @param[in]     E       Subdiagonal elements (n-1).
 * @param[in]     B       Right hand side matrix (ldb x nrhs).
 * @param[in]     ldb     Leading dimension of B.
 * @param[in]     X       Computed solution (ldx x nrhs).
 * @param[in]     ldx     Leading dimension of X.
 * @param[in]     XACT    Exact solution (ldxact x nrhs).
 * @param[in]     ldxact  Leading dimension of XACT.
 * @param[in]     FERR    Forward error bounds (nrhs).
 * @param[in]     BERR    Backward error bounds (nrhs).
 * @param[out]    reslts  Array of length 2 with test results.
 */
void dptt05(
    const INT n,
    const INT nrhs,
    const f64* const restrict D,
    const f64* const restrict E,
    const f64* const restrict B,
    const INT ldb,
    const f64* const restrict X,
    const INT ldx,
    const f64* const restrict XACT,
    const INT ldxact,
    const f64* const restrict FERR,
    const f64* const restrict BERR,
    f64* const restrict reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Quick exit if N = 0 or NRHS = 0. */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    f64 eps = dlamch("E");
    f64 unfl = dlamch("S");
    f64 ovfl = ONE / unfl;
    INT nz = 4;

    /* Test 1: Compute the maximum of
       norm(X - XACT) / ( norm(X) * FERR )
       over all the vectors X and XACT using the infinity-norm. */
    f64 errbnd = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        INT imax = cblas_idamax(n, &X[j * ldx], 1);
        f64 xnorm = fmax(fabs(X[imax + j * ldx]), unfl);
        f64 diff = ZERO;
        for (INT i = 0; i < n; i++) {
            diff = fmax(diff, fabs(X[i + j * ldx] - XACT[i + j * ldxact]));
        }

        if (xnorm > ONE) {
            /* Normal case */
        } else if (diff <= ovfl * xnorm) {
            /* Normal case */
        } else {
            errbnd = ONE / eps;
            continue;
        }

        if (diff / xnorm <= FERR[j]) {
            errbnd = fmax(errbnd, (diff / xnorm) / FERR[j]);
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    /* Test 2: Compute the maximum of BERR / ( NZ*EPS + (*) ), where
       (*) = NZ*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i ) */
    for (INT k = 0; k < nrhs; k++) {
        f64 axbi;
        if (n == 1) {
            axbi = fabs(B[k * ldb]) + fabs(D[0] * X[k * ldx]);
        } else {
            axbi = fabs(B[k * ldb]) + fabs(D[0] * X[k * ldx]) +
                   fabs(E[0] * X[1 + k * ldx]);
            for (INT i = 1; i < n - 1; i++) {
                f64 tmp = fabs(B[i + k * ldb]) +
                            fabs(E[i - 1] * X[i - 1 + k * ldx]) +
                            fabs(D[i] * X[i + k * ldx]) +
                            fabs(E[i] * X[i + 1 + k * ldx]);
                axbi = fmin(axbi, tmp);
            }
            f64 tmp = fabs(B[n - 1 + k * ldb]) +
                        fabs(E[n - 2] * X[n - 2 + k * ldx]) +
                        fabs(D[n - 1] * X[n - 1 + k * ldx]);
            axbi = fmin(axbi, tmp);
        }
        f64 tmp = BERR[k] / (nz * eps + nz * unfl / fmax(axbi, nz * unfl));
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            reslts[1] = fmax(reslts[1], tmp);
        }
    }
}
