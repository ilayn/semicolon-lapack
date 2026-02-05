/**
 * @file dptt05.c
 * @brief DPTT05 tests the error bounds from iterative refinement for
 *        symmetric tridiagonal systems.
 *
 * Port of LAPACK's TESTING/LIN/dptt05.f to C.
 */

#include "semicolon_lapack_double.h"
#include "verify.h"
#include <cblas.h>
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
    const int n,
    const int nrhs,
    const double* const restrict D,
    const double* const restrict E,
    const double* const restrict B,
    const int ldb,
    const double* const restrict X,
    const int ldx,
    const double* const restrict XACT,
    const int ldxact,
    const double* const restrict FERR,
    const double* const restrict BERR,
    double* const restrict reslts)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    /* Quick exit if N = 0 or NRHS = 0. */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    double eps = dlamch("E");
    double unfl = dlamch("S");
    double ovfl = ONE / unfl;
    int nz = 4;

    /* Test 1: Compute the maximum of
       norm(X - XACT) / ( norm(X) * FERR )
       over all the vectors X and XACT using the infinity-norm. */
    double errbnd = ZERO;
    for (int j = 0; j < nrhs; j++) {
        int imax = cblas_idamax(n, &X[j * ldx], 1);
        double xnorm = fmax(fabs(X[imax + j * ldx]), unfl);
        double diff = ZERO;
        for (int i = 0; i < n; i++) {
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
    for (int k = 0; k < nrhs; k++) {
        double axbi;
        if (n == 1) {
            axbi = fabs(B[k * ldb]) + fabs(D[0] * X[k * ldx]);
        } else {
            axbi = fabs(B[k * ldb]) + fabs(D[0] * X[k * ldx]) +
                   fabs(E[0] * X[1 + k * ldx]);
            for (int i = 1; i < n - 1; i++) {
                double tmp = fabs(B[i + k * ldb]) +
                            fabs(E[i - 1] * X[i - 1 + k * ldx]) +
                            fabs(D[i] * X[i + k * ldx]) +
                            fabs(E[i] * X[i + 1 + k * ldx]);
                axbi = fmin(axbi, tmp);
            }
            double tmp = fabs(B[n - 1 + k * ldb]) +
                        fabs(E[n - 2] * X[n - 2 + k * ldx]) +
                        fabs(D[n - 1] * X[n - 1 + k * ldx]);
            axbi = fmin(axbi, tmp);
        }
        double tmp = BERR[k] / (nz * eps + nz * unfl / fmax(axbi, nz * unfl));
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            reslts[1] = fmax(reslts[1], tmp);
        }
    }
}
