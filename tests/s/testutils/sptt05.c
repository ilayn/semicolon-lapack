/**
 * @file sptt05.c
 * @brief SPTT05 tests the error bounds from iterative refinement for
 *        symmetric tridiagonal systems.
 *
 * Port of LAPACK's TESTING/LIN/sptt05.f to C.
 */

#include "semicolon_lapack_single.h"
#include "verify.h"
#include <cblas.h>
#include <math.h>

/**
 * SPTT05 tests the error bounds from iterative refinement for the
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
void sptt05(
    const int n,
    const int nrhs,
    const f32* const restrict D,
    const f32* const restrict E,
    const f32* const restrict B,
    const int ldb,
    const f32* const restrict X,
    const int ldx,
    const f32* const restrict XACT,
    const int ldxact,
    const f32* const restrict FERR,
    const f32* const restrict BERR,
    f32* const restrict reslts)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    /* Quick exit if N = 0 or NRHS = 0. */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    f32 eps = slamch("E");
    f32 unfl = slamch("S");
    f32 ovfl = ONE / unfl;
    int nz = 4;

    /* Test 1: Compute the maximum of
       norm(X - XACT) / ( norm(X) * FERR )
       over all the vectors X and XACT using the infinity-norm. */
    f32 errbnd = ZERO;
    for (int j = 0; j < nrhs; j++) {
        int imax = cblas_isamax(n, &X[j * ldx], 1);
        f32 xnorm = fmaxf(fabsf(X[imax + j * ldx]), unfl);
        f32 diff = ZERO;
        for (int i = 0; i < n; i++) {
            diff = fmaxf(diff, fabsf(X[i + j * ldx] - XACT[i + j * ldxact]));
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
            errbnd = fmaxf(errbnd, (diff / xnorm) / FERR[j]);
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    /* Test 2: Compute the maximum of BERR / ( NZ*EPS + (*) ), where
       (*) = NZ*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i ) */
    for (int k = 0; k < nrhs; k++) {
        f32 axbi;
        if (n == 1) {
            axbi = fabsf(B[k * ldb]) + fabsf(D[0] * X[k * ldx]);
        } else {
            axbi = fabsf(B[k * ldb]) + fabsf(D[0] * X[k * ldx]) +
                   fabsf(E[0] * X[1 + k * ldx]);
            for (int i = 1; i < n - 1; i++) {
                f32 tmp = fabsf(B[i + k * ldb]) +
                            fabsf(E[i - 1] * X[i - 1 + k * ldx]) +
                            fabsf(D[i] * X[i + k * ldx]) +
                            fabsf(E[i] * X[i + 1 + k * ldx]);
                axbi = fminf(axbi, tmp);
            }
            f32 tmp = fabsf(B[n - 1 + k * ldb]) +
                        fabsf(E[n - 2] * X[n - 2 + k * ldx]) +
                        fabsf(D[n - 1] * X[n - 1 + k * ldx]);
            axbi = fminf(axbi, tmp);
        }
        f32 tmp = BERR[k] / (nz * eps + nz * unfl / fmaxf(axbi, nz * unfl));
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            reslts[1] = fmaxf(reslts[1], tmp);
        }
    }
}
