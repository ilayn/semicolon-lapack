/**
 * @file cptt05.c
 * @brief CPTT05 tests the error bounds from iterative refinement for
 *        Hermitian tridiagonal systems.
 *
 * Port of LAPACK's TESTING/LIN/cptt05.f to C.
 */

#include "semicolon_lapack_complex_single.h"
#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * CPTT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * Hermitian tridiagonal matrix of order n.
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
 * @param[in]     D       Diagonal elements (n). Real.
 * @param[in]     E       Subdiagonal elements (n-1). Complex.
 * @param[in]     B       Right hand side matrix (ldb x nrhs). Complex.
 * @param[in]     ldb     Leading dimension of B.
 * @param[in]     X       Computed solution (ldx x nrhs). Complex.
 * @param[in]     ldx     Leading dimension of X.
 * @param[in]     XACT    Exact solution (ldxact x nrhs). Complex.
 * @param[in]     ldxact  Leading dimension of XACT.
 * @param[in]     FERR    Forward error bounds (nrhs). Real.
 * @param[in]     BERR    Backward error bounds (nrhs). Real.
 * @param[out]    reslts  Array of length 2 with test results. Real.
 */
void cptt05(
    const INT n,
    const INT nrhs,
    const f32* const restrict D,
    const c64* const restrict E,
    const c64* const restrict B,
    const INT ldb,
    const c64* const restrict X,
    const INT ldx,
    const c64* const restrict XACT,
    const INT ldxact,
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
    INT nz = 4;

    /* Test 1: Compute the maximum of
       norm(X - XACT) / ( norm(X) * FERR )
       over all the vectors X and XACT using the infinity-norm. */
    f32 errbnd = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        INT imax = cblas_icamax(n, &X[j * ldx], 1);
        f32 xnorm = fmaxf(cabs1f(X[imax + j * ldx]), unfl);
        f32 diff = ZERO;
        for (INT i = 0; i < n; i++) {
            diff = fmaxf(diff, cabs1f(X[i + j * ldx] - XACT[i + j * ldxact]));
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
    for (INT k = 0; k < nrhs; k++) {
        f32 axbi;
        if (n == 1) {
            axbi = cabs1f(B[k * ldb]) + cabs1f(D[0] * X[k * ldx]);
        } else {
            axbi = cabs1f(B[k * ldb]) + cabs1f(D[0] * X[k * ldx]) +
                   cabs1f(E[0]) * cabs1f(X[1 + k * ldx]);
            for (INT i = 1; i < n - 1; i++) {
                f32 tmp = cabs1f(B[i + k * ldb]) +
                            cabs1f(E[i - 1]) * cabs1f(X[i - 1 + k * ldx]) +
                            cabs1f(D[i] * X[i + k * ldx]) +
                            cabs1f(E[i]) * cabs1f(X[i + 1 + k * ldx]);
                axbi = fminf(axbi, tmp);
            }
            f32 tmp = cabs1f(B[n - 1 + k * ldb]) +
                        cabs1f(E[n - 2]) * cabs1f(X[n - 2 + k * ldx]) +
                        cabs1f(D[n - 1] * X[n - 1 + k * ldx]);
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
