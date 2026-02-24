/**
 * @file zgtt05.c
 * @brief ZGTT05 tests the error bounds from iterative refinement for the
 *        computed solution to a tridiagonal system of equations.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZGTT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * general tridiagonal matrix of order n and op(A) = A or A**T,
 * depending on trans.
 *
 * reslts[0] = test of the error bound
 *           = norm(X - XACT) / (norm(X) * FERR)
 *
 * A large value is returned if this ratio is not less than one.
 *
 * reslts[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / (NZ*EPS + (*)), where
 *             (*) = NZ*UNFL / (min_i (abs(op(A))*abs(X) + abs(b))_i)
 *             and NZ = max. number of nonzeros in any row of A, plus 1
 *
 * @param[in]  trans  Specifies the form of the system of equations.
 *                    = 'N': A * X = B     (No transpose)
 *                    = 'T': A**T * X = B  (Transpose)
 *                    = 'C': A**H * X = B  (Conjugate transpose)
 * @param[in]  n      The number of rows of the matrices X and XACT. n >= 0.
 * @param[in]  nrhs   The number of columns of the matrices X and XACT. nrhs >= 0.
 * @param[in]  DL     The (n-1) sub-diagonal elements of A. Array of dimension (n-1).
 * @param[in]  D      The diagonal elements of A. Array of dimension (n).
 * @param[in]  DU     The (n-1) super-diagonal elements of A. Array of dimension (n-1).
 * @param[in]  B      The right hand side vectors. Array of dimension (ldb, nrhs).
 * @param[in]  ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[in]  X      The computed solution vectors. Array of dimension (ldx, nrhs).
 * @param[in]  ldx    The leading dimension of X. ldx >= max(1, n).
 * @param[in]  XACT   The exact solution vectors. Array of dimension (ldxact, nrhs).
 * @param[in]  ldxact The leading dimension of XACT. ldxact >= max(1, n).
 * @param[in]  FERR   The estimated forward error bounds for each solution vector.
 *                    Array of dimension (nrhs).
 * @param[in]  BERR   The componentwise relative backward error of each solution.
 *                    Array of dimension (nrhs).
 * @param[out] reslts The maximum ratios:
 *                    reslts[0] = norm(X - XACT) / (norm(X) * FERR)
 *                    reslts[1] = BERR / (NZ*EPS + (*))
 *                    Array of dimension (2).
 */
void zgtt05(const char* trans, const INT n, const INT nrhs,
            const c128* DL, const c128* D, const c128* DU,
            const c128* B, const INT ldb,
            const c128* X, const INT ldx,
            const c128* XACT, const INT ldxact,
            const f64* FERR,
            const f64* BERR,
            f64* reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT notran;
    INT i, imax, j, k, nz;
    f64 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

    /* Quick exit if n = 0 or nrhs = 0 */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    eps = dlamch("Epsilon");
    unfl = dlamch("Safe minimum");
    ovfl = ONE / unfl;
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nz = 4;

    /*
     * Test 1: Compute the maximum of
     *    norm(X - XACT) / (norm(X) * FERR)
     * over all vectors X and XACT using the infinity-norm.
     */
    errbnd = ZERO;
    for (j = 0; j < nrhs; j++) {
        imax = cblas_izamax(n, X + j * ldx, 1);
        xnorm = cabs1(X[imax + j * ldx]);
        if (xnorm < unfl) xnorm = unfl;

        diff = ZERO;
        for (i = 0; i < n; i++) {
            f64 d = cabs1(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (d > diff) diff = d;
        }

        if (xnorm > ONE) {
            /* Normal case */
        } else if (diff <= ovfl * xnorm) {
            /* Safe to compute diff/xnorm */
        } else {
            errbnd = ONE / eps;
            continue;
        }

        if (diff / xnorm <= FERR[j]) {
            f64 ratio = (diff / xnorm) / FERR[j];
            if (ratio > errbnd) errbnd = ratio;
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    /*
     * Test 2: Compute the maximum of BERR / (NZ*EPS + (*)), where
     * (*) = NZ*UNFL / (min_i (abs(op(A))*abs(X) + abs(b))_i)
     */
    reslts[1] = ZERO;
    for (k = 0; k < nrhs; k++) {
        if (notran) {
            if (n == 1) {
                axbi = cabs1(B[k * ldb]) + cabs1(D[0]) * cabs1(X[k * ldx]);
            } else {
                axbi = cabs1(B[k * ldb]) + cabs1(D[0]) * cabs1(X[k * ldx])
                       + cabs1(DU[0]) * cabs1(X[1 + k * ldx]);
                for (i = 1; i < n - 1; i++) {
                    tmp = cabs1(B[i + k * ldb])
                          + cabs1(DL[i - 1]) * cabs1(X[(i - 1) + k * ldx])
                          + cabs1(D[i]) * cabs1(X[i + k * ldx])
                          + cabs1(DU[i]) * cabs1(X[(i + 1) + k * ldx]);
                    if (tmp < axbi) axbi = tmp;
                }
                tmp = cabs1(B[(n - 1) + k * ldb])
                      + cabs1(DL[n - 2]) * cabs1(X[(n - 2) + k * ldx])
                      + cabs1(D[n - 1]) * cabs1(X[(n - 1) + k * ldx]);
                if (tmp < axbi) axbi = tmp;
            }
        } else {
            if (n == 1) {
                axbi = cabs1(B[k * ldb]) + cabs1(D[0]) * cabs1(X[k * ldx]);
            } else {
                axbi = cabs1(B[k * ldb]) + cabs1(D[0]) * cabs1(X[k * ldx])
                       + cabs1(DL[0]) * cabs1(X[1 + k * ldx]);
                for (i = 1; i < n - 1; i++) {
                    tmp = cabs1(B[i + k * ldb])
                          + cabs1(DU[i - 1]) * cabs1(X[(i - 1) + k * ldx])
                          + cabs1(D[i]) * cabs1(X[i + k * ldx])
                          + cabs1(DL[i]) * cabs1(X[(i + 1) + k * ldx]);
                    if (tmp < axbi) axbi = tmp;
                }
                tmp = cabs1(B[(n - 1) + k * ldb])
                      + cabs1(DU[n - 2]) * cabs1(X[(n - 2) + k * ldx])
                      + cabs1(D[n - 1]) * cabs1(X[(n - 1) + k * ldx]);
                if (tmp < axbi) axbi = tmp;
            }
        }

        f64 denom = nz * eps + nz * unfl / (axbi > nz * unfl ? axbi : nz * unfl);
        tmp = BERR[k] / denom;

        if (tmp > reslts[1]) {
            reslts[1] = tmp;
        }
    }
}
