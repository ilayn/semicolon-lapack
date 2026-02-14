/**
 * @file dget07.c
 * @brief DGET07 tests the error bounds from iterative refinement.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"
#include <stdbool.h>

// Forward declarations
extern f64 dlamch(const char* cmach);

/**
 * DGET07 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations op(A)*X = B, where A is a
 * general n by n matrix and op(A) = A or A**T, depending on TRANS.
 *
 * RESLTS[0] = test of the error bound
 *           = norm(X - XACT) / ( norm(X) * FERR )
 *
 * A large value is returned if this ratio is not less than one.
 *
 * RESLTS[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / ( (n+1)*EPS + (*) ), where
 *             (*) = (n+1)*UNFL / (min_i (abs(op(A))*abs(X) +abs(b))_i )
 *
 * @param[in]     trans     Specifies the form of the system of equations.
 *                          = 'N': A * X = B (No transpose)
 *                          = 'T': A**T * X = B (Transpose)
 *                          = 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     n         The number of rows of the matrices X and XACT.
 *                          n >= 0.
 * @param[in]     nrhs      The number of columns of the matrices X and XACT.
 *                          nrhs >= 0.
 * @param[in]     A         Double precision array, dimension (lda, n).
 *                          The original n by n matrix A.
 * @param[in]     lda       The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     B         Double precision array, dimension (ldb, nrhs).
 *                          The right hand side vectors for the system of linear
 *                          equations.
 * @param[in]     ldb       The leading dimension of the array B. ldb >= max(1,n).
 * @param[in]     X         Double precision array, dimension (ldx, nrhs).
 *                          The computed solution vectors. Each vector is stored
 *                          as a column of the matrix X.
 * @param[in]     ldx       The leading dimension of the array X. ldx >= max(1,n).
 * @param[in]     XACT      Double precision array, dimension (ldxact, nrhs).
 *                          The exact solution vectors. Each vector is stored as
 *                          a column of the matrix XACT.
 * @param[in]     ldxact    The leading dimension of the array XACT.
 *                          ldxact >= max(1,n).
 * @param[in]     ferr      Double precision array, dimension (nrhs).
 *                          The estimated forward error bounds for each solution
 *                          vector X.
 * @param[in]     chkferr   Set to true to check FERR, false not to check FERR.
 *                          When the test system is ill-conditioned, the "true"
 *                          solution in XACT may be incorrect.
 * @param[in]     berr      Double precision array, dimension (nrhs).
 *                          The componentwise relative backward error of each
 *                          solution vector.
 * @param[out]    reslts    Double precision array, dimension (2).
 *                          The maximum over the nrhs solution vectors of the
 *                          ratios:
 *                          RESLTS[0] = norm(X - XACT) / ( norm(X) * FERR )
 *                          RESLTS[1] = BERR / ( (n+1)*EPS + (*) )
 */
void dget07(
    const char* trans,
    const int n,
    const int nrhs,
    const f64 * const restrict A,
    const int lda,
    const f64 * const restrict B,
    const int ldb,
    const f64 * const restrict X,
    const int ldx,
    const f64 * const restrict XACT,
    const int ldxact,
    const f64 * const restrict ferr,
    const bool chkferr,
    const f64 * const restrict berr,
    f64 * const restrict reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i, imax, j, k;
    f64 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;
    int notran = (trans[0] == 'N' || trans[0] == 'n');

    // Quick exit if n = 0 or nrhs = 0
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    eps = dlamch("E");
    unfl = dlamch("S");
    ovfl = ONE / unfl;

    // Test 1: Compute the maximum of
    //    norm(X - XACT) / ( norm(X) * FERR )
    // over all the vectors X and XACT using the infinity-norm.
    errbnd = ZERO;
    if (chkferr) {
        for (j = 0; j < nrhs; j++) {
            // Find index of max element in X(:,j)
            // cblas_idamax returns 0-based index
            imax = cblas_idamax(n, &X[j * ldx], 1);
            xnorm = fabs(X[imax + j * ldx]);
            if (xnorm < unfl) {
                xnorm = unfl;
            }

            // Compute infinity-norm of X(:,j) - XACT(:,j)
            diff = ZERO;
            for (i = 0; i < n; i++) {
                f64 d = fabs(X[i + j * ldx] - XACT[i + j * ldxact]);
                if (d > diff) {
                    diff = d;
                }
            }

            if (xnorm > ONE) {
                // Normal case
                if (diff / xnorm <= ferr[j]) {
                    f64 ratio = (diff / xnorm) / ferr[j];
                    if (ratio > errbnd) {
                        errbnd = ratio;
                    }
                } else {
                    errbnd = ONE / eps;
                }
            } else if (diff <= ovfl * xnorm) {
                // Avoid overflow
                if (diff / xnorm <= ferr[j]) {
                    f64 ratio = (diff / xnorm) / ferr[j];
                    if (ratio > errbnd) {
                        errbnd = ratio;
                    }
                } else {
                    errbnd = ONE / eps;
                }
            } else {
                errbnd = ONE / eps;
            }
        }
    }
    reslts[0] = errbnd;

    // Test 2: Compute the maximum of BERR / ( (n+1)*EPS + (*) ), where
    // (*) = (n+1)*UNFL / (min_i (abs(op(A))*abs(X) + abs(b))_i )
    for (k = 0; k < nrhs; k++) {
        for (i = 0; i < n; i++) {
            tmp = fabs(B[i + k * ldb]);
            if (notran) {
                for (j = 0; j < n; j++) {
                    tmp += fabs(A[i + j * lda]) * fabs(X[j + k * ldx]);
                }
            } else {
                for (j = 0; j < n; j++) {
                    tmp += fabs(A[j + i * lda]) * fabs(X[j + k * ldx]);
                }
            }
            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) {
                    axbi = tmp;
                }
            }
        }

        f64 denom = (n + 1) * eps + (n + 1) * unfl / fmax(axbi, (n + 1) * unfl);
        tmp = berr[k] / denom;

        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) {
                reslts[1] = tmp;
            }
        }
    }
}
