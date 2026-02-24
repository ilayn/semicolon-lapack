/**
 * @file dpot05.c
 * @brief DPOT05 tests the error bounds from iterative refinement for
 *        symmetric positive definite systems.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DPOT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * symmetric n by n matrix.
 *
 * RESLTS[0] = test of the error bound
 *           = norm(X - XACT) / ( norm(X) * FERR )
 *
 * A large value is returned if this ratio is not less than one.
 *
 * RESLTS[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / ( (n+1)*EPS + (*) ), where
 *             (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) +abs(b))_i )
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows of the matrices X, B, and XACT. n >= 0.
 * @param[in]     nrhs    The number of columns of X, B, and XACT. nrhs >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The symmetric matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     B       Double precision array, dimension (ldb, nrhs).
 *                        The right hand side vectors.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,n).
 * @param[in]     X       Double precision array, dimension (ldx, nrhs).
 *                        The computed solution vectors.
 * @param[in]     ldx     The leading dimension of the array X. ldx >= max(1,n).
 * @param[in]     XACT    Double precision array, dimension (ldxact, nrhs).
 *                        The exact solution vectors.
 * @param[in]     ldxact  The leading dimension of the array XACT.
 *                        ldxact >= max(1,n).
 * @param[in]     ferr    Double precision array, dimension (nrhs).
 *                        The estimated forward error bounds for each solution.
 * @param[in]     berr    Double precision array, dimension (nrhs).
 *                        The componentwise relative backward error of each solution.
 * @param[out]    reslts  Double precision array, dimension (2).
 *                        RESLTS[0] = max over NRHS of norm(X-XACT)/(norm(X)*FERR)
 *                        RESLTS[1] = max over NRHS of BERR/((n+1)*EPS+(*))
 */
void dpot05(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* const restrict A,
    const INT lda,
    const f64* const restrict B,
    const INT ldb,
    const f64* const restrict X,
    const INT ldx,
    const f64* const restrict XACT,
    const INT ldxact,
    const f64* const restrict ferr,
    const f64* const restrict berr,
    f64* const restrict reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    // Quick exit if n = 0 or nrhs = 0
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    f64 eps = dlamch("E");
    f64 unfl = dlamch("S");
    f64 ovfl = ONE / unfl;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');

    // Test 1: Compute the maximum of
    //   norm(X - XACT) / ( norm(X) * FERR )
    // over all the vectors X and XACT using the infinity-norm.
    f64 errbnd = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        // Find infinity norm of X(:,j)
        INT imax = cblas_idamax(n, &X[j * ldx], 1);
        f64 xnorm = fabs(X[imax + j * ldx]);
        if (xnorm < unfl) xnorm = unfl;

        f64 diff = ZERO;
        for (INT i = 0; i < n; i++) {
            f64 d = fabs(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (d > diff) diff = d;
        }

        if (xnorm > ONE) {
            // Normal case
        } else if (diff <= ovfl * xnorm) {
            // Normal case
        } else {
            errbnd = ONE / eps;
            continue;
        }

        if (diff / xnorm <= ferr[j]) {
            f64 tmp = (diff / xnorm) / ferr[j];
            if (tmp > errbnd) errbnd = tmp;
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    // Test 2: Compute the maximum of BERR / ( (n+1)*EPS + (*) ), where
    // (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i )
    for (INT k = 0; k < nrhs; k++) {
        f64 axbi = ZERO;
        for (INT i = 0; i < n; i++) {
            f64 tmp = fabs(B[i + k * ldb]);
            if (upper) {
                // Upper: A(j,i) for j<=i stored as A(j,i), A(i,j) for j>i
                for (INT j = 0; j <= i; j++) {
                    tmp += fabs(A[j + i * lda]) * fabs(X[j + k * ldx]);
                }
                for (INT j = i + 1; j < n; j++) {
                    tmp += fabs(A[i + j * lda]) * fabs(X[j + k * ldx]);
                }
            } else {
                // Lower: A(i,j) for j<i stored as A(i,j), A(j,i) for j>=i
                for (INT j = 0; j < i; j++) {
                    tmp += fabs(A[i + j * lda]) * fabs(X[j + k * ldx]);
                }
                for (INT j = i; j < n; j++) {
                    tmp += fabs(A[j + i * lda]) * fabs(X[j + k * ldx]);
                }
            }
            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) axbi = tmp;
            }
        }
        f64 np1 = (f64)(n + 1);
        f64 denom = np1 * eps + np1 * unfl / (axbi > np1 * unfl ? axbi : np1 * unfl);
        f64 tmp2 = berr[k] / denom;
        if (k == 0) {
            reslts[1] = tmp2;
        } else {
            if (tmp2 > reslts[1]) reslts[1] = tmp2;
        }
    }
}
