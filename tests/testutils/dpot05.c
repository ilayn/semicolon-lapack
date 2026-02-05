/**
 * @file dpot05.c
 * @brief DPOT05 tests the error bounds from iterative refinement for
 *        symmetric positive definite systems.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern double dlamch(const char* cmach);

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
    const int n,
    const int nrhs,
    const double* const restrict A,
    const int lda,
    const double* const restrict B,
    const int ldb,
    const double* const restrict X,
    const int ldx,
    const double* const restrict XACT,
    const int ldxact,
    const double* const restrict ferr,
    const double* const restrict berr,
    double* const restrict reslts)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    // Quick exit if n = 0 or nrhs = 0
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    double eps = dlamch("E");
    double unfl = dlamch("S");
    double ovfl = ONE / unfl;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');

    // Test 1: Compute the maximum of
    //   norm(X - XACT) / ( norm(X) * FERR )
    // over all the vectors X and XACT using the infinity-norm.
    double errbnd = ZERO;
    for (int j = 0; j < nrhs; j++) {
        // Find infinity norm of X(:,j)
        int imax = cblas_idamax(n, &X[j * ldx], 1);
        double xnorm = fabs(X[imax + j * ldx]);
        if (xnorm < unfl) xnorm = unfl;

        double diff = ZERO;
        for (int i = 0; i < n; i++) {
            double d = fabs(X[i + j * ldx] - XACT[i + j * ldxact]);
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
            double tmp = (diff / xnorm) / ferr[j];
            if (tmp > errbnd) errbnd = tmp;
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    // Test 2: Compute the maximum of BERR / ( (n+1)*EPS + (*) ), where
    // (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i )
    for (int k = 0; k < nrhs; k++) {
        double axbi = ZERO;
        for (int i = 0; i < n; i++) {
            double tmp = fabs(B[i + k * ldb]);
            if (upper) {
                // Upper: A(j,i) for j<=i stored as A(j,i), A(i,j) for j>i
                for (int j = 0; j <= i; j++) {
                    tmp += fabs(A[j + i * lda]) * fabs(X[j + k * ldx]);
                }
                for (int j = i + 1; j < n; j++) {
                    tmp += fabs(A[i + j * lda]) * fabs(X[j + k * ldx]);
                }
            } else {
                // Lower: A(i,j) for j<i stored as A(i,j), A(j,i) for j>=i
                for (int j = 0; j < i; j++) {
                    tmp += fabs(A[i + j * lda]) * fabs(X[j + k * ldx]);
                }
                for (int j = i; j < n; j++) {
                    tmp += fabs(A[j + i * lda]) * fabs(X[j + k * ldx]);
                }
            }
            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) axbi = tmp;
            }
        }
        double np1 = (double)(n + 1);
        double denom = np1 * eps + np1 * unfl / (axbi > np1 * unfl ? axbi : np1 * unfl);
        double tmp2 = berr[k] / denom;
        if (k == 0) {
            reslts[1] = tmp2;
        } else {
            if (tmp2 > reslts[1]) reslts[1] = tmp2;
        }
    }
}
