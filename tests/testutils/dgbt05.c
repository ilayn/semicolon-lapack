/**
 * @file dgbt05.c
 * @brief DGBT05 tests the error bounds from iterative refinement.
 *
 * Port of LAPACK TESTING/LIN/dgbt05.f
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "verify.h"

/**
 * DGBT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations op(A)*X = B, where A is a
 * general band matrix of order n with kl subdiagonals and ku
 * superdiagonals and op(A) = A or A**T, depending on TRANS.
 *
 * RESLTS[0] = test of the error bound
 *           = norm(X - XACT) / ( norm(X) * FERR )
 *
 * A large value is returned if this ratio is not less than one.
 *
 * RESLTS[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / ( NZ*EPS + (*) ), where
 *             (*) = NZ*UNFL / (min_i (abs(op(A))*abs(X) +abs(b))_i )
 *             and NZ = max. number of nonzeros in any row of A, plus 1
 *
 * @param[in] trans  'N': A * X = B (No transpose)
 *                   'T': A**T * X = B (Transpose)
 *                   'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in] n      The order of matrix A. n >= 0.
 * @param[in] kl     The number of subdiagonals within the band. kl >= 0.
 * @param[in] ku     The number of superdiagonals within the band. ku >= 0.
 * @param[in] nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in] AB     The original band matrix in band storage. Dimension (ldab, n).
 * @param[in] ldab   The leading dimension of AB. ldab >= kl+ku+1.
 * @param[in] B      The right hand side vectors. Dimension (ldb, nrhs).
 * @param[in] ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[in] X      The computed solution vectors. Dimension (ldx, nrhs).
 * @param[in] ldx    The leading dimension of X. ldx >= max(1, n).
 * @param[in] XACT   The exact solution vectors. Dimension (ldxact, nrhs).
 * @param[in] ldxact The leading dimension of XACT. ldxact >= max(1, n).
 * @param[in] FERR   The estimated forward error bounds. Dimension (nrhs).
 * @param[in] BERR   The componentwise backward error. Dimension (nrhs).
 * @param[out] reslts Results array, dimension (2).
 */
void dgbt05(const char* trans, int n, int kl, int ku, int nrhs,
            const f64* AB, int ldab,
            const f64* B, int ldb,
            const f64* X, int ldx,
            const f64* XACT, int ldxact,
            const f64* FERR,
            const f64* BERR,
            f64* reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Quick exit if n = 0 or nrhs = 0. */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    f64 eps = dlamch("Epsilon");
    f64 unfl = dlamch("Safe minimum");
    f64 ovfl = ONE / unfl;
    int notran = (trans[0] == 'N' || trans[0] == 'n');
    int nz = (kl + ku + 2 < n + 1) ? kl + ku + 2 : n + 1;

    /* Test 1: Compute the maximum of
       norm(X - XACT) / (norm(X) * FERR)
       over all the vectors X and XACT using the infinity-norm. */
    f64 errbnd = ZERO;
    for (int j = 0; j < nrhs; j++) {
        int imax = cblas_idamax(n, &X[j * ldx], 1);
        f64 xnorm = fabs(X[imax + j * ldx]);
        if (xnorm < unfl) {
            xnorm = unfl;
        }

        f64 diff = ZERO;
        for (int i = 0; i < n; i++) {
            f64 d = fabs(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (d > diff) {
                diff = d;
            }
        }

        if (xnorm > ONE) {
            /* Normal case */
        } else if (diff <= ovfl * xnorm) {
            /* Still ok */
        } else {
            errbnd = ONE / eps;
            continue;
        }

        if (diff / xnorm <= FERR[j]) {
            f64 temp = (diff / xnorm) / FERR[j];
            if (temp > errbnd) {
                errbnd = temp;
            }
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    /* Test 2: Compute the maximum of BERR / (NZ*EPS + (*)), where
       (*) = NZ*UNFL / (min_i (abs(op(A))*abs(X) + abs(b))_i ) */
    reslts[1] = ZERO;
    for (int k = 0; k < nrhs; k++) {
        f64 axbi = ZERO;
        for (int i = 0; i < n; i++) {
            f64 tmp = fabs(B[i + k * ldb]);
            if (notran) {
                int j_start = (i - kl > 0) ? i - kl : 0;
                int j_end = (i + ku + 1 < n) ? i + ku + 1 : n;
                for (int j = j_start; j < j_end; j++) {
                    tmp += fabs(AB[(ku + i - j) + j * ldab]) * fabs(X[j + k * ldx]);
                }
            } else {
                int j_start = (i - ku > 0) ? i - ku : 0;
                int j_end = (i + kl + 1 < n) ? i + kl + 1 : n;
                for (int j = j_start; j < j_end; j++) {
                    tmp += fabs(AB[(ku + j - i) + i * ldab]) * fabs(X[j + k * ldx]);
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
        f64 denom = nz * eps + nz * unfl / ((axbi > nz * unfl) ? axbi : nz * unfl);
        f64 tmp = BERR[k] / denom;
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) {
                reslts[1] = tmp;
            }
        }
    }
}
