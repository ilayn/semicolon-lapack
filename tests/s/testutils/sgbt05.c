/**
 * @file sgbt05.c
 * @brief SGBT05 tests the error bounds from iterative refinement.
 *
 * Port of LAPACK TESTING/LIN/sgbt05.f
 */

#include <math.h>
#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SGBT05 tests the error bounds from iterative refinement for the
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
void sgbt05(const char* trans, INT n, INT kl, INT ku, INT nrhs,
            const f32* AB, INT ldab,
            const f32* B, INT ldb,
            const f32* X, INT ldx,
            const f32* XACT, INT ldxact,
            const f32* FERR,
            const f32* BERR,
            f32* reslts)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    /* Quick exit if n = 0 or nrhs = 0. */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    f32 eps = slamch("Epsilon");
    f32 unfl = slamch("Safe minimum");
    f32 ovfl = ONE / unfl;
    INT notran = (trans[0] == 'N' || trans[0] == 'n');
    INT nz = (kl + ku + 2 < n + 1) ? kl + ku + 2 : n + 1;

    /* Test 1: Compute the maximum of
       norm(X - XACT) / (norm(X) * FERR)
       over all the vectors X and XACT using the infinity-norm. */
    f32 errbnd = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        INT imax = cblas_isamax(n, &X[j * ldx], 1);
        f32 xnorm = fabsf(X[imax + j * ldx]);
        if (xnorm < unfl) {
            xnorm = unfl;
        }

        f32 diff = ZERO;
        for (INT i = 0; i < n; i++) {
            f32 d = fabsf(X[i + j * ldx] - XACT[i + j * ldxact]);
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
            f32 temp = (diff / xnorm) / FERR[j];
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
    for (INT k = 0; k < nrhs; k++) {
        f32 axbi = ZERO;
        for (INT i = 0; i < n; i++) {
            f32 tmp = fabsf(B[i + k * ldb]);
            if (notran) {
                INT j_start = (i - kl > 0) ? i - kl : 0;
                INT j_end = (i + ku + 1 < n) ? i + ku + 1 : n;
                for (INT j = j_start; j < j_end; j++) {
                    tmp += fabsf(AB[(ku + i - j) + j * ldab]) * fabsf(X[j + k * ldx]);
                }
            } else {
                INT j_start = (i - ku > 0) ? i - ku : 0;
                INT j_end = (i + kl + 1 < n) ? i + kl + 1 : n;
                for (INT j = j_start; j < j_end; j++) {
                    tmp += fabsf(AB[(ku + j - i) + i * ldab]) * fabsf(X[j + k * ldx]);
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
        f32 denom = nz * eps + nz * unfl / ((axbi > nz * unfl) ? axbi : nz * unfl);
        f32 tmp = BERR[k] / denom;
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) {
                reslts[1] = tmp;
            }
        }
    }
}
