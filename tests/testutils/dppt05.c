/**
 * @file dppt05.c
 * @brief DPPT05 tests the error bounds from iterative refinement for packed symmetric matrices.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

extern f64 dlamch(const char* cmach);

/**
 * DPPT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * symmetric matrix in packed storage format.
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
 * @param[in] uplo
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The number of rows of the matrices X, B, and XACT. n >= 0.
 *
 * @param[in] nrhs
 *          The number of columns of the matrices X, B, and XACT.
 *
 * @param[in] AP
 *          The upper or lower triangle of the symmetric matrix A, packed
 *          columnwise in a linear array. Dimension (n*(n+1)/2).
 *
 * @param[in] B
 *          The right hand side vectors. Dimension (ldb, nrhs).
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[in] X
 *          The computed solution vectors. Dimension (ldx, nrhs).
 *
 * @param[in] ldx
 *          The leading dimension of the array X. ldx >= max(1, n).
 *
 * @param[in] XACT
 *          The exact solution vectors. Dimension (ldxact, nrhs).
 *
 * @param[in] ldxact
 *          The leading dimension of the array XACT. ldxact >= max(1, n).
 *
 * @param[in] FERR
 *          The estimated forward error bounds. Dimension (nrhs).
 *
 * @param[in] BERR
 *          The componentwise relative backward error. Dimension (nrhs).
 *
 * @param[out] reslts
 *          RESLTS[0] = norm(X - XACT) / ( norm(X) * FERR )
 *          RESLTS[1] = BERR / ( (n+1)*EPS + (*) )
 *          Dimension (2).
 */
void dppt05(const char* uplo, const int n, const int nrhs,
            const f64* const restrict AP,
            const f64* const restrict B, const int ldb,
            const f64* const restrict X, const int ldx,
            const f64* const restrict XACT, const int ldxact,
            const f64* const restrict FERR,
            const f64* const restrict BERR,
            f64* const restrict reslts)
{
    int upper;
    int i, imax, j, jc, k;
    f64 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

    if (n <= 0 || nrhs <= 0) {
        reslts[0] = 0.0;
        reslts[1] = 0.0;
        return;
    }

    eps = dlamch("E");
    unfl = dlamch("S");
    ovfl = 1.0 / unfl;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    errbnd = 0.0;
    for (j = 0; j < nrhs; j++) {
        imax = cblas_idamax(n, &X[j * ldx], 1);
        xnorm = fabs(X[imax + j * ldx]);
        if (xnorm < unfl) xnorm = unfl;
        diff = 0.0;
        for (i = 0; i < n; i++) {
            f64 d = fabs(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (d > diff) diff = d;
        }

        if (xnorm > 1.0) {
            /* continue to check */
        } else if (diff <= ovfl * xnorm) {
            /* continue to check */
        } else {
            errbnd = 1.0 / eps;
            continue;
        }

        if (diff / xnorm <= FERR[j]) {
            f64 ratio = (diff / xnorm) / FERR[j];
            if (ratio > errbnd) errbnd = ratio;
        } else {
            errbnd = 1.0 / eps;
        }
    }
    reslts[0] = errbnd;

    for (k = 0; k < nrhs; k++) {
        axbi = 0.0;
        for (i = 0; i < n; i++) {
            tmp = fabs(B[i + k * ldb]);
            if (upper) {
                jc = (i * (i + 1)) / 2;
                for (j = 0; j <= i; j++) {
                    tmp = tmp + fabs(AP[jc + j]) * fabs(X[j + k * ldx]);
                }
                jc = jc + i + 1;
                for (j = i + 1; j < n; j++) {
                    tmp = tmp + fabs(AP[jc]) * fabs(X[j + k * ldx]);
                    jc = jc + j + 1;
                }
            } else {
                jc = i;
                for (j = 0; j < i; j++) {
                    tmp = tmp + fabs(AP[jc]) * fabs(X[j + k * ldx]);
                    jc = jc + n - j - 1;
                }
                for (j = i; j < n; j++) {
                    tmp = tmp + fabs(AP[jc + j - i]) * fabs(X[j + k * ldx]);
                }
            }
            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) axbi = tmp;
            }
        }
        f64 denom = (n + 1) * unfl;
        if (axbi > denom) denom = axbi;
        tmp = BERR[k] / ((n + 1) * eps + (n + 1) * unfl / denom);
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) reslts[1] = tmp;
        }
    }
}
