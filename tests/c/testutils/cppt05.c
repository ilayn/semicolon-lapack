/**
 * @file cppt05.c
 * @brief CPPT05 tests the error bounds from iterative refinement for
 *        Hermitian packed matrices.
 *
 * Port of LAPACK TESTING/LIN/cppt05.f
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CPPT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * Hermitian matrix in packed storage format.
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
 *          The upper or lower triangle of the Hermitian matrix A, packed
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
void cppt05(const char* uplo, const INT n, const INT nrhs,
            const c64* const restrict AP,
            const c64* const restrict B, const INT ldb,
            const c64* const restrict X, const INT ldx,
            const c64* const restrict XACT, const INT ldxact,
            const f32* const restrict FERR,
            const f32* const restrict BERR,
            f32* const restrict reslts)
{
    INT upper;
    INT i, imax, j, jc, k;
    f32 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

    if (n <= 0 || nrhs <= 0) {
        reslts[0] = 0.0f;
        reslts[1] = 0.0f;
        return;
    }

    eps = slamch("E");
    unfl = slamch("S");
    ovfl = 1.0f / unfl;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    errbnd = 0.0f;
    for (j = 0; j < nrhs; j++) {
        imax = cblas_icamax(n, &X[j * ldx], 1);
        xnorm = cabs1f(X[imax + j * ldx]);
        if (xnorm < unfl) xnorm = unfl;
        diff = 0.0f;
        for (i = 0; i < n; i++) {
            f32 d = cabs1f(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (d > diff) diff = d;
        }

        if (xnorm > 1.0f) {
            /* continue to check */
        } else if (diff <= ovfl * xnorm) {
            /* continue to check */
        } else {
            errbnd = 1.0f / eps;
            continue;
        }

        if (diff / xnorm <= FERR[j]) {
            f32 ratio = (diff / xnorm) / FERR[j];
            if (ratio > errbnd) errbnd = ratio;
        } else {
            errbnd = 1.0f / eps;
        }
    }
    reslts[0] = errbnd;

    for (k = 0; k < nrhs; k++) {
        axbi = 0.0f;
        for (i = 0; i < n; i++) {
            tmp = cabs1f(B[i + k * ldb]);
            if (upper) {
                jc = (i * (i + 1)) / 2;
                for (j = 0; j < i; j++) {
                    tmp = tmp + cabs1f(AP[jc + j]) * cabs1f(X[j + k * ldx]);
                }
                tmp = tmp + fabsf(crealf(AP[jc + i])) * cabs1f(X[i + k * ldx]);
                jc = jc + i + i + 1;
                for (j = i + 1; j < n; j++) {
                    tmp = tmp + cabs1f(AP[jc]) * cabs1f(X[j + k * ldx]);
                    jc = jc + j + 1;
                }
            } else {
                jc = i;
                for (j = 0; j < i; j++) {
                    tmp = tmp + cabs1f(AP[jc]) * cabs1f(X[j + k * ldx]);
                    jc = jc + n - j - 1;
                }
                tmp = tmp + fabsf(crealf(AP[jc])) * cabs1f(X[i + k * ldx]);
                for (j = i + 1; j < n; j++) {
                    tmp = tmp + cabs1f(AP[jc + j - i]) * cabs1f(X[j + k * ldx]);
                }
            }
            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) axbi = tmp;
            }
        }
        f32 denom = (n + 1) * unfl;
        if (axbi > denom) denom = axbi;
        tmp = BERR[k] / ((n + 1) * eps + (n + 1) * unfl / denom);
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) reslts[1] = tmp;
        }
    }
}
