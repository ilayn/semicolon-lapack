/**
 * @file stpt05.c
 * @brief STPT05 tests the error bounds from iterative refinement for a
 *        triangular system when A is in packed format.
 *
 * Port of LAPACK TESTING/LIN/stpt05.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f32 slamch(const char* cmach);

/**
 * STPT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * triangular matrix in packed storage format.
 *
 * RESLTS[0] = test of the error bound
 *           = norm(X - XACT) / (norm(X) * FERR)
 *
 * RESLTS[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / ((n+1)*EPS + (*)), where
 *             (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i)
 *
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     trans   = 'N': A * X = B; = 'T' or 'C': A' * X = B.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     AP      Array (n*(n+1)/2). The triangular matrix A in packed storage.
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1, n).
 * @param[in]     XACT    Array (ldxact, nrhs). The exact solution vectors.
 * @param[in]     ldxact  The leading dimension of XACT. ldxact >= max(1, n).
 * @param[in]     ferr    Array (nrhs). The estimated forward error bounds.
 * @param[in]     berr    Array (nrhs). The componentwise relative backward errors.
 * @param[out]    reslts  Array (2). The test results.
 */
void stpt05(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f32* AP, const f32* B, const int ldb,
            const f32* X, const int ldx,
            const f32* XACT, const int ldxact,
            const f32* ferr, const f32* berr,
            f32* reslts)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    int notran, unit, upper;
    int i, ifu, imax, j, jc, k;
    f32 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

    /* Quick exit if N = 0 or NRHS = 0 */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    eps = slamch("E");
    unfl = slamch("S");
    ovfl = ONE / unfl;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    unit = (diag[0] == 'U' || diag[0] == 'u');

    /* Test 1: Compute the maximum of
     * norm(X - XACT) / (norm(X) * FERR)
     * over all the vectors X and XACT using the infinity-norm. */
    errbnd = ZERO;
    for (j = 0; j < nrhs; j++) {
        imax = cblas_isamax(n, &X[j * ldx], 1);
        xnorm = fmaxf(fabsf(X[imax + j * ldx]), unfl);
        diff = ZERO;
        for (i = 0; i < n; i++) {
            f32 d = fabsf(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (d > diff) diff = d;
        }

        if (xnorm > ONE) {
            /* Continue to ratio computation */
        } else if (diff <= ovfl * xnorm) {
            /* Continue to ratio computation */
        } else {
            errbnd = ONE / eps;
            continue;
        }

        if (diff / xnorm <= ferr[j]) {
            f32 r = (diff / xnorm) / ferr[j];
            if (r > errbnd) errbnd = r;
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    /* Test 2: Compute the maximum of BERR / ((n+1)*EPS + (*)), where
     * (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i) */
    ifu = unit ? 1 : 0;
    reslts[1] = ZERO;

    for (k = 0; k < nrhs; k++) {
        axbi = ZERO;
        for (i = 0; i < n; i++) {
            tmp = fabsf(B[i + k * ldb]);
            if (upper) {
                jc = (i * (i + 1)) / 2;
                if (!notran) {
                    for (j = 0; j < i + 1 - ifu; j++) {
                        tmp += fabsf(AP[jc + j]) * fabsf(X[j + k * ldx]);
                    }
                    if (unit) {
                        tmp += fabsf(X[i + k * ldx]);
                    }
                } else {
                    jc += i;
                    if (unit) {
                        tmp += fabsf(X[i + k * ldx]);
                        jc += i + 1;
                    }
                    for (j = i + ifu; j < n; j++) {
                        tmp += fabsf(AP[jc]) * fabsf(X[j + k * ldx]);
                        jc += j + 1;
                    }
                }
            } else {
                if (notran) {
                    jc = i;
                    for (j = 0; j < i + 1 - ifu; j++) {
                        tmp += fabsf(AP[jc]) * fabsf(X[j + k * ldx]);
                        jc += n - j - 1;
                    }
                    if (unit) {
                        tmp += fabsf(X[i + k * ldx]);
                    }
                } else {
                    jc = i * (n - i) + (i * (i + 1)) / 2;
                    if (unit) {
                        tmp += fabsf(X[i + k * ldx]);
                    }
                    for (j = i + ifu; j < n; j++) {
                        tmp += fabsf(AP[jc + j - i]) * fabsf(X[j + k * ldx]);
                    }
                }
            }
            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) axbi = tmp;
            }
        }
        tmp = berr[k] / ((n + 1) * eps + (n + 1) * unfl /
              fmaxf(axbi, (n + 1) * unfl));
        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) reslts[1] = tmp;
        }
    }
}
