/**
 * @file dtrt05.c
 * @brief DTRT05 tests the error bounds from iterative refinement for triangular systems.
 *
 * Port of LAPACK TESTING/LIN/dtrt05.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * DTRT05 tests the error bounds from iterative refinement for the
 * computed solution to a system of equations A*X = B, where A is a
 * triangular n by n matrix.
 *
 * RESLTS[0] = test of the error bound
 *           = norm(X - XACT) / ( norm(X) * FERR )
 *
 * A large value is returned if this ratio is not less than one.
 *
 * RESLTS[1] = residual from the iterative refinement routine
 *           = the maximum of BERR / ( (n+1)*EPS + (*) ), where
 *             (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i )
 *
 * @param[in]     uplo    'U' for upper triangular, 'L' for lower triangular.
 * @param[in]     trans   'N' for A*X = B, 'T' or 'C' for A'*X = B.
 * @param[in]     diag    'N' for non-unit triangular, 'U' for unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     A       Array (lda, n). The triangular matrix A.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1,n).
 * @param[in]     XACT    Array (ldxact, nrhs). The exact solution vectors.
 * @param[in]     ldxact  The leading dimension of XACT. ldxact >= max(1,n).
 * @param[in]     ferr    Array (nrhs). The estimated forward error bounds.
 * @param[in]     berr    Array (nrhs). The componentwise relative backward error.
 * @param[out]    reslts  Array (2). The test results.
 */
void dtrt05(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f64* A, const INT lda,
            const f64* B, const INT ldb, const f64* X, const INT ldx,
            const f64* XACT, const INT ldxact,
            const f64* ferr, const f64* berr, f64* reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, j, k, ifu, imax;
    f64 eps, unfl, ovfl, tmp, diff, xnorm, errbnd, axbi;
    INT upper, notran, unit;

    /* Quick exit if n = 0 or nrhs = 0 */
    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    eps = dlamch("E");
    unfl = dlamch("S");
    ovfl = ONE / unfl;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    unit = (diag[0] == 'U' || diag[0] == 'u');

    /* Test 1: Compute the maximum of
       norm(X - XACT) / ( norm(X) * FERR )
       over all the vectors X and XACT using the infinity-norm. */
    errbnd = ZERO;
    for (j = 0; j < nrhs; j++) {
        /* Find max element of X(:,j) */
        imax = cblas_idamax(n, &X[j * ldx], 1);
        xnorm = fabs(X[j * ldx + imax]);
        if (xnorm < unfl) xnorm = unfl;

        diff = ZERO;
        for (i = 0; i < n; i++) {
            f64 d = fabs(X[j * ldx + i] - XACT[j * ldxact + i]);
            if (d > diff) diff = d;
        }

        if (xnorm > ONE) {
            /* Normal case */
        } else if (diff <= ovfl * xnorm) {
            /* Normal case */
        } else {
            errbnd = ONE / eps;
            continue;
        }

        if (diff / xnorm <= ferr[j]) {
            f64 ratio = (diff / xnorm) / ferr[j];
            if (ratio > errbnd) errbnd = ratio;
        } else {
            errbnd = ONE / eps;
        }
    }
    reslts[0] = errbnd;

    /* Test 2: Compute the maximum of BERR / ( (n+1)*EPS + (*) ), where
       (*) = (n+1)*UNFL / (min_i (abs(A)*abs(X) + abs(b))_i ) */
    ifu = unit ? 1 : 0;

    reslts[1] = ZERO;
    for (k = 0; k < nrhs; k++) {
        for (i = 0; i < n; i++) {
            tmp = fabs(B[k * ldb + i]);

            if (upper) {
                if (!notran) {
                    /* A'*X = B, upper triangular: column i of A is row i */
                    for (j = 0; j < i - ifu + 1; j++) {
                        tmp += fabs(A[i * lda + j]) * fabs(X[k * ldx + j]);
                    }
                    if (unit)
                        tmp += fabs(X[k * ldx + i]);
                } else {
                    /* A*X = B, upper triangular: row i of A */
                    if (unit)
                        tmp += fabs(X[k * ldx + i]);
                    for (j = i + ifu; j < n; j++) {
                        tmp += fabs(A[j * lda + i]) * fabs(X[k * ldx + j]);
                    }
                }
            } else {
                /* Lower triangular */
                if (notran) {
                    /* A*X = B, lower triangular: row i of A */
                    for (j = 0; j < i - ifu + 1; j++) {
                        tmp += fabs(A[j * lda + i]) * fabs(X[k * ldx + j]);
                    }
                    if (unit)
                        tmp += fabs(X[k * ldx + i]);
                } else {
                    /* A'*X = B, lower triangular: column i of A is row i */
                    if (unit)
                        tmp += fabs(X[k * ldx + i]);
                    for (j = i + ifu; j < n; j++) {
                        tmp += fabs(A[i * lda + j]) * fabs(X[k * ldx + j]);
                    }
                }
            }

            if (i == 0) {
                axbi = tmp;
            } else {
                if (tmp < axbi) axbi = tmp;
            }
        }

        f64 denom = (n + 1) * eps + (n + 1) * unfl / fmax(axbi, (n + 1) * unfl);
        tmp = berr[k] / denom;

        if (k == 0) {
            reslts[1] = tmp;
        } else {
            if (tmp > reslts[1]) reslts[1] = tmp;
        }
    }
}
