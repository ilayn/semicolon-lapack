/**
 * @file zget07.c
 * @brief ZGET07 tests the error bounds from iterative refinement.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zget07(
    const char* trans,
    const INT n,
    const INT nrhs,
    const c128* const restrict A,
    const INT lda,
    const c128* const restrict B,
    const INT ldb,
    const c128* const restrict X,
    const INT ldx,
    const c128* const restrict XACT,
    const INT ldxact,
    const f64* const restrict ferr,
    const INT chkferr,
    const f64* const restrict berr,
    f64* const restrict reslts)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT notran;
    INT i, imax, j, k;
    f64 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    eps = dlamch("E");
    unfl = dlamch("S");
    ovfl = ONE / unfl;
    notran = (trans[0] == 'N' || trans[0] == 'n');

    errbnd = ZERO;
    if (chkferr) {
        for (j = 0; j < nrhs; j++) {
            imax = (INT)cblas_izamax(n, &X[j * ldx], 1);
            xnorm = cabs1(X[imax + j * ldx]);
            if (xnorm < unfl) {
                xnorm = unfl;
            }

            diff = ZERO;
            for (i = 0; i < n; i++) {
                f64 d = cabs1(X[i + j * ldx] - XACT[i + j * ldxact]);
                if (d > diff) {
                    diff = d;
                }
            }

            if (xnorm > ONE) {
                if (diff / xnorm <= ferr[j]) {
                    f64 ratio = (diff / xnorm) / ferr[j];
                    if (ratio > errbnd) {
                        errbnd = ratio;
                    }
                } else {
                    errbnd = ONE / eps;
                }
            } else if (diff <= ovfl * xnorm) {
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

    for (k = 0; k < nrhs; k++) {
        for (i = 0; i < n; i++) {
            tmp = cabs1(B[i + k * ldb]);
            if (notran) {
                for (j = 0; j < n; j++) {
                    tmp += cabs1(A[i + j * lda]) * cabs1(X[j + k * ldx]);
                }
            } else {
                for (j = 0; j < n; j++) {
                    tmp += cabs1(A[j + i * lda]) * cabs1(X[j + k * ldx]);
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
