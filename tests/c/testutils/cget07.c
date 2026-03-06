/**
 * @file cget07.c
 * @brief CGET07 tests the error bounds from iterative refinement.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void cget07(
    const char* trans,
    const INT n,
    const INT nrhs,
    const c64* const restrict A,
    const INT lda,
    const c64* const restrict B,
    const INT ldb,
    const c64* const restrict X,
    const INT ldx,
    const c64* const restrict XACT,
    const INT ldxact,
    const f32* const restrict ferr,
    const INT chkferr,
    const f32* const restrict berr,
    f32* const restrict reslts)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT notran;
    INT i, imax, j, k;
    f32 axbi, diff, eps, errbnd, ovfl, tmp, unfl, xnorm;

    if (n <= 0 || nrhs <= 0) {
        reslts[0] = ZERO;
        reslts[1] = ZERO;
        return;
    }

    eps = slamch("E");
    unfl = slamch("S");
    ovfl = ONE / unfl;
    notran = (trans[0] == 'N' || trans[0] == 'n');

    errbnd = ZERO;
    if (chkferr) {
        for (j = 0; j < nrhs; j++) {
            imax = (INT)cblas_icamax(n, &X[j * ldx], 1);
            xnorm = cabs1f(X[imax + j * ldx]);
            if (xnorm < unfl) {
                xnorm = unfl;
            }

            diff = ZERO;
            for (i = 0; i < n; i++) {
                f32 d = cabs1f(X[i + j * ldx] - XACT[i + j * ldxact]);
                if (d > diff) {
                    diff = d;
                }
            }

            if (xnorm > ONE) {
                if (diff / xnorm <= ferr[j]) {
                    f32 ratio = (diff / xnorm) / ferr[j];
                    if (ratio > errbnd) {
                        errbnd = ratio;
                    }
                } else {
                    errbnd = ONE / eps;
                }
            } else if (diff <= ovfl * xnorm) {
                if (diff / xnorm <= ferr[j]) {
                    f32 ratio = (diff / xnorm) / ferr[j];
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
            tmp = cabs1f(B[i + k * ldb]);
            if (notran) {
                for (j = 0; j < n; j++) {
                    tmp += cabs1f(A[i + j * lda]) * cabs1f(X[j + k * ldx]);
                }
            } else {
                for (j = 0; j < n; j++) {
                    tmp += cabs1f(A[j + i * lda]) * cabs1f(X[j + k * ldx]);
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

        f32 denom = (n + 1) * eps + (n + 1) * unfl / fmaxf(axbi, (n + 1) * unfl);
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
