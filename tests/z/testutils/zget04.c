/**
 * @file zget04.c
 * @brief ZGET04 computes the difference between a computed solution and the
 *        true solution to a system of linear equations.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zget04(
    const INT n,
    const INT nrhs,
    const c128* const restrict X,
    const INT ldx,
    const c128* const restrict XACT,
    const INT ldxact,
    const f64 rcond,
    f64* resid)
{
    const f64 ZERO = 0.0;

    INT i, ix, j;
    f64 diffnm, eps, xnorm;

    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    if (rcond < ZERO) {
        *resid = 1.0 / eps;
        return;
    }

    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        ix = (INT)cblas_izamax(n, &XACT[j * ldxact], 1);
        xnorm = cabs1(XACT[ix + j * ldxact]);

        diffnm = ZERO;
        for (i = 0; i < n; i++) {
            f64 diff = cabs1(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (diff > diffnm) {
                diffnm = diff;
            }
        }

        if (xnorm <= ZERO) {
            if (diffnm > ZERO) {
                *resid = 1.0 / eps;
            }
        } else {
            f64 ratio = (diffnm / xnorm) * rcond;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }

    if (*resid * eps < 1.0) {
        *resid = *resid / eps;
    }
}
