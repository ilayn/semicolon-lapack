/**
 * @file cget04.c
 * @brief CGET04 computes the difference between a computed solution and the
 *        true solution to a system of linear equations.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void cget04(
    const INT n,
    const INT nrhs,
    const c64* const restrict X,
    const INT ldx,
    const c64* const restrict XACT,
    const INT ldxact,
    const f32 rcond,
    f32* resid)
{
    const f32 ZERO = 0.0f;

    INT i, ix, j;
    f32 diffnm, eps, xnorm;

    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    if (rcond < ZERO) {
        *resid = 1.0f / eps;
        return;
    }

    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        ix = (INT)cblas_icamax(n, &XACT[j * ldxact], 1);
        xnorm = cabs1f(XACT[ix + j * ldxact]);

        diffnm = ZERO;
        for (i = 0; i < n; i++) {
            f32 diff = cabs1f(X[i + j * ldx] - XACT[i + j * ldxact]);
            if (diff > diffnm) {
                diffnm = diff;
            }
        }

        if (xnorm <= ZERO) {
            if (diffnm > ZERO) {
                *resid = 1.0f / eps;
            }
        } else {
            f32 ratio = (diffnm / xnorm) * rcond;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }

    if (*resid * eps < 1.0f) {
        *resid = *resid / eps;
    }
}
