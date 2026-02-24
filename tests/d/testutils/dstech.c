/**
 * @file dstech.c
 * @brief DSTECH checks whether eigenvalues of a tridiagonal matrix are
 *        accurate by expanding each eigenvalue into an interval and
 *        verifying the Sturm count.
 */

#include <math.h>
#include "verify.h"

/* Forward declaration */
/**
 * DSTECH checks whether EIG[0],...,EIG[n-1] are accurate eigenvalues
 * of the tridiagonal matrix T with diagonal A and off-diagonal B.
 *
 * It expands each eigenvalue into an interval [EIG-EPS, EIG+EPS],
 * merges overlapping intervals, and uses Sturm sequences to verify
 * that each interval contains the correct number of eigenvalues.
 *
 * @param[in]     n    Dimension of the tridiagonal matrix.
 * @param[in]     A    Double array (n). Diagonal entries of T.
 * @param[in]     B    Double array (n-1). Off-diagonal entries of T.
 * @param[in]     eig  Double array (n). Purported eigenvalues.
 * @param[in]     tol  Error tolerance (multiple of machine precision).
 * @param[out]    work Double array (n). Workspace.
 * @param[out]    info 0 if all eigenvalues correct; >0 if interval
 *                     containing the info-th eigenvalue has wrong count.
 */
void dstech(const INT n, const f64* const restrict A,
            const f64* const restrict B,
            const f64* const restrict eig, const f64 tol,
            f64* const restrict work, INT* info)
{
    const f64 ZERO = 0.0;

    *info = 0;
    if (n == 0)
        return;
    if (n < 0) {
        *info = -1;
        return;
    }
    if (tol < ZERO) {
        *info = -5;
        return;
    }

    /* Get machine constants */
    f64 eps = dlamch("E") * dlamch("B");  /* Epsilon * Base = ulp */
    f64 unflep = dlamch("S") / eps;
    eps = tol * eps;

    /* Compute maximum absolute eigenvalue */
    f64 mx = fabs(eig[0]);
    for (INT i = 1; i < n; i++) {
        mx = fmax(mx, fabs(eig[i]));
    }
    eps = fmax(eps * mx, unflep);

    /* Sort eigenvalues from eig into work (selection sort ascending) */
    for (INT i = 0; i < n; i++) {
        work[i] = eig[i];
    }
    for (INT i = 0; i < n - 1; i++) {
        INT isub = 0;
        f64 emin = work[0];
        for (INT j = 1; j < n - i; j++) {
            if (work[j] < emin) {
                isub = j;
                emin = work[j];
            }
        }
        if (isub != n - 1 - i) {
            work[isub] = work[n - 1 - i];
            work[n - 1 - i] = emin;
        }
    }
    /* After sorting: work[0] is largest, work[n-1] is smallest.
     * This is descending order. The Fortran code uses 1-based TPNT/BPNT
     * where TPNT starts at 1 (largest) and moves right. */

    /* tpnt/bpnt: indices into sorted work array.
     * In the Fortran code, tpnt starts at 1 (top/right of interval)
     * and bpnt at 1 (bottom/left of interval).
     * After the selection sort above, work is in descending order. */
    INT tpnt = 0;
    INT bpnt = 0;

    /* Loop over all intervals */
    while (tpnt < n) {
        f64 upper = work[tpnt] + eps;
        f64 lower = work[bpnt] - eps;

        /* Merge overlapping intervals */
        while (bpnt < n - 1) {
            f64 tuppr = work[bpnt + 1] + eps;
            if (tuppr < lower)
                break;
            bpnt++;
            lower = work[bpnt] - eps;
        }

        /* Count eigenvalues in interval [lower, upper] */
        INT numl, numu;
        dstect(n, A, B, lower, &numl);
        dstect(n, A, B, upper, &numu);
        INT count = numu - numl;
        if (count != bpnt - tpnt + 1) {
            /* Wrong number of eigenvalues in interval */
            *info = tpnt + 1;  /* 1-based index */
            return;
        }
        tpnt = bpnt + 1;
        bpnt = tpnt;
    }
}
