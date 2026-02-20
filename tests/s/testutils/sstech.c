/**
 * @file sstech.c
 * @brief SSTECH checks whether eigenvalues of a tridiagonal matrix are
 *        accurate by expanding each eigenvalue into an interval and
 *        verifying the Sturm count.
 */

#include <math.h>
#include "verify.h"

/* Forward declaration */
extern f32 slamch(const char* cmach);

/**
 * SSTECH checks whether EIG[0],...,EIG[n-1] are accurate eigenvalues
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
void sstech(const int n, const f32* const restrict A,
            const f32* const restrict B,
            const f32* const restrict eig, const f32 tol,
            f32* const restrict work, int* info)
{
    const f32 ZERO = 0.0f;

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
    f32 eps = slamch("E") * slamch("B");  /* Epsilon * Base = ulp */
    f32 unflep = slamch("S") / eps;
    eps = tol * eps;

    /* Compute maximum absolute eigenvalue */
    f32 mx = fabsf(eig[0]);
    for (int i = 1; i < n; i++) {
        mx = fmaxf(mx, fabsf(eig[i]));
    }
    eps = fmaxf(eps * mx, unflep);

    /* Sort eigenvalues from eig into work (selection sort ascending) */
    for (int i = 0; i < n; i++) {
        work[i] = eig[i];
    }
    for (int i = 0; i < n - 1; i++) {
        int isub = 0;
        f32 emin = work[0];
        for (int j = 1; j < n - i; j++) {
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
    int tpnt = 0;
    int bpnt = 0;

    /* Loop over all intervals */
    while (tpnt < n) {
        f32 upper = work[tpnt] + eps;
        f32 lower = work[bpnt] - eps;

        /* Merge overlapping intervals */
        while (bpnt < n - 1) {
            f32 tuppr = work[bpnt + 1] + eps;
            if (tuppr < lower)
                break;
            bpnt++;
            lower = work[bpnt] - eps;
        }

        /* Count eigenvalues in interval [lower, upper] */
        int numl, numu;
        sstect(n, A, B, lower, &numl);
        sstect(n, A, B, upper, &numu);
        int count = numu - numl;
        if (count != bpnt - tpnt + 1) {
            /* Wrong number of eigenvalues in interval */
            *info = tpnt + 1;  /* 1-based index */
            return;
        }
        tpnt = bpnt + 1;
        bpnt = tpnt;
    }
}
