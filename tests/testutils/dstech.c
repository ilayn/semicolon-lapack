/**
 * @file dstech.c
 * @brief DSTECH checks whether eigenvalues of a tridiagonal matrix are
 *        accurate by expanding each eigenvalue into an interval and
 *        verifying the Sturm count.
 */

#include <math.h>
#include "verify.h"

/* Forward declaration */
extern f64 dlamch(const char* cmach);

/**
 * DSTECT counts the number of eigenvalues of a tridiagonal matrix T
 * which are less than or equal to SHIFT. Uses the Sturm sequence
 * property with Kahan's scaling for numerical stability.
 *
 * @param[in]  n     Dimension of the tridiagonal matrix.
 * @param[in]  A     Double array (n). Diagonal entries.
 * @param[in]  B     Double array (n-1). Off-diagonal entries.
 * @param[in]  shift The shift value.
 * @param[out] num   Number of eigenvalues <= shift.
 */
static void dstect(const int n, const f64* const restrict A,
                   const f64* const restrict B,
                   const f64 shift, int* num)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 THREE = 3.0;

    f64 unfl = dlamch("S");
    f64 ovfl = dlamch("O");

    /* Find largest entry */
    f64 mx = fabs(A[0]);
    for (int i = 0; i < n - 1; i++) {
        mx = fmax(mx, fmax(fabs(A[i + 1]), fabs(B[i])));
    }

    /* Handle easy cases */
    if (shift >= THREE * mx) {
        *num = n;
        return;
    }
    if (shift < -THREE * mx) {
        *num = 0;
        return;
    }

    /* Compute scale factors as in Kahan's report */
    f64 sun = sqrt(unfl);
    f64 ssun = sqrt(sun);
    f64 sov = sqrt(ovfl);
    f64 tom = ssun * sov;

    f64 m1, m2;
    if (mx <= ONE) {
        m1 = ONE / mx;
        m2 = tom;
    } else {
        m1 = ONE;
        m2 = tom / mx;
    }

    /* Begin counting via Sturm sequence */
    *num = 0;
    f64 sshift = (shift * m1) * m2;
    f64 u = (A[0] * m1) * m2 - sshift;
    if (u <= sun) {
        if (u <= ZERO) {
            (*num)++;
            if (u > -sun)
                u = -sun;
        } else {
            u = sun;
        }
    }
    for (int i = 1; i < n; i++) {
        f64 tmp = (B[i - 1] * m1) * m2;
        u = ((A[i] * m1) * m2 - tmp * (tmp / u)) - sshift;
        if (u <= sun) {
            if (u <= ZERO) {
                (*num)++;
                if (u > -sun)
                    u = -sun;
            } else {
                u = sun;
            }
        }
    }
}

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
void dstech(const int n, const f64* const restrict A,
            const f64* const restrict B,
            const f64* const restrict eig, const f64 tol,
            f64* const restrict work, int* info)
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
    for (int i = 1; i < n; i++) {
        mx = fmax(mx, fabs(eig[i]));
    }
    eps = fmax(eps * mx, unflep);

    /* Sort eigenvalues from eig into work (selection sort ascending) */
    for (int i = 0; i < n; i++) {
        work[i] = eig[i];
    }
    for (int i = 0; i < n - 1; i++) {
        int isub = 0;
        f64 emin = work[0];
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
        int numl, numu;
        dstect(n, A, B, lower, &numl);
        dstect(n, A, B, upper, &numu);
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
