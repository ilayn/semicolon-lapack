/**
 * @file dlarrk.c
 * @brief DLARRK computes one eigenvalue of a symmetric tridiagonal matrix T to suitable accuracy.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLARRK computes one eigenvalue of a symmetric tridiagonal
 * matrix T to suitable accuracy. This is an auxiliary code to be
 * called from DSTEMR.
 *
 * To avoid overflow, the matrix must be scaled so that its
 * largest element is no greater than overflow^(1/2) * underflow^(1/4)
 * in absolute value, and for greatest accuracy, it should not be much
 * smaller than that.
 *
 * See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
 * Matrix", Report CS41, Computer Science Dept., Stanford
 * University, July 21, 1966.
 *
 * @param[in]     n       The order of the tridiagonal matrix T. N >= 0.
 * @param[in]     iw      The index of the eigenvalue to be returned.
 * @param[in]     gl      A lower bound on the eigenvalue.
 * @param[in]     gu      An upper bound on the eigenvalue.
 * @param[in]     D       Double precision array, dimension (N).
 *                        The N diagonal elements of the tridiagonal matrix T.
 * @param[in]     E2      Double precision array, dimension (N-1).
 *                        The (N-1) squared off-diagonal elements of the tridiagonal matrix T.
 * @param[in]     pivmin  The minimum pivot allowed in the Sturm sequence for T.
 * @param[in]     reltol  The minimum relative width of an interval.
 * @param[out]    w       The eigenvalue approximation.
 * @param[out]    werr    The error bound on the corresponding eigenvalue approximation in W.
 * @param[out]    info
 *                         - = 0: Eigenvalue converged.
 *                         - = -1: Eigenvalue did NOT converge.
 */
void dlarrk(const int n, const int iw, const f64 gl, const f64 gu,
            const f64* const restrict D, const f64* const restrict E2,
            const f64 pivmin, const f64 reltol,
            f64* w, f64* werr, int* info)
{
    /* FUDGE = 2, a "fudge factor" to widen the Gershgorin intervals */
    const f64 FUDGE = 2.0;
    const f64 HALF = 0.5;
    const f64 TWO = 2.0;

    int i, it, itmax, negcnt;
    f64 atoli, eps, left, mid, right, rtoli, tmp1, tmp2, tnorm;

    /* Quick return if possible */
    if (n <= 0) {
        *info = 0;
        return;
    }

    /* Get machine constants */
    eps = dlamch("P");

    tnorm = fmax(fabs(gl), fabs(gu));
    rtoli = reltol;
    atoli = FUDGE * TWO * pivmin;

    itmax = (int)((log(tnorm + pivmin) - log(pivmin)) / log(TWO)) + 2;

    *info = -1;

    left = gl - FUDGE * tnorm * eps * n - FUDGE * TWO * pivmin;
    right = gu + FUDGE * tnorm * eps * n + FUDGE * TWO * pivmin;
    it = 0;

    for (;;) {
        /* Check if interval converged or maximum number of iterations reached */
        tmp1 = fabs(right - left);
        tmp2 = fmax(fabs(right), fabs(left));
        if (tmp1 < fmax(atoli, fmax(pivmin, rtoli * tmp2))) {
            *info = 0;
            break;
        }
        if (it > itmax) {
            break;
        }

        /* Count number of negative pivots for mid-point */
        it = it + 1;
        mid = HALF * (left + right);
        negcnt = 0;
        tmp1 = D[0] - mid;
        if (fabs(tmp1) < pivmin) {
            tmp1 = -pivmin;
        }
        if (tmp1 <= 0.0) {
            negcnt = negcnt + 1;
        }

        for (i = 1; i < n; i++) {
            tmp1 = D[i] - E2[i - 1] / tmp1 - mid;
            if (fabs(tmp1) < pivmin) {
                tmp1 = -pivmin;
            }
            if (tmp1 <= 0.0) {
                negcnt = negcnt + 1;
            }
        }

        if (negcnt >= iw) {
            right = mid;
        } else {
            left = mid;
        }
    }

    /* Converged or maximum number of iterations reached */
    *w = HALF * (left + right);
    *werr = HALF * fabs(right - left);
}
