/**
 * @file slarrk.c
 * @brief SLARRK computes one eigenvalue of a symmetric tridiagonal matrix T to suitable accuracy.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLARRK computes one eigenvalue of a symmetric tridiagonal
 * matrix T to suitable accuracy. This is an auxiliary code to be
 * called from SSTEMR.
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
void slarrk(const int n, const int iw, const f32 gl, const f32 gu,
            const f32* restrict D, const f32* restrict E2,
            const f32 pivmin, const f32 reltol,
            f32* w, f32* werr, int* info)
{
    /* FUDGE = 2, a "fudge factor" to widen the Gershgorin intervals */
    const f32 FUDGE = 2.0f;
    const f32 HALF = 0.5f;
    const f32 TWO = 2.0f;

    int i, it, itmax, negcnt;
    f32 atoli, eps, left, mid, right, rtoli, tmp1, tmp2, tnorm;

    /* Quick return if possible */
    if (n <= 0) {
        *info = 0;
        return;
    }

    /* Get machine constants */
    eps = slamch("P");

    tnorm = fmaxf(fabsf(gl), fabsf(gu));
    rtoli = reltol;
    atoli = FUDGE * TWO * pivmin;

    itmax = (int)((logf(tnorm + pivmin) - logf(pivmin)) / logf(TWO)) + 2;

    *info = -1;

    left = gl - FUDGE * tnorm * eps * n - FUDGE * TWO * pivmin;
    right = gu + FUDGE * tnorm * eps * n + FUDGE * TWO * pivmin;
    it = 0;

    for (;;) {
        /* Check if interval converged or maximum number of iterations reached */
        tmp1 = fabsf(right - left);
        tmp2 = fmaxf(fabsf(right), fabsf(left));
        if (tmp1 < fmaxf(atoli, fmaxf(pivmin, rtoli * tmp2))) {
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
        if (fabsf(tmp1) < pivmin) {
            tmp1 = -pivmin;
        }
        if (tmp1 <= 0.0f) {
            negcnt = negcnt + 1;
        }

        for (i = 1; i < n; i++) {
            tmp1 = D[i] - E2[i - 1] / tmp1 - mid;
            if (fabsf(tmp1) < pivmin) {
                tmp1 = -pivmin;
            }
            if (tmp1 <= 0.0f) {
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
    *werr = HALF * fabsf(right - left);
}
