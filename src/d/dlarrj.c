/**
 * @file dlarrj.c
 * @brief DLARRJ performs refinement of the initial estimates of the eigenvalues of the matrix T.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * Given the initial eigenvalue approximations of T, DLARRJ
 * does bisection to refine the eigenvalues of T,
 * W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial
 * guesses for these eigenvalues are input in W, the corresponding estimate
 * of the error in these guesses in WERR. During bisection, intervals
 * [left, right] are maintained by storing their mid-points and
 * semi-widths in the arrays W and WERR respectively.
 *
 * @param[in]     n       The order of the matrix. N >= 0.
 * @param[in]     D       Double precision array, dimension (N).
 *                        The N diagonal elements of T.
 * @param[in]     E2      Double precision array, dimension (N-1).
 *                        The squares of the (N-1) subdiagonal elements of T.
 * @param[in]     ifirst  The index of the first eigenvalue to be computed (0-based).
 * @param[in]     ilast   The index of the last eigenvalue to be computed (0-based).
 * @param[in]     rtol    Tolerance for the convergence of the bisection intervals.
 *                        An interval [LEFT,RIGHT] has converged if
 *                        RIGHT-LEFT < RTOL*MAX(|LEFT|,|RIGHT|).
 * @param[in]     offset  Offset for the arrays W and WERR, i.e., the IFIRST-OFFSET
 *                        through ILAST-OFFSET elements of these arrays are to be used.
 * @param[in,out] W       Double precision array, dimension (N).
 *                        On input, W( IFIRST-OFFSET ) through W( ILAST-OFFSET ) are
 *                        estimates of the eigenvalues of L D L^T indexed IFIRST through
 *                        ILAST.
 *                        On output, these estimates are refined.
 * @param[in,out] werr    Double precision array, dimension (N).
 *                        On input, WERR( IFIRST-OFFSET ) through WERR( ILAST-OFFSET ) are
 *                        the errors in the estimates of the corresponding elements in W.
 *                        On output, these errors are refined.
 * @param[out]    work    Double precision array, dimension (2*N).
 *                        Workspace.
 * @param[out]    iwork   Integer array, dimension (2*N).
 *                        Workspace.
 * @param[in]     pivmin  The minimum pivot in the Sturm sequence for T.
 * @param[in]     spdiam  The spectral diameter of T.
 * @param[out]    info
 *                           Error flag.
 */
void dlarrj(const int n, const double* D, const double* E2,
            const int ifirst, const int ilast, const double rtol,
            const int offset, double* W, double* werr,
            double* work, int* iwork, const double pivmin,
            const double spdiam, int* info)
{
    const double TWO = 2.0;
    const double HALF = 0.5;

    int cnt, i, i1, i2, ii, iter, j, k, next, nint, olnint, p, prev, savi1;
    int maxitr;
    double dplus, fac, left, mid, right, s, tmp, width;

    *info = 0;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    maxitr = (int)((log(spdiam + pivmin) - log(pivmin)) / log(TWO)) + 2;

    /*
     * Initialize unconverged intervals in [ work[2*i], work[2*i+1] ].
     * The Sturm Count, Count( work[2*i] ) is arranged to be i, while
     * Count( work[2*i+1] ) is stored in iwork[2*i+1]. The integer iwork[2*i]
     * for an unconverged interval is set to the index of the next unconverged
     * interval, and is -1 or 0 for a converged interval. Thus a linked
     * list of unconverged intervals is set up.
     */

    i1 = ifirst;
    i2 = ilast;
    /* The number of unconverged intervals */
    nint = 0;
    /* The last unconverged interval found */
    prev = -1;

    for (i = i1; i <= i2; i++) {
        k = 2 * i;
        ii = i - offset;
        left = W[ii] - werr[ii];
        mid = W[ii];
        right = W[ii] + werr[ii];
        width = right - mid;
        tmp = fmax(fabs(left), fabs(right));

        /* The following test prevents the test of converged intervals */
        if (width < rtol * tmp) {
            /* This interval has already converged and does not need refinement. */
            /* (Note that the gaps might change through refining the */
            /*  eigenvalues, however, they can only get bigger.) */
            /* Remove it from the list. */
            iwork[k] = -1;
            /* Make sure that i1 always points to the first unconverged interval */
            if ((i == i1) && (i < i2)) {
                i1 = i + 1;
            }
            if ((prev >= i1) && (i <= i2)) {
                iwork[2 * prev] = i + 1;
            }
        } else {
            /* unconverged interval found */
            prev = i;
            /* Make sure that [left,right] contains the desired eigenvalue */

            /* Do while( cnt(left) > i ) */
            fac = 1.0;
            for (;;) {
                cnt = 0;
                s = left;
                dplus = D[0] - s;
                if (dplus < 0.0) cnt = cnt + 1;
                for (j = 1; j < n; j++) {
                    dplus = D[j] - s - E2[j - 1] / dplus;
                    if (dplus < 0.0) cnt = cnt + 1;
                }
                if (cnt > i) {
                    left = left - werr[ii] * fac;
                    fac = TWO * fac;
                } else {
                    break;
                }
            }

            /* Do while( cnt(right) < i+1 ) */
            fac = 1.0;
            for (;;) {
                cnt = 0;
                s = right;
                dplus = D[0] - s;
                if (dplus < 0.0) cnt = cnt + 1;
                for (j = 1; j < n; j++) {
                    dplus = D[j] - s - E2[j - 1] / dplus;
                    if (dplus < 0.0) cnt = cnt + 1;
                }
                if (cnt < i + 1) {
                    right = right + werr[ii] * fac;
                    fac = TWO * fac;
                } else {
                    break;
                }
            }
            nint = nint + 1;
            iwork[k] = i + 1;
            iwork[k + 1] = cnt;
        }
        work[k] = left;
        work[k + 1] = right;
    }

    savi1 = i1;

    /*
     * Do while( nint > 0 ), i.e. there are still unconverged intervals
     * and while (iter < maxitr)
     */
    iter = 0;

    for (;;) {
        prev = i1 - 1;
        i = i1;
        olnint = nint;

        for (p = 0; p < olnint; p++) {
            k = 2 * i;
            ii = i - offset;
            next = iwork[k];
            left = work[k];
            right = work[k + 1];
            mid = HALF * (left + right);

            /* semiwidth of interval */
            width = right - mid;
            tmp = fmax(fabs(left), fabs(right));

            if ((width < rtol * tmp) || (iter == maxitr)) {
                /* reduce number of unconverged intervals */
                nint = nint - 1;
                /* Mark interval as converged. */
                iwork[k] = 0;
                if (i1 == i) {
                    i1 = next;
                } else {
                    /* Prev holds the last unconverged interval previously examined */
                    if (prev >= i1) {
                        iwork[2 * prev] = next;
                    }
                }
                i = next;
                continue;
            }
            prev = i;

            /* Perform one bisection step */
            cnt = 0;
            s = mid;
            dplus = D[0] - s;
            if (dplus < 0.0) cnt = cnt + 1;
            for (j = 1; j < n; j++) {
                dplus = D[j] - s - E2[j - 1] / dplus;
                if (dplus < 0.0) cnt = cnt + 1;
            }
            if (cnt <= i) {
                work[k] = mid;
            } else {
                work[k + 1] = mid;
            }
            i = next;
        }
        iter = iter + 1;
        /* do another loop if there are still unconverged intervals */
        /* However, in the last iteration, all intervals are accepted */
        /* since this is the best we can do. */
        if (!((nint > 0) && (iter <= maxitr))) {
            break;
        }
    }

    /* At this point, all the intervals have converged */
    for (i = savi1; i <= ilast; i++) {
        k = 2 * i;
        ii = i - offset;
        /* All intervals marked by '0' have been refined. */
        if (iwork[k] == 0) {
            W[ii] = HALF * (work[k] + work[k + 1]);
            werr[ii] = work[k + 1] - W[ii];
        }
    }
}
