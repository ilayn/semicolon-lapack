/**
 * @file slarrb.c
 * @brief SLARRB provides limited bisection to locate eigenvalues for more accuracy.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * Given the relatively robust representation(RRR) L D L^T, SLARRB
 * does "limited" bisection to refine the eigenvalues of L D L^T,
 * W( IFIRST-OFFSET ) through W( ILAST-OFFSET ), to more accuracy. Initial
 * guesses for these eigenvalues are input in W, the corresponding estimate
 * of the error in these guesses and their gaps are input in WERR
 * and WGAP, respectively. During bisection, intervals
 * [left, right] are maintained by storing their mid-points and
 * semi-widths in the arrays W and WERR respectively.
 *
 * @param[in]     n       The order of the matrix.
 * @param[in]     D       Double precision array, dimension (N).
 *                        The N diagonal elements of the diagonal matrix D.
 * @param[in]     lld     Double precision array, dimension (N-1).
 *                        The (N-1) elements L(i)*L(i)*D(i).
 * @param[in]     ifirst  The index of the first eigenvalue to be computed (0-based).
 * @param[in]     ilast   The index of the last eigenvalue to be computed (0-based).
 * @param[in]     rtol1   Tolerance for the convergence of the bisection intervals.
 * @param[in]     rtol2   Tolerance for the convergence of the bisection intervals.
 *                        An interval [LEFT,RIGHT] has converged if
 *                        RIGHT-LEFT < MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) )
 *                        where GAP is the (estimated) distance to the nearest
 *                        eigenvalue.
 * @param[in]     offset  Offset for the arrays W, WGAP and WERR, i.e., the
 *                        IFIRST-OFFSET through ILAST-OFFSET elements of these
 *                        arrays are to be used.
 * @param[in,out] W       Double precision array, dimension (N).
 *                        On input, W( IFIRST-OFFSET ) through W( ILAST-OFFSET )
 *                        are estimates of the eigenvalues of L D L^T indexed
 *                        IFIRST through ILAST.
 *                        On output, these estimates are refined.
 * @param[in,out] wgap    Double precision array, dimension (N-1).
 *                        On input, the (estimated) gaps between consecutive
 *                        eigenvalues of L D L^T. Note that if IFIRST = ILAST
 *                        then WGAP(IFIRST-OFFSET) must be set to ZERO.
 *                        On output, these gaps are refined.
 * @param[in,out] werr    Double precision array, dimension (N).
 *                        On input, WERR( IFIRST-OFFSET ) through WERR( ILAST-OFFSET )
 *                        are the errors in the estimates of the corresponding
 *                        elements in W.
 *                        On output, these errors are refined.
 * @param[out]    work    Double precision array, dimension (2*N). Workspace.
 * @param[out]    iwork   Integer array, dimension (2*N). Workspace.
 * @param[in]     pivmin  The minimum pivot in the Sturm sequence.
 * @param[in]     spdiam  The spectral diameter of the matrix.
 * @param[in]     twist   The twist index for the twisted factorization that is used
 *                        for the negcount.
 *                        TWIST = N-1: Compute negcount from L D L^T - LAMBDA I = L+ D+ L+^T
 *                        TWIST = 0:   Compute negcount from L D L^T - LAMBDA I = U- D- U-^T
 *                        TWIST = R:   Compute negcount from L D L^T - LAMBDA I = N(r) D(r) N(r)
 * @param[out]    info
 *                           Error flag.
 */
void slarrb(const int n, const f32* D, const f32* lld,
            const int ifirst, const int ilast,
            const f32 rtol1, const f32 rtol2, const int offset,
            f32* W, f32* wgap, f32* werr,
            f32* work, int* iwork,
            const f32 pivmin, const f32 spdiam,
            const int twist, int* info)
{
    const f32 ZERO = 0.0f;
    const f32 TWO = 2.0f;
    const f32 HALF = 0.5f;

    int i, i1, ii, ip, iter, k, negcnt, next, nint, olnint, prev, r;
    int maxitr;
    f32 back, cvrgd, gap, left, lgap, mid, mnwdth, rgap, right, tmp, width;

    *info = 0;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    maxitr = (int)((logf(spdiam + pivmin) - logf(pivmin)) / logf(TWO)) + 2;
    mnwdth = TWO * pivmin;

    /* twist is 0-based; valid range is [0, n-1] */
    r = twist;
    if ((r < 0) || (r > n - 1)) r = n - 1;

    /*
     * Initialize unconverged intervals in [ WORK(2*I), WORK(2*I+1) ].
     * The Sturm Count, Count( WORK(2*I) ) is arranged to be I, while
     * Count( WORK(2*I+1) ) is stored in IWORK( 2*I+1 ). The integer IWORK( 2*I )
     * for an unconverged interval is set to the index of the next unconverged
     * interval, and is -1 or 0 for a converged interval. Thus a linked
     * list of unconverged intervals is set up.
     */
    i1 = ifirst;
    /* The number of unconverged intervals */
    nint = 0;
    /* The last unconverged interval found */
    prev = -1;

    rgap = wgap[i1 - offset];
    for (i = i1; i <= ilast; i++) {
        k = 2 * i;
        ii = i - offset;
        left = W[ii] - werr[ii];
        right = W[ii] + werr[ii];
        lgap = rgap;
        rgap = wgap[ii];
        gap = (lgap < rgap) ? lgap : rgap;

        /*
         * Make sure that [LEFT,RIGHT] contains the desired eigenvalue
         * Compute negcount from dstqds facto L+D+L+^T = L D L^T - LEFT
         *
         * Do while( NEGCNT(LEFT).GT.I )
         */
        back = werr[ii];
        for (;;) {
            negcnt = slaneg(n, D, lld, left, pivmin, r);
            if (negcnt > i) {
                left = left - back;
                back = TWO * back;
            } else {
                break;
            }
        }

        /*
         * Do while( NEGCNT(RIGHT).LT.I+1 )
         * Compute negcount from dstqds facto L+D+L+^T = L D L^T - RIGHT
         */
        back = werr[ii];
        for (;;) {
            negcnt = slaneg(n, D, lld, right, pivmin, r);
            if (negcnt < i + 1) {
                right = right + back;
                back = TWO * back;
            } else {
                break;
            }
        }
        width = HALF * fabsf(left - right);
        tmp = fabsf(left);
        if (fabsf(right) > tmp) tmp = fabsf(right);
        cvrgd = rtol1 * gap;
        if (rtol2 * tmp > cvrgd) cvrgd = rtol2 * tmp;
        if (width <= cvrgd || width <= mnwdth) {
            /*
             * This interval has already converged and does not need refinement.
             * (Note that the gaps might change through refining the
             *  eigenvalues, however, they can only get bigger.)
             * Remove it from the list.
             */
            iwork[k] = -1;
            /* Make sure that I1 always points to the first unconverged interval */
            if ((i == i1) && (i < ilast)) i1 = i + 1;
            if ((prev >= i1) && (i <= ilast)) iwork[2 * prev] = i + 1;
        } else {
            /* unconverged interval found */
            prev = i;
            nint = nint + 1;
            iwork[k] = i + 1;
            iwork[k + 1] = negcnt;
        }
        work[k] = left;
        work[k + 1] = right;
    }

    /*
     * Do while( NINT.GT.0 ), i.e. there are still unconverged intervals
     * and while (ITER.LT.MAXITR)
     */
    iter = 0;
    do {
        prev = i1 - 1;
        i = i1;
        olnint = nint;

        for (ip = 0; ip < olnint; ip++) {
            k = 2 * i;
            ii = i - offset;
            rgap = wgap[ii];
            lgap = rgap;
            if (ii > 0) lgap = wgap[ii - 1];
            gap = (lgap < rgap) ? lgap : rgap;
            next = iwork[k];
            left = work[k];
            right = work[k + 1];
            mid = HALF * (left + right);

            /* semiwidth of interval */
            width = right - mid;
            tmp = fabsf(left);
            if (fabsf(right) > tmp) tmp = fabsf(right);
            cvrgd = rtol1 * gap;
            if (rtol2 * tmp > cvrgd) cvrgd = rtol2 * tmp;
            if ((width <= cvrgd) || (width <= mnwdth) ||
                (iter == maxitr)) {
                /* reduce number of unconverged intervals */
                nint = nint - 1;
                /* Mark interval as converged. */
                iwork[k] = 0;
                if (i1 == i) {
                    i1 = next;
                } else {
                    /* Prev holds the last unconverged interval previously examined */
                    if (prev >= i1) iwork[2 * prev] = next;
                }
                i = next;
                continue;
            }
            prev = i;

            /* Perform one bisection step */
            negcnt = slaneg(n, D, lld, mid, pivmin, r);
            if (negcnt <= i) {
                work[k] = mid;
            } else {
                work[k + 1] = mid;
            }
            i = next;
        }
        iter = iter + 1;
        /* do another loop if there are still unconverged intervals
         * However, in the last iteration, all intervals are accepted
         * since this is the best we can do. */
    } while ((nint > 0) && (iter <= maxitr));

    /* At this point, all the intervals have converged */
    for (i = ifirst; i <= ilast; i++) {
        k = 2 * i;
        ii = i - offset;
        /* All intervals marked by '0' have been refined. */
        if (iwork[k] == 0) {
            W[ii] = HALF * (work[k] + work[k + 1]);
            werr[ii] = work[k + 1] - W[ii];
        }
    }

    for (i = ifirst + 1; i <= ilast; i++) {
        ii = i - offset;
        tmp = W[ii] - werr[ii] - W[ii - 1] - werr[ii - 1];
        wgap[ii - 1] = (ZERO > tmp) ? ZERO : tmp;
    }
}
