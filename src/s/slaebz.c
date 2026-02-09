/**
 * @file slaebz.c
 * @brief SLAEBZ computes the number of eigenvalues of a real symmetric
 *        tridiagonal matrix which are less than or equal to a given value,
 *        and performs other tasks required by the routine SSTEBZ.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SLAEBZ contains the iteration loops which compute and use the
 * function N(w), which is the count of eigenvalues of a symmetric
 * tridiagonal matrix T less than or equal to its argument w. It
 * performs a choice of two types of loops:
 *
 * IJOB=1, followed by
 * IJOB=2: It takes as input a list of intervals and returns a list of
 *         sufficiently small intervals whose union contains the same
 *         eigenvalues as the union of the original intervals.
 *         The input intervals are (AB(j,1),AB(j,2)], j=0,...,MINP-1.
 *         The output interval (AB(j,1),AB(j,2)] will contain
 *         eigenvalues NAB(j,1)+1,...,NAB(j,2), where 0 <= j < MOUT.
 *
 * IJOB=3: It performs a binary search in each input interval
 *         (AB(j,1),AB(j,2)] for a point w(j) such that
 *         N(w(j))=NVAL(j), and uses C(j) as the starting point of
 *         the search. If such a w(j) is found, then on output
 *         AB(j,1)=AB(j,2)=w. If no such w(j) is found, then on output
 *         (AB(j,1),AB(j,2)] will be a small interval containing the
 *         point where N(w) jumps through NVAL(j), unless that point
 *         lies outside the initial interval.
 *
 * Note that the intervals are in all cases half-open intervals,
 * i.e., of the form (a,b], which includes b but not a.
 *
 * @param[in]     ijob    Specifies what is to be done:
 *                        = 1: Compute NAB for the initial intervals.
 *                        = 2: Perform bisection iteration to find eigenvalues.
 *                        = 3: Perform bisection iteration to invert N(w).
 *                        Other values will cause SLAEBZ to return with info=-1.
 * @param[in]     nitmax  The maximum number of "levels" of bisection to be
 *                        performed.
 * @param[in]     n       The dimension of the tridiagonal matrix T. n >= 1.
 * @param[in]     mmax    The maximum number of intervals.
 * @param[in]     minp    The initial number of intervals.
 * @param[in]     nbmin   The smallest number of intervals that should be
 *                        processed using a vector loop. If zero, then only the
 *                        scalar loop will be used.
 * @param[in]     abstol  The minimum (absolute) width of an interval.
 * @param[in]     reltol  The minimum relative width of an interval.
 * @param[in]     pivmin  The minimum absolute value of a "pivot" in the Sturm
 *                        sequence loop.
 * @param[in]     D       Double precision array, dimension (n).
 *                        The diagonal elements of the tridiagonal matrix T.
 * @param[in]     E       Double precision array, dimension (n).
 *                        The offdiagonal elements of T in positions 0..n-2.
 * @param[in]     E2      Double precision array, dimension (n).
 *                        The squares of the offdiagonal elements of T.
 * @param[in,out] nval    Integer array, dimension (minp).
 *                        If ijob=1 or 2, not referenced.
 *                        If ijob=3, the desired values of N(w).
 * @param[in,out] AB      Double precision array, dimension (mmax, 2),
 *                        column-major. The endpoints of the intervals.
 * @param[in,out] C       Double precision array, dimension (mmax).
 *                        If ijob=1, ignored. If ijob=2, workspace.
 *                        If ijob=3, the starting search points.
 * @param[out]    mout    If ijob=1, the number of eigenvalues in the intervals.
 *                        If ijob=2 or 3, the number of intervals output.
 * @param[in,out] NAB     Integer array, dimension (mmax, 2), column-major.
 *                        The eigenvalue counts at the interval endpoints.
 * @param[out]    work    Double precision workspace, dimension (mmax).
 * @param[out]    iwork   Integer workspace, dimension (mmax).
 * @param[out]    info    = 0: All intervals converged.
 *                        = 1..MMAX: The last info intervals did not converge.
 *                        = MMAX+1: More than MMAX intervals were generated.
 */
void slaebz(const int ijob, const int nitmax, const int n,
            const int mmax, const int minp, const int nbmin,
            const float abstol, const float reltol, const float pivmin,
            const float* const restrict D,
            const float* const restrict E,
            const float* const restrict E2,
            int* const restrict nval,
            float* const restrict AB,
            float* const restrict C,
            int* mout,
            int* const restrict NAB,
            float* const restrict work,
            int* const restrict iwork,
            int* info)
{
    (void)E;  /* E is in the API for compatibility but E2 is used instead */
    const float HALF = 0.5f;

    int itmp1, itmp2, j, ji, jit, kf, kfnew, kl, klnew;
    float tmp1, tmp2;

    /* Check for errors */
    *info = 0;
    if (ijob < 1 || ijob > 3) {
        *info = -1;
        return;
    }

    /* Initialize NAB */
    if (ijob == 1) {
        /*
         * Compute the number of eigenvalues in the initial intervals.
         */
        *mout = 0;
        for (ji = 0; ji < minp; ji++) {
            for (int jp = 0; jp < 2; jp++) {
                /* Sturm sequence: tmp1 = D(1) - AB(ji, jp) */
                tmp1 = D[0] - AB[ji + jp * mmax];
                if (fabsf(tmp1) < pivmin)
                    tmp1 = -pivmin;
                NAB[ji + jp * mmax] = 0;
                if (tmp1 <= 0.0f)
                    NAB[ji + jp * mmax] = 1;

                for (j = 1; j < n; j++) {
                    /* tmp1 = D(j+1) - E2(j) / tmp1 - AB(ji, jp) */
                    tmp1 = D[j] - E2[j - 1] / tmp1 - AB[ji + jp * mmax];
                    if (fabsf(tmp1) < pivmin)
                        tmp1 = -pivmin;
                    if (tmp1 <= 0.0f)
                        NAB[ji + jp * mmax] = NAB[ji + jp * mmax] + 1;
                }
            }
            *mout = *mout + NAB[ji + 1 * mmax] - NAB[ji + 0 * mmax];
        }
        return;
    }

    /*
     * Initialize for loop
     *
     * KF and KL have the following meaning:
     *   Intervals 0,...,kf-1 have converged.
     *   Intervals kf,...,kl still need to be refined.
     */
    kf = 0;
    kl = minp - 1;

    /* If ijob=2, initialize C.
     * If ijob=3, use the user-supplied starting point. */
    if (ijob == 2) {
        for (ji = 0; ji < minp; ji++) {
            C[ji] = HALF * (AB[ji + 0 * mmax] + AB[ji + 1 * mmax]);
        }
    }

    /* Iteration loop */
    for (jit = 0; jit < nitmax; jit++) {

        /* Loop over intervals */
        if (kl - kf + 1 >= nbmin && nbmin > 0) {

            /*
             * Begin of Parallel Version of the loop
             */
            for (ji = kf; ji <= kl; ji++) {

                /* Compute N(c), the number of eigenvalues less than c */
                work[ji] = D[0] - C[ji];
                iwork[ji] = 0;
                if (work[ji] <= pivmin) {
                    iwork[ji] = 1;
                    work[ji] = work[ji] < -pivmin ? work[ji] : -pivmin;
                }

                for (j = 1; j < n; j++) {
                    work[ji] = D[j] - E2[j - 1] / work[ji] - C[ji];
                    if (work[ji] <= pivmin) {
                        iwork[ji] = iwork[ji] + 1;
                        work[ji] = work[ji] < -pivmin ? work[ji] : -pivmin;
                    }
                }
            }

            if (ijob <= 2) {

                /*
                 * IJOB=2: Choose all intervals containing eigenvalues.
                 */
                klnew = kl;
                for (ji = kf; ji <= kl; ji++) {

                    /* Ensure that N(w) is monotone:
                     * iwork[ji] = min(NAB(ji,2), max(NAB(ji,1), iwork[ji])) */
                    if (iwork[ji] < NAB[ji + 0 * mmax])
                        iwork[ji] = NAB[ji + 0 * mmax];
                    if (iwork[ji] > NAB[ji + 1 * mmax])
                        iwork[ji] = NAB[ji + 1 * mmax];

                    /*
                     * Update the Queue -- add intervals if both halves
                     * contain eigenvalues.
                     */
                    if (iwork[ji] == NAB[ji + 1 * mmax]) {
                        /*
                         * No eigenvalue in the upper interval:
                         * just use the lower interval.
                         */
                        AB[ji + 1 * mmax] = C[ji];

                    } else if (iwork[ji] == NAB[ji + 0 * mmax]) {
                        /*
                         * No eigenvalue in the lower interval:
                         * just use the upper interval.
                         */
                        AB[ji + 0 * mmax] = C[ji];
                    } else {
                        klnew = klnew + 1;
                        if (klnew <= mmax - 1) {
                            /*
                             * Eigenvalue in both intervals -- add upper to
                             * queue.
                             */
                            AB[klnew + 1 * mmax] = AB[ji + 1 * mmax];
                            NAB[klnew + 1 * mmax] = NAB[ji + 1 * mmax];
                            AB[klnew + 0 * mmax] = C[ji];
                            NAB[klnew + 0 * mmax] = iwork[ji];
                            AB[ji + 1 * mmax] = C[ji];
                            NAB[ji + 1 * mmax] = iwork[ji];
                        } else {
                            *info = mmax + 1;
                        }
                    }
                }
                if (*info != 0)
                    return;
                kl = klnew;
            } else {

                /*
                 * IJOB=3: Binary search. Keep only the interval containing
                 *         w s.t. N(w) = NVAL
                 */
                for (ji = kf; ji <= kl; ji++) {
                    if (iwork[ji] <= nval[ji]) {
                        AB[ji + 0 * mmax] = C[ji];
                        NAB[ji + 0 * mmax] = iwork[ji];
                    }
                    if (iwork[ji] >= nval[ji]) {
                        AB[ji + 1 * mmax] = C[ji];
                        NAB[ji + 1 * mmax] = iwork[ji];
                    }
                }
            }

        } else {

            /*
             * End of Parallel Version of the loop
             *
             * Begin of Serial Version of the loop
             */
            klnew = kl;
            for (ji = kf; ji <= kl; ji++) {

                /* Compute N(w), the number of eigenvalues less than w */
                tmp1 = C[ji];
                tmp2 = D[0] - tmp1;
                itmp1 = 0;
                if (tmp2 <= pivmin) {
                    itmp1 = 1;
                    tmp2 = tmp2 < -pivmin ? tmp2 : -pivmin;
                }

                for (j = 1; j < n; j++) {
                    tmp2 = D[j] - E2[j - 1] / tmp2 - tmp1;
                    if (tmp2 <= pivmin) {
                        itmp1 = itmp1 + 1;
                        tmp2 = tmp2 < -pivmin ? tmp2 : -pivmin;
                    }
                }

                if (ijob <= 2) {

                    /*
                     * IJOB=2: Choose all intervals containing eigenvalues.
                     *
                     * Ensure that N(w) is monotone
                     */
                    {
                        int lo = NAB[ji + 0 * mmax] > itmp1 ? NAB[ji + 0 * mmax] : itmp1;
                        itmp1 = NAB[ji + 1 * mmax] < lo ? NAB[ji + 1 * mmax] : lo;
                    }

                    /*
                     * Update the Queue -- add intervals if both halves
                     * contain eigenvalues.
                     */
                    if (itmp1 == NAB[ji + 1 * mmax]) {
                        /*
                         * No eigenvalue in the upper interval:
                         * just use the lower interval.
                         */
                        AB[ji + 1 * mmax] = tmp1;

                    } else if (itmp1 == NAB[ji + 0 * mmax]) {
                        /*
                         * No eigenvalue in the lower interval:
                         * just use the upper interval.
                         */
                        AB[ji + 0 * mmax] = tmp1;

                    } else if (klnew < mmax - 1) {
                        /*
                         * Eigenvalue in both intervals -- add upper to queue.
                         */
                        klnew = klnew + 1;
                        AB[klnew + 1 * mmax] = AB[ji + 1 * mmax];
                        NAB[klnew + 1 * mmax] = NAB[ji + 1 * mmax];
                        AB[klnew + 0 * mmax] = tmp1;
                        NAB[klnew + 0 * mmax] = itmp1;
                        AB[ji + 1 * mmax] = tmp1;
                        NAB[ji + 1 * mmax] = itmp1;
                    } else {
                        *info = mmax + 1;
                        return;
                    }
                } else {

                    /*
                     * IJOB=3: Binary search. Keep only the interval
                     *         containing w s.t. N(w) = NVAL
                     */
                    if (itmp1 <= nval[ji]) {
                        AB[ji + 0 * mmax] = tmp1;
                        NAB[ji + 0 * mmax] = itmp1;
                    }
                    if (itmp1 >= nval[ji]) {
                        AB[ji + 1 * mmax] = tmp1;
                        NAB[ji + 1 * mmax] = itmp1;
                    }
                }
            }
            kl = klnew;

        }

        /*
         * Check for convergence
         */
        kfnew = kf;
        for (ji = kf; ji <= kl; ji++) {
            tmp1 = fabsf(AB[ji + 1 * mmax] - AB[ji + 0 * mmax]);
            tmp2 = fabsf(AB[ji + 1 * mmax]);
            {
                float atmp = fabsf(AB[ji + 0 * mmax]);
                if (atmp > tmp2)
                    tmp2 = atmp;
            }
            {
                float thresh = abstol > pivmin ? abstol : pivmin;
                float rthresh = reltol * tmp2;
                if (rthresh > thresh)
                    thresh = rthresh;

                if (tmp1 < thresh || NAB[ji + 0 * mmax] >= NAB[ji + 1 * mmax]) {
                    /*
                     * Converged -- Swap with position kfnew,
                     *              then increment kfnew
                     */
                    if (ji > kfnew) {
                        tmp1 = AB[ji + 0 * mmax];
                        tmp2 = AB[ji + 1 * mmax];
                        itmp1 = NAB[ji + 0 * mmax];
                        itmp2 = NAB[ji + 1 * mmax];
                        AB[ji + 0 * mmax] = AB[kfnew + 0 * mmax];
                        AB[ji + 1 * mmax] = AB[kfnew + 1 * mmax];
                        NAB[ji + 0 * mmax] = NAB[kfnew + 0 * mmax];
                        NAB[ji + 1 * mmax] = NAB[kfnew + 1 * mmax];
                        AB[kfnew + 0 * mmax] = tmp1;
                        AB[kfnew + 1 * mmax] = tmp2;
                        NAB[kfnew + 0 * mmax] = itmp1;
                        NAB[kfnew + 1 * mmax] = itmp2;
                        if (ijob == 3) {
                            itmp1 = nval[ji];
                            nval[ji] = nval[kfnew];
                            nval[kfnew] = itmp1;
                        }
                    }
                    kfnew = kfnew + 1;
                }
            }
        }
        kf = kfnew;

        /* Choose Midpoints */
        for (ji = kf; ji <= kl; ji++) {
            C[ji] = HALF * (AB[ji + 0 * mmax] + AB[ji + 1 * mmax]);
        }

        /* If no more intervals to refine, quit. */
        if (kf > kl)
            break;
    }

    /* Converged */
    *info = (kl + 1 - kf) > 0 ? (kl + 1 - kf) : 0;
    *mout = kl + 1;
}
