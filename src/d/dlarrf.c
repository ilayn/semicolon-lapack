/** @file dlarrf.c
 * @brief DLARRF finds a new relatively robust representation such that at least one of the eigenvalues is relatively isolated.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DLARRF finds a new relatively robust representation L(+) D(+) L(+)^T
 * such that at least one of the eigenvalues of L(+) D(+) L(+)^T is
 * relatively isolated.
 *
 * Given the initial representation L D L^T and its cluster of close
 * eigenvalues (in a relative measure), W(clstrt), W(clstrt+1), ...
 * W(clend), DLARRF finds a new relatively robust representation
 * L D L^T - SIGMA I = L(+) D(+) L(+)^T such that at least one of the
 * eigenvalues of L(+) D(+) L(+)^T is relatively isolated.
 *
 * @param[in]     n       The order of the matrix (subblock, if the matrix split).
 * @param[in]     D       Double precision array, dimension (n).
 *                        The n diagonal elements of the diagonal matrix D.
 * @param[in]     L       Double precision array, dimension (n-1).
 *                        The (n-1) subdiagonal elements of the unit bidiagonal
 *                        matrix L.
 * @param[in]     LD      Double precision array, dimension (n-1).
 *                        The (n-1) elements L(i)*D(i).
 * @param[in]     clstrt  The index of the first eigenvalue in the cluster (0-based).
 * @param[in]     clend   The index of the last eigenvalue in the cluster (0-based).
 * @param[in]     W       Double precision array.
 *                        The eigenvalue approximations of L D L^T in ascending order.
 *                        W(clstrt) through W(clend) form the cluster.
 * @param[in,out] wgap    Double precision array.
 *                        The separation from the right neighbor eigenvalue in W.
 * @param[in]     werr    Double precision array.
 *                        The semiwidth of the uncertainty interval of the
 *                        corresponding eigenvalue approximation in W.
 * @param[in]     spdiam  Estimate of the spectral diameter obtained from the
 *                        Gerschgorin intervals.
 * @param[in]     clgapl  Absolute gap on the left end of the cluster.
 * @param[in]     clgapr  Absolute gap on the right end of the cluster.
 * @param[in]     pivmin  The minimum pivot allowed in the Sturm sequence.
 * @param[out]    sigma   The shift used to form L(+) D(+) L(+)^T.
 * @param[out]    dplus   Double precision array, dimension (n).
 *                        The n diagonal elements of the diagonal matrix D(+).
 * @param[out]    lplus   Double precision array, dimension (n-1).
 *                        The first (n-1) elements contain the subdiagonal
 *                        elements of the unit bidiagonal matrix L(+).
 * @param[out]    work    Double precision array, dimension (2*n).
 *                        Workspace.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - = 1: failure
 */
void dlarrf(const INT n, const f64* restrict D,
            const f64* restrict L, const f64* restrict LD,
            const INT clstrt, const INT clend,
            const f64* restrict W, f64* restrict wgap,
            const f64* restrict werr,
            const f64 spdiam, const f64 clgapl, const f64 clgapr,
            const f64 pivmin, f64* sigma,
            f64* restrict dplus, f64* restrict lplus,
            f64* restrict work, INT* info)
{
    /* Constants */
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 FOUR = 4.0;
    const f64 QUART = 0.25;
    const f64 MAXGROWTH1 = 8.0;
    const f64 MAXGROWTH2 = 8.0;
    const INT KTRYMAX = 1;
    const INT SLEFT = 1;
    const INT SRIGHT = 2;

    /* Local variables */
    INT i, indx, ktry, shift;
    INT forcer, nofail, sawnan1, sawnan2, dorrr1, tryrrr1;
    f64 avgap, bestshift, clwdth, eps, fact, fail, fail2;
    f64 growthbound, ldelta, ldmax, lsigma;
    f64 max1, max2, mingap, oldp, prod, rdelta, rdmax;
    f64 rrr1, rrr2, rsigma, s, smlgrowth, tmp, znm2;

    *info = 0;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    fact = (f64)(1 << KTRYMAX);  /* 2^KTRYMAX */
    eps = dlamch("P");
    forcer = 0;

    /* Note that we cannot guarantee that for any of the shifts tried,
     * the factorization has a small or even moderate element growth.
     * There could be Ritz values at both ends of the cluster and despite
     * backing off, there are examples where all factorizations tried
     * (in IEEE mode, allowing zero pivots & infinities) have INFINITE
     * element growth.
     * For this reason, we should use PIVMIN in this subroutine so that at
     * least the L D L^T factorization exists. It can be checked afterwards
     * whether the element growth caused bad residuals/orthogonality. */

    /* Decide whether the code should accept the best among all
     * representations despite large element growth or signal INFO=1
     * Setting NOFAIL to 0 for quick fix for bug 113 */
    nofail = 0;

    /* Compute the average gap length of the cluster */
    clwdth = fabs(W[clend] - W[clstrt]) + werr[clend] + werr[clstrt];
    avgap = clwdth / (f64)(clend - clstrt);
    mingap = fmin(clgapl, clgapr);

    /* Initial values for shifts to both ends of cluster */
    lsigma = fmin(W[clstrt], W[clend]) - werr[clstrt];
    rsigma = fmax(W[clstrt], W[clend]) + werr[clend];

    /* Use a small fudge to make sure that we really shift to the outside */
    lsigma = lsigma - fabs(lsigma) * FOUR * eps;
    rsigma = rsigma + fabs(rsigma) * FOUR * eps;

    /* Compute upper bounds for how much to back off the initial shifts */
    ldmax = QUART * mingap + TWO * pivmin;
    rdmax = QUART * mingap + TWO * pivmin;

    ldelta = fmax(avgap, wgap[clstrt]) / fact;
    rdelta = fmax(avgap, wgap[clend - 1]) / fact;

    /* Initialize the record of the best representation found */
    s = dlamch("S");
    smlgrowth = ONE / s;
    fail = (f64)(n - 1) * mingap / (spdiam * eps);
    fail2 = (f64)(n - 1) * mingap / (spdiam * sqrt(eps));
    bestshift = lsigma;

    /* Main retry loop: while (ktry <= KTRYMAX) */
    ktry = 0;
    growthbound = MAXGROWTH1 * spdiam;

    for (;;) {
        /* Label 5 in Fortran */
        sawnan1 = 0;
        sawnan2 = 0;

        /* Ensure that we do not back off too much of the initial shifts */
        ldelta = fmin(ldmax, ldelta);
        rdelta = fmin(rdmax, rdelta);

        /* Compute the element growth when shifting to both ends of the cluster
         * accept the shift if there is no element growth at one of the two ends */

        /* Left end */
        s = -lsigma;
        dplus[0] = D[0] + s;
        if (fabs(dplus[0]) < pivmin) {
            dplus[0] = -pivmin;
            /* Need to set sawnan1 because refined RRR test should not be used
             * in this case */
            sawnan1 = 1;
        }
        max1 = fabs(dplus[0]);
        for (i = 0; i < n - 1; i++) {
            lplus[i] = LD[i] / dplus[i];
            s = s * lplus[i] * L[i] - lsigma;
            dplus[i + 1] = D[i + 1] + s;
            if (fabs(dplus[i + 1]) < pivmin) {
                dplus[i + 1] = -pivmin;
                /* Need to set sawnan1 because refined RRR test should not be used
                 * in this case */
                sawnan1 = 1;
            }
            max1 = fmax(max1, fabs(dplus[i + 1]));
        }
        sawnan1 = sawnan1 || disnan(max1);

        if (forcer || (max1 <= growthbound && !sawnan1)) {
            *sigma = lsigma;
            shift = SLEFT;
            goto done;
        }

        /* Right end */
        s = -rsigma;
        work[0] = D[0] + s;
        if (fabs(work[0]) < pivmin) {
            work[0] = -pivmin;
            /* Need to set sawnan2 because refined RRR test should not be used
             * in this case */
            sawnan2 = 1;
        }
        max2 = fabs(work[0]);
        for (i = 0; i < n - 1; i++) {
            work[n + i] = LD[i] / work[i];
            s = s * work[n + i] * L[i] - rsigma;
            work[i + 1] = D[i + 1] + s;
            if (fabs(work[i + 1]) < pivmin) {
                work[i + 1] = -pivmin;
                /* Need to set sawnan2 because refined RRR test should not be used
                 * in this case */
                sawnan2 = 1;
            }
            max2 = fmax(max2, fabs(work[i + 1]));
        }
        sawnan2 = sawnan2 || disnan(max2);

        if (forcer || (max2 <= growthbound && !sawnan2)) {
            *sigma = rsigma;
            shift = SRIGHT;
            goto done;
        }

        /* If we are at this point, both shifts led to too much element growth */

        /* Record the better of the two shifts (provided it didn't lead to NaN) */
        if (sawnan1 && sawnan2) {
            /* both max1 and max2 are NaN */
            goto backoff;
        } else {
            if (!sawnan1) {
                indx = 1;
                if (max1 <= smlgrowth) {
                    smlgrowth = max1;
                    bestshift = lsigma;
                }
            }
            if (!sawnan2) {
                if (sawnan1 || max2 <= max1) {
                    indx = 2;
                }
                if (max2 <= smlgrowth) {
                    smlgrowth = max2;
                    bestshift = rsigma;
                }
            }
        }

        /* If we are here, both the left and the right shift led to
         * element growth. If the element growth is moderate, then
         * we may still accept the representation, if it passes a
         * refined test for RRR. This test supposes that no NaN occurred.
         * Moreover, we use the refined RRR test only for isolated clusters. */
        if ((clwdth < mingap / (f64)(128)) &&
            (fmin(max1, max2) < fail2) &&
            (!sawnan1) && (!sawnan2)) {
            dorrr1 = 1;
        } else {
            dorrr1 = 0;
        }
        tryrrr1 = 1;
        if (tryrrr1 && dorrr1) {
            if (indx == 1) {
                tmp = fabs(dplus[n - 1]);
                znm2 = ONE;
                prod = ONE;
                oldp = ONE;
                for (i = n - 2; i >= 0; i--) {
                    if (prod <= eps) {
                        prod = ((dplus[i + 1] * work[n + i + 1]) /
                                (dplus[i] * work[n + i])) * oldp;
                    } else {
                        prod = prod * fabs(work[n + i]);
                    }
                    oldp = prod;
                    znm2 = znm2 + prod * prod;
                    tmp = fmax(tmp, fabs(dplus[i] * prod));
                }
                rrr1 = tmp / (spdiam * sqrt(znm2));
                if (rrr1 <= MAXGROWTH2) {
                    *sigma = lsigma;
                    shift = SLEFT;
                    goto done;
                }
            } else if (indx == 2) {
                tmp = fabs(work[n - 1]);
                znm2 = ONE;
                prod = ONE;
                oldp = ONE;
                for (i = n - 2; i >= 0; i--) {
                    if (prod <= eps) {
                        prod = ((work[i + 1] * lplus[i + 1]) /
                                (work[i] * lplus[i])) * oldp;
                    } else {
                        prod = prod * fabs(lplus[i]);
                    }
                    oldp = prod;
                    znm2 = znm2 + prod * prod;
                    tmp = fmax(tmp, fabs(work[i] * prod));
                }
                rrr2 = tmp / (spdiam * sqrt(znm2));
                if (rrr2 <= MAXGROWTH2) {
                    *sigma = rsigma;
                    shift = SRIGHT;
                    goto done;
                }
            }
        }

backoff:
        if (ktry < KTRYMAX) {
            /* If we are here, both shifts failed also the RRR test.
             * Back off to the outside */
            lsigma = fmax(lsigma - ldelta, lsigma - ldmax);
            rsigma = fmin(rsigma + rdelta, rsigma + rdmax);
            ldelta = TWO * ldelta;
            rdelta = TWO * rdelta;
            ktry = ktry + 1;
            /* continue the for(;;) loop (equivalent to GOTO 5) */
        } else {
            /* None of the representations investigated satisfied our
             * criteria. Take the best one we found. */
            if ((smlgrowth < fail) || nofail) {
                lsigma = bestshift;
                rsigma = bestshift;
                forcer = 1;
                /* continue the for(;;) loop (equivalent to GOTO 5) */
            } else {
                *info = 1;
                return;
            }
        }
    } /* end for (;;) */

done:
    if (shift == SLEFT) {
        /* dplus and lplus already contain the factorization */
    } else if (shift == SRIGHT) {
        /* store new L and D back into DPLUS, LPLUS */
        cblas_dcopy(n, work, 1, dplus, 1);
        cblas_dcopy(n - 1, &work[n], 1, lplus, 1);
    }

    return;
}
