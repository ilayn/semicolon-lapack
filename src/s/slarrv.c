/**
 * @file slarrv.c
 * @brief SLARRV computes the eigenvectors of the tridiagonal matrix
 *        T = L D L^T given L, D and the eigenvalues of L D L^T.
 */

#include <math.h>
#include <string.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"


/**
 * SLARRV computes the eigenvectors of the tridiagonal matrix
 * T = L D L^T given L, D and APPROXIMATIONS to the eigenvalues of L D L^T.
 * The input eigenvalues should have been computed by SLARRE.
 *
 * @param[in]     n       The order of the matrix. n >= 0.
 * @param[in]     vl      Lower bound of the interval that contains the desired
 *                        eigenvalues. vl < vu.
 * @param[in]     vu      Upper bound of the interval that contains the desired
 *                        eigenvalues. vl < vu. Currently not used by this
 *                        implementation.
 * @param[in,out] D       Double precision array, dimension (n).
 *                        On entry, the n diagonal elements of the diagonal matrix D.
 *                        On exit, D may be overwritten.
 * @param[in,out] L       Double precision array, dimension (n).
 *                        On entry, the (n-1) subdiagonal elements of the unit
 *                        bidiagonal matrix L are in elements 0 to n-2.
 *                        At the end of each block is stored the corresponding shift
 *                        as given by SLARRE.
 *                        On exit, L is overwritten.
 * @param[in]     pivmin  The minimum pivot allowed in the Sturm sequence.
 * @param[in]     isplit  Integer array, dimension (n).
 *                        The splitting points (0-based last indices of blocks).
 * @param[in]     m       The total number of input eigenvalues. 0 <= m <= n.
 * @param[in]     dol     If the user wants to compute only selected eigenvectors,
 *                        dol is the 0-based index of the first desired eigenvector.
 *                        Otherwise dol = 0.
 * @param[in]     dou     0-based index of the last desired eigenvector.
 *                        Otherwise dou = m-1.
 * @param[in]     minrgp  Minimum relative gap parameter.
 * @param[in]     rtol1   Tolerance for bisection convergence.
 * @param[in]     rtol2   Tolerance for bisection convergence.
 *                        An interval [LEFT,RIGHT] has converged if
 *                        RIGHT-LEFT < MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) )
 * @param[in,out] W       Double precision array, dimension (n).
 *                        The first m elements contain the approximate eigenvalues
 *                        for which eigenvectors are to be computed. On exit, W holds
 *                        the eigenvalues of the UNshifted matrix.
 * @param[in,out] werr    Double precision array, dimension (n).
 *                        The first m elements contain the semiwidth of the
 *                        uncertainty interval of the corresponding eigenvalue in W.
 * @param[in,out] wgap    Double precision array, dimension (n).
 *                        The separation from the right neighbor eigenvalue in W.
 * @param[in]     iblock  Integer array, dimension (n).
 *                        The 0-based block indices for each eigenvalue.
 * @param[in]     indexw  Integer array, dimension (n).
 *                        The 0-based local indices of eigenvalues within each block.
 * @param[in]     gers    Double precision array, dimension (2*n).
 *                        The n Gerschgorin intervals (gers[2*i], gers[2*i+1]).
 * @param[out]    Z       Double precision array, dimension (ldz, max(1,m)).
 *                        The orthonormal eigenvectors of the matrix T.
 * @param[in]     ldz     The leading dimension of Z. ldz >= max(1,n).
 * @param[out]    isuppz  Integer array, dimension (2*max(1,m)).
 *                        The support of eigenvectors in Z (0-based indices).
 *                        isuppz[2*i] through isuppz[2*i+1].
 * @param[out]    work    Double precision array, dimension (12*n).
 * @param[out]    iwork   Integer array, dimension (7*n).
 * @param[out]    info    = 0: successful exit
 *                        > 0: a problem occurred in SLARRV.
 *                        < 0: one of the called subroutines signaled an internal
 *                             problem.
 *                        = -1: problem in SLARRB when refining a child's eigenvalues.
 *                        = -2: problem in SLARRF when computing the RRR of a child.
 *                        = -3: problem in SLARRB when refining a single eigenvalue
 *                              after the Rayleigh correction was rejected.
 *                        = 5: the Rayleigh Quotient Iteration failed to converge to
 *                             full accuracy in MAXITR steps.
 */
void slarrv(const int n, const float vl, const float vu,
            float* D, float* L, const float pivmin,
            const int* isplit, const int m, const int dol, const int dou,
            const float minrgp, const float rtol1, const float rtol2,
            float* W, float* werr, float* wgap,
            const int* iblock, const int* indexw, const float* gers,
            float* Z, const int ldz, int* isuppz,
            float* work, int* iwork, int* info)
{
    /* Parameters */
    const int MAXITR = 10;
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float TWO = 2.0f;
    const float THREE = 3.0f;
    const float FOUR = 4.0f;
    const float HALF = 0.5f;

    /* Local Scalars */
    int eskip, needbs, stp2ii, tryrqc, usedbs, usedrq;
    int done, i, ibegin, idone, iend, ii, iindc1,
        iindc2, iindr, iindwk, iinfo, im, in, indeig,
        indld, indlld, indwrk, isupmn, isupmx, iter,
        itmp1, j, jblk, k, miniwsize, minwsize, nclus,
        ndepth, negcnt, newcls, newfst, newftt, newlst,
        newsiz, offset, oldcls, oldfst, oldien, oldlst,
        oldncl, p, parity, q, wbegin, wend, windex,
        windmn, windpl, zfrom, zto, zusedl, zusedu,
        zusedw;
    float bstres, bstw, eps, fudge, gap, gaptol, gl, gu,
           lambda, left, lgap, mingma, nrminv, resid,
           rgap, right, rqcorr, rqtol, savgap, sgndef,
           sigma, spdiam, ssigma, tau, tmp, tol, ztz;

    /* Effective rtol values (may be overridden for partial eigenvector computation) */
    float eff_rtol1 = rtol1;
    float eff_rtol2 = rtol2;

    (void)vu; /* Currently unused */

    *info = 0;

    /* Quick return if possible */
    if ((n <= 0) || (m <= 0)) {
        return;
    }

    /* The first n entries of WORK are reserved for the eigenvalues */
    indld = n;
    indlld = 2 * n;
    indwrk = 3 * n;
    minwsize = 12 * n;

    memset(work, 0, minwsize * sizeof(float));

    /* IWORK(IINDR..IINDR+N-1) hold the twist indices R for the
     * factorization used to compute the FP vector */
    iindr = 0;
    /* IWORK(IINDC1..IINDC1+N-1) are used to store the clusters of the current
     * layer and the one above. */
    iindc1 = n;
    iindc2 = 2 * n;
    iindwk = 3 * n;

    miniwsize = 7 * n;
    memset(iwork, 0, miniwsize * sizeof(int));

    for (i = 0; i < n; i++) {
        iwork[iindr + i] = -1;
    }

    zusedl = 0;
    if (dol > 0) {
        /* Set lower bound for use of Z */
        zusedl = dol - 1;
    }
    zusedu = m - 1;
    if (dou < m - 1) {
        /* Set upper bound for use of Z */
        zusedu = dou + 1;
    }
    /* The width of the part of Z that is used */
    zusedw = zusedu - zusedl + 1;

    slaset("Full", n, zusedw, ZERO, ZERO, &Z[zusedl * ldz], ldz);

    eps = slamch("Precision");
    rqtol = TWO * eps;

    /* Set expert flags for standard code. */
    tryrqc = 1;

    if ((dol == 0) && (dou == m - 1)) {
        /* All eigenpairs computed */
    } else {
        /* Only selected eigenpairs are computed. Since the other evalues
         * are not refined by RQ iteration, bisection has to compute to full
         * accuracy. */
        eff_rtol1 = FOUR * eps;
        eff_rtol2 = FOUR * eps;
    }

    /* The entries WBEGIN:WEND in W, WERR, WGAP correspond to the
     * desired eigenvalues. The support of the nonzero eigenvector
     * entries is contained in the interval IBEGIN:IEND.
     * Remark that if k eigenpairs are desired, then the eigenvectors
     * are stored in k contiguous columns of Z. */

    /* DONE is the number of eigenvectors already computed */
    done = 0;
    ibegin = 0;
    wbegin = 0;

    for (jblk = 0; jblk <= iblock[m - 1]; jblk++) {
        iend = isplit[jblk];
        sigma = L[iend];

        /* Find the eigenvectors of the submatrix indexed IBEGIN through IEND. */
        wend = wbegin - 1;
        while (wend < m - 1) {
            if (iblock[wend + 1] == jblk) {
                wend = wend + 1;
            } else {
                break;
            }
        }
        if (wend < wbegin) {
            ibegin = iend + 1;
            continue;
        } else if ((wend < dol) || (wbegin > dou)) {
            ibegin = iend + 1;
            wbegin = wend + 1;
            continue;
        }

        /* Find local spectral diameter of the block */
        gl = gers[2 * ibegin];
        gu = gers[2 * ibegin + 1];
        for (i = ibegin + 1; i <= iend; i++) {
            if (gers[2 * i] < gl) gl = gers[2 * i];
            if (gers[2 * i + 1] > gu) gu = gers[2 * i + 1];
        }
        spdiam = gu - gl;

        /* OLDIEN is the offset to convert local (within-block) 0-based indices
         * to global 0-based indices. In Fortran (1-based), this is IBEGIN-1.
         * In 0-based C, this is simply IBEGIN since local index 0 maps to
         * global index IBEGIN. */
        oldien = ibegin;
        /* Calculate the size of the current block */
        in = iend - ibegin + 1;
        /* The number of eigenvalues in the current block */
        im = wend - wbegin + 1;

        /* This is for a 1x1 block */
        if (ibegin == iend) {
            done = done + 1;
            Z[ibegin + wbegin * ldz] = ONE;
            isuppz[2 * wbegin] = ibegin;
            isuppz[2 * wbegin + 1] = ibegin;
            W[wbegin] = W[wbegin] + sigma;
            work[wbegin] = W[wbegin];
            ibegin = iend + 1;
            wbegin = wbegin + 1;
            continue;
        }

        /* The desired (shifted) eigenvalues are stored in W(WBEGIN:WEND)
         * Note that these can be approximations, in this case, the corresp.
         * entries of WERR give the size of the uncertainty interval.
         * The eigenvalue approximations will be refined when necessary as
         * high relative accuracy is required for the computation of the
         * corresponding eigenvectors. */
        cblas_scopy(im, &W[wbegin], 1, &work[wbegin], 1);

        /* We store in W the eigenvalue approximations w.r.t. the original
         * matrix T. */
        for (i = 0; i < im; i++) {
            W[wbegin + i] = W[wbegin + i] + sigma;
        }

        /* NDEPTH is the current depth of the representation tree */
        ndepth = 0;
        /* PARITY is either 1 or 0 */
        parity = 1;
        /* NCLUS is the number of clusters for the next level of the
         * representation tree, we start with NCLUS = 1 for the root */
        nclus = 1;
        iwork[iindc1 + 0] = 0;
        iwork[iindc1 + 1] = im - 1;

        /* IDONE is the number of eigenvectors already computed in the current
         * block */
        idone = 0;

        /* loop while( IDONE < IM )
         * generate the representation tree for the current block and
         * compute the eigenvectors */
        while (idone < im) {
            /* This is a crude protection against infinitely deep trees */
            if (ndepth > m) {
                *info = -2;
                return;
            }
            /* breadth first processing of the current level of the representation
             * tree: OLDNCL = number of clusters on current level */
            oldncl = nclus;
            /* reset NCLUS to count the number of child clusters */
            nclus = 0;

            parity = 1 - parity;
            if (parity == 0) {
                oldcls = iindc1;
                newcls = iindc2;
            } else {
                oldcls = iindc2;
                newcls = iindc1;
            }

            /* Process the clusters on the current level */
            for (i = 0; i < oldncl; i++) {
                j = oldcls + 2 * i;
                /* OLDFST, OLDLST = first, last index of current cluster.
                 * cluster indices start with 0 and are relative
                 * to WBEGIN when accessing W, WGAP, WERR, Z */
                oldfst = iwork[j];
                oldlst = iwork[j + 1];

                if (ndepth > 0) {
                    /* Retrieve relatively robust representation (RRR) of cluster
                     * that has been computed at the previous level
                     * The RRR is stored in Z and overwritten once the eigenvectors
                     * have been computed or when the cluster is refined */

                    if ((dol == 0) && (dou == m - 1)) {
                        /* Get representation from location of the leftmost evalue
                         * of the cluster */
                        j = wbegin + oldfst;
                    } else {
                        if (wbegin + oldfst < dol) {
                            /* Get representation from the left end of Z array */
                            j = dol - 1;
                        } else if (wbegin + oldfst > dou) {
                            /* Get representation from the right end of Z array */
                            j = dou;
                        } else {
                            j = wbegin + oldfst;
                        }
                    }
                    cblas_scopy(in, &Z[ibegin + j * ldz], 1, &D[ibegin], 1);
                    cblas_scopy(in - 1, &Z[ibegin + (j + 1) * ldz], 1, &L[ibegin], 1);
                    sigma = Z[iend + (j + 1) * ldz];

                    /* Set the corresponding entries in Z to zero */
                    slaset("Full", in, 2, ZERO, ZERO, &Z[ibegin + j * ldz], ldz);
                }

                /* Compute DL and DLL of current RRR */
                for (j = ibegin; j <= iend - 1; j++) {
                    tmp = D[j] * L[j];
                    work[indld + j] = tmp;
                    work[indlld + j] = tmp * L[j];
                }

                if (ndepth > 0) {
                    /* P and Q are indices of the first and last eigenvalue to compute
                     * within the current block */
                    p = indexw[wbegin + oldfst];
                    q = indexw[wbegin + oldlst];
                    /* Offset for the arrays WORK, WGAP and WERR */
                    offset = indexw[wbegin];
                    /* perform limited bisection (if necessary) to get approximate
                     * eigenvalues to the precision needed. */
                    slarrb(in, &D[ibegin],
                           &work[indlld + ibegin],
                           p, q, eff_rtol1, eff_rtol2, offset,
                           &work[wbegin], &wgap[wbegin], &werr[wbegin],
                           &work[indwrk], &iwork[iindwk],
                           pivmin, spdiam, in, &iinfo);
                    if (iinfo != 0) {
                        *info = -1;
                        return;
                    }
                    /* We also recompute the extremal gaps. W holds all eigenvalues
                     * of the unshifted matrix and must be used for computation
                     * of WGAP, the entries of WORK might stem from RRRs with
                     * different shifts. The gaps from WBEGIN+OLDFST to
                     * WBEGIN+OLDLST are correctly computed in SLARRB.
                     * However, we only allow the gaps to become greater since
                     * this is what should happen when we decrease WERR */
                    if (oldfst > 0) {
                        tmp = W[wbegin + oldfst] - werr[wbegin + oldfst]
                              - W[wbegin + oldfst - 1] - werr[wbegin + oldfst - 1];
                        if (tmp > wgap[wbegin + oldfst - 1])
                            wgap[wbegin + oldfst - 1] = tmp;
                    }
                    if (wbegin + oldlst < wend) {
                        tmp = W[wbegin + oldlst + 1] - werr[wbegin + oldlst + 1]
                              - W[wbegin + oldlst] - werr[wbegin + oldlst];
                        if (tmp > wgap[wbegin + oldlst])
                            wgap[wbegin + oldlst] = tmp;
                    }
                    /* Each time the eigenvalues in WORK get refined, we store
                     * the newly found approximation with all shifts applied in W */
                    for (j = oldfst; j <= oldlst; j++) {
                        W[wbegin + j] = work[wbegin + j] + sigma;
                    }
                }

                /* Process the current node. */
                newfst = oldfst;
                for (j = oldfst; j <= oldlst; j++) {
                    if (j == oldlst) {
                        /* we are at the right end of the cluster, this is also the
                         * boundary of the child cluster */
                        newlst = j;
                    } else if (wgap[wbegin + j] >=
                               minrgp * fabsf(work[wbegin + j])) {
                        /* the right relative gap is big enough, the child cluster
                         * (NEWFST,..,NEWLST) is well separated from the following */
                        newlst = j;
                    } else {
                        /* inside a child cluster, the relative gap is not
                         * big enough. */
                        continue;
                    }

                    /* Compute size of child cluster found */
                    newsiz = newlst - newfst + 1;

                    /* NEWFTT is the place in Z where the new RRR or the computed
                     * eigenvector is to be stored */
                    if ((dol == 0) && (dou == m - 1)) {
                        /* Store representation at location of the leftmost evalue
                         * of the cluster */
                        newftt = wbegin + newfst;
                    } else {
                        if (wbegin + newfst < dol) {
                            /* Store representation at the left end of Z array */
                            newftt = dol - 1;
                        } else if (wbegin + newfst > dou) {
                            /* Store representation at the right end of Z array */
                            newftt = dou;
                        } else {
                            newftt = wbegin + newfst;
                        }
                    }

                    if (newsiz > 1) {
                        /*
                         * Current child is not a singleton but a cluster.
                         * Compute and store new representation of child.
                         *
                         * Compute left and right cluster gap.
                         *
                         * LGAP and RGAP are not computed from WORK because
                         * the eigenvalue approximations may stem from RRRs
                         * different shifts. However, W hold all eigenvalues
                         * of the unshifted matrix. Still, the entries in WGAP
                         * have to be computed from WORK since the entries
                         * in W might be of the same order so that gaps are not
                         * exhibited correctly for very close eigenvalues.
                         */
                        if (newfst == 0) {
                            lgap = W[wbegin] - werr[wbegin] - vl;
                            if (lgap < ZERO) lgap = ZERO;
                        } else {
                            lgap = wgap[wbegin + newfst - 1];
                        }
                        rgap = wgap[wbegin + newlst];

                        /*
                         * Compute left- and rightmost eigenvalue of child
                         * to high precision in order to shift as close
                         * as possible and obtain as large relative gaps
                         * as possible
                         */
                        for (k = 0; k < 2; k++) {
                            if (k == 0) {
                                p = indexw[wbegin + newfst];
                            } else {
                                p = indexw[wbegin + newlst];
                            }
                            offset = indexw[wbegin];
                            slarrb(in, &D[ibegin],
                                   &work[indlld + ibegin], p, p,
                                   rqtol, rqtol, offset,
                                   &work[wbegin], &wgap[wbegin],
                                   &werr[wbegin], &work[indwrk],
                                   &iwork[iindwk], pivmin, spdiam,
                                   in, &iinfo);
                        }

                        if ((wbegin + newlst < dol) ||
                            (wbegin + newfst > dou)) {
                            /* if the cluster contains no desired eigenvalues
                             * skip the computation of that branch of the rep. tree
                             *
                             * We could skip before the refinement of the extremal
                             * eigenvalues of the child, but then the representation
                             * tree could be different from the one when nothing is
                             * skipped. For this reason we skip at this place. */
                            idone = idone + newlst - newfst + 1;
                            goto next_child;
                        }

                        /*
                         * Compute RRR of child cluster.
                         * Note that the new RRR is stored in Z
                         *
                         * SLARRF needs LWORK = 2*N
                         */
                        slarrf(in, &D[ibegin], &L[ibegin],
                               &work[indld + ibegin],
                               newfst, newlst, &work[wbegin],
                               &wgap[wbegin], &werr[wbegin],
                               spdiam, lgap, rgap, pivmin, &tau,
                               &Z[ibegin + newftt * ldz],
                               &Z[ibegin + (newftt + 1) * ldz],
                               &work[indwrk], &iinfo);
                        if (iinfo == 0) {
                            /* a new RRR for the cluster was found by SLARRF
                             * update shift and store it */
                            ssigma = sigma + tau;
                            Z[iend + (newftt + 1) * ldz] = ssigma;
                            /* WORK() are the midpoints and WERR() the semi-width
                             * Note that the entries in W are unchanged. */
                            for (k = newfst; k <= newlst; k++) {
                                fudge = THREE * eps * fabsf(work[wbegin + k]);
                                work[wbegin + k] = work[wbegin + k] - tau;
                                fudge = fudge + FOUR * eps * fabsf(work[wbegin + k]);
                                /* Fudge errors */
                                werr[wbegin + k] = werr[wbegin + k] + fudge;
                                /* Gaps are not fudged. Provided that WERR is small
                                 * when eigenvalues are close, a zero gap indicates
                                 * that a new representation is needed for resolving
                                 * the cluster. A fudge could lead to a wrong decision
                                 * of judging eigenvalues 'separated' which in
                                 * reality are not. This could have a negative impact
                                 * on the orthogonality of the computed eigenvectors. */
                            }

                            nclus = nclus + 1;
                            k = newcls + 2 * nclus;
                            iwork[k - 2] = newfst;
                            iwork[k - 1] = newlst;
                        } else {
                            *info = -2;
                            return;
                        }
                    } else {
                        /*
                         * Compute eigenvector of singleton
                         */
                        iter = 0;

                        tol = FOUR * logf((float)in) * eps;

                        k = newfst;
                        windex = wbegin + k;
                        windmn = (windex - 1 > 0) ? windex - 1 : 0;
                        windpl = (windex + 1 < m - 1) ? windex + 1 : m - 1;
                        lambda = work[windex];
                        done = done + 1;

                        /* Check if eigenvector computation is to be skipped */
                        if ((windex < dol) || (windex > dou)) {
                            eskip = 1;
                            goto label_125;
                        } else {
                            eskip = 0;
                        }
                        left = work[windex] - werr[windex];
                        right = work[windex] + werr[windex];
                        indeig = indexw[windex];
                        /* Note that since we compute the eigenpairs for a child,
                         * all eigenvalue approximations are w.r.t the same shift.
                         * In this case, the entries in WORK should be used for
                         * computing the gaps since they exhibit even very small
                         * differences in the eigenvalues, as opposed to the
                         * entries in W which might "look" the same. */

                        if (k == 0) {
                            /* In the case RANGE='I' and with not much initial
                             * accuracy in LAMBDA and VL, the formula
                             * LGAP = MAX( ZERO, (SIGMA - VL) + LAMBDA )
                             * can lead to an overestimation of the left gap and
                             * thus to inadequately early RQI 'convergence'.
                             * Prevent this by forcing a small left gap. */
                            lgap = eps * fmaxf(fabsf(left), fabsf(right));
                        } else {
                            lgap = wgap[windmn];
                        }
                        if (k == im - 1) {
                            /* In the case RANGE='I' and with not much initial
                             * accuracy in LAMBDA and VU, the formula
                             * can lead to an overestimation of the right gap and
                             * thus to inadequately early RQI 'convergence'.
                             * Prevent this by forcing a small right gap. */
                            rgap = eps * fmaxf(fabsf(left), fabsf(right));
                        } else {
                            rgap = wgap[windex];
                        }
                        gap = (lgap < rgap) ? lgap : rgap;
                        if ((k == 0) || (k == im - 1)) {
                            /* The eigenvector support can become wrong
                             * because significant entries could be cut off due to a
                             * large GAPTOL parameter in LAR1V. Prevent this. */
                            gaptol = ZERO;
                        } else {
                            gaptol = gap * eps;
                        }
                        isupmn = in;
                        isupmx = 1;
                        /* Update WGAP so that it holds the minimum gap
                         * to the left or the right. This is crucial in the
                         * case where bisection is used to ensure that the
                         * eigenvalue is refined up to the required precision.
                         * The correct value is restored afterwards. */
                        savgap = wgap[windex];
                        wgap[windex] = gap;
                        /* We want to use the Rayleigh Quotient Correction
                         * as often as possible since it converges quadratically
                         * when we are close enough to the desired eigenvalue.
                         * However, the Rayleigh Quotient can have the wrong sign
                         * and lead us away from the desired eigenvalue. In this
                         * case, the best we can do is to use bisection. */
                        usedbs = 0;
                        usedrq = 0;
                        /* Bisection is initially turned off unless it is forced */
                        needbs = !tryrqc;
label_120:
                        /* Check if bisection should be used to refine eigenvalue */
                        if (needbs) {
                            /* Take the bisection as new iterate */
                            usedbs = 1;
                            itmp1 = iwork[iindr + windex];
                            offset = indexw[wbegin];
                            slarrb(in, &D[ibegin],
                                   &work[indlld + ibegin], indeig, indeig,
                                   ZERO, TWO * eps, offset,
                                   &work[wbegin], &wgap[wbegin],
                                   &werr[wbegin], &work[indwrk],
                                   &iwork[iindwk], pivmin, spdiam,
                                   itmp1, &iinfo);
                            if (iinfo != 0) {
                                *info = -3;
                                return;
                            }
                            lambda = work[windex];
                            /* Reset twist index from inaccurate LAMBDA to
                             * force computation of true MINGMA */
                            iwork[iindr + windex] = -1;
                        }
                        /* Given LAMBDA, compute the eigenvector. */
                        slar1v(in, 0, in - 1, lambda, &D[ibegin],
                               &L[ibegin], &work[indld + ibegin],
                               &work[indlld + ibegin],
                               pivmin, gaptol, &Z[ibegin + windex * ldz],
                               !usedbs, &negcnt, &ztz, &mingma,
                               &iwork[iindr + windex], &isuppz[2 * windex],
                               &nrminv, &resid, &rqcorr, &work[indwrk]);
                        if (iter == 0) {
                            bstres = resid;
                            bstw = lambda;
                        } else if (resid < bstres) {
                            bstres = resid;
                            bstw = lambda;
                        }
                        isupmn = (isupmn < isuppz[2 * windex]) ? isupmn : isuppz[2 * windex];
                        isupmx = (isupmx > isuppz[2 * windex + 1]) ? isupmx : isuppz[2 * windex + 1];
                        iter = iter + 1;

                        /* sin alpha <= |resid|/gap
                         * Note that both the residual and the gap are
                         * proportional to the matrix, so ||T|| doesn't play
                         * a role in the quotient */

                        /*
                         * Convergence test for Rayleigh-Quotient iteration
                         * (omitted when Bisection has been used)
                         */
                        if (resid > tol * gap && fabsf(rqcorr) > rqtol * fabsf(lambda)
                            && !usedbs) {
                            /* We need to check that the RQCORR update doesn't
                             * move the eigenvalue away from the desired one and
                             * towards a neighbor. -> protection with bisection */
                            if (indeig <= negcnt) {
                                /* The wanted eigenvalue lies to the left */
                                sgndef = -ONE;
                            } else {
                                /* The wanted eigenvalue lies to the right */
                                sgndef = ONE;
                            }
                            /* We only use the RQCORR if it improves the
                             * the iterate reasonably. */
                            if ((rqcorr * sgndef >= ZERO)
                                && (lambda + rqcorr <= right)
                                && (lambda + rqcorr >= left)) {
                                usedrq = 1;
                                /* Store new midpoint of bisection interval in WORK */
                                if (sgndef == ONE) {
                                    /* The current LAMBDA is on the left of the true
                                     * eigenvalue */
                                    left = lambda;
                                } else {
                                    /* The current LAMBDA is on the right of the true
                                     * eigenvalue */
                                    right = lambda;
                                }
                                work[windex] = HALF * (right + left);
                                /* Take RQCORR since it has the correct sign and
                                 * improves the iterate reasonably */
                                lambda = lambda + rqcorr;
                                /* Update width of error interval */
                                werr[windex] = HALF * (right - left);
                            } else {
                                needbs = 1;
                            }
                            if (right - left < rqtol * fabsf(lambda)) {
                                /* The eigenvalue is computed to bisection accuracy
                                 * compute eigenvector and stop */
                                usedbs = 1;
                                goto label_120;
                            } else if (iter < MAXITR) {
                                goto label_120;
                            } else if (iter == MAXITR) {
                                needbs = 1;
                                goto label_120;
                            } else {
                                *info = 5;
                                return;
                            }
                        } else {
                            stp2ii = 0;
                            if (usedrq && usedbs && bstres <= resid) {
                                lambda = bstw;
                                stp2ii = 1;
                            }
                            if (stp2ii) {
                                /* improve error angle by second step */
                                slar1v(in, 0, in - 1, lambda,
                                       &D[ibegin], &L[ibegin],
                                       &work[indld + ibegin],
                                       &work[indlld + ibegin],
                                       pivmin, gaptol, &Z[ibegin + windex * ldz],
                                       !usedbs, &negcnt, &ztz, &mingma,
                                       &iwork[iindr + windex],
                                       &isuppz[2 * windex],
                                       &nrminv, &resid, &rqcorr, &work[indwrk]);
                            }
                            work[windex] = lambda;
                        }

                        /*
                         * Compute FP-vector support w.r.t. whole matrix
                         */
                        isuppz[2 * windex] = isuppz[2 * windex] + oldien;
                        isuppz[2 * windex + 1] = isuppz[2 * windex + 1] + oldien;
                        zfrom = isuppz[2 * windex];
                        zto = isuppz[2 * windex + 1];
                        isupmn = isupmn + oldien;
                        isupmx = isupmx + oldien;
                        /* Ensure vector is ok if support in the RQI has changed */
                        if (isupmn < zfrom) {
                            for (ii = isupmn; ii <= zfrom - 1; ii++) {
                                Z[ii + windex * ldz] = ZERO;
                            }
                        }
                        if (isupmx > zto) {
                            for (ii = zto + 1; ii <= isupmx; ii++) {
                                Z[ii + windex * ldz] = ZERO;
                            }
                        }
                        cblas_sscal(zto - zfrom + 1, nrminv,
                                    &Z[zfrom + windex * ldz], 1);
label_125:
                        /* Update W */
                        W[windex] = lambda + sigma;
                        /* Recompute the gaps on the left and right
                         * But only allow them to become larger and not
                         * smaller (which can only happen through "bad"
                         * cancellation and doesn't reflect the theory
                         * where the initial gaps are underestimated due
                         * to WERR being too crude.) */
                        if (!eskip) {
                            if (k > 0) {
                                tmp = W[windex] - werr[windex]
                                      - W[windmn] - werr[windmn];
                                if (tmp > wgap[windmn])
                                    wgap[windmn] = tmp;
                            }
                            if (windex < wend) {
                                tmp = W[windpl] - werr[windpl]
                                      - W[windex] - werr[windex];
                                if (tmp > savgap)
                                    wgap[windex] = tmp;
                                else
                                    wgap[windex] = savgap;
                            }
                        }
                        idone = idone + 1;
                    }
                    /* here ends the code for the current child */

next_child:
                    /* Proceed to any remaining child nodes */
                    newfst = j + 1;
                } /* end for j = oldfst..oldlst */
            } /* end for i = 0..oldncl-1 */
            ndepth = ndepth + 1;
        } /* end while (idone < im) */
        ibegin = iend + 1;
        wbegin = wend + 1;
    } /* end for jblk */
}
