/**
 * @file slarre.c
 * @brief SLARRE sets small off-diagonal elements to zero, finds base
 *        representations and eigenvalues for each unreduced block.
 */

#include <math.h>
#include <string.h>
#include "semicolon_lapack_single.h"

/**
 * To find the desired eigenvalues of a given real symmetric
 * tridiagonal matrix T, SLARRE sets any "small" off-diagonal
 * elements to zero, and for each unreduced block T_i, it finds
 * (a) a suitable shift at one end of the block's spectrum,
 * (b) the base representation, T_i - sigma_i I = L_i D_i L_i^T, and
 * (c) eigenvalues of each L_i D_i L_i^T.
 * The representations and eigenvalues found are then used by
 * SSTEMR to compute the eigenvectors of T.
 * The accuracy varies depending on whether bisection is used to
 * find a few eigenvalues or the dqds algorithm (subroutine SLASQ2) to
 * compute all and then discard any unwanted one.
 * As an added benefit, SLARRE also outputs the n
 * Gerschgorin intervals for the matrices L_i D_i L_i^T.
 *
 * @param[in]     range   CHARACTER*1
 *                        = 'A': ("All") all eigenvalues will be found.
 *                        = 'V': ("Value") all eigenvalues in (VL, VU].
 *                        = 'I': ("Index") the IL-th through IU-th eigenvalues
 *                                (0-based) will be found.
 * @param[in]     n       The order of the matrix. N > 0.
 * @param[in,out] vl      If RANGE='V', the lower bound for eigenvalues.
 *                        If RANGE='I' or ='A', computed by the routine.
 * @param[in,out] vu      If RANGE='V', the upper bound for eigenvalues.
 *                        If RANGE='I' or ='A', computed by the routine.
 * @param[in]     il      If RANGE='I', the index (0-based) of the smallest
 *                        eigenvalue to be returned. 0 <= il <= iu <= n-1.
 * @param[in]     iu      If RANGE='I', the index (0-based) of the largest
 *                        eigenvalue to be returned. 0 <= il <= iu <= n-1.
 * @param[in,out] D       Double precision array, dimension (N).
 *                        On entry, the N diagonal elements of T.
 *                        On exit, the diagonal elements of D_i.
 * @param[in,out] E       Double precision array, dimension (N).
 *                        On entry, the first (N-1) subdiagonal elements of T;
 *                        E[N-1] need not be set.
 *                        On exit, the subdiagonal elements of L_i. The entries
 *                        E[ISPLIT[i]], 0 <= i < NSPLIT, contain the base points
 *                        sigma_i on output.
 * @param[in,out] E2      Double precision array, dimension (N).
 *                        On entry, the first (N-1) entries contain the SQUARES
 *                        of the subdiagonal elements of T; E2[N-1] need not be set.
 *                        On exit, the entries E2[ISPLIT[i]], 0 <= i < NSPLIT,
 *                        have been set to zero.
 * @param[in]     rtol1   Tolerance for bisection convergence.
 * @param[in]     rtol2   Tolerance for bisection convergence.
 *                        An interval [LEFT,RIGHT] has converged if
 *                        RIGHT-LEFT < MAX( RTOL1*GAP, RTOL2*MAX(|LEFT|,|RIGHT|) )
 * @param[in]     spltol  The threshold for splitting.
 * @param[out]    nsplit  The number of blocks T splits into. 1 <= NSPLIT <= N.
 * @param[out]    isplit  Integer array, dimension (N).
 *                        The splitting points (0-based last indices of each block).
 * @param[out]    m       The total number of eigenvalues found.
 * @param[out]    W       Double precision array, dimension (N).
 *                        The first M elements contain the eigenvalues.
 * @param[out]    werr    Double precision array, dimension (N).
 *                        The error bound on the corresponding eigenvalue in W.
 * @param[out]    wgap    Double precision array, dimension (N).
 *                        The separation from the right neighbor eigenvalue in W.
 * @param[out]    iblock  Integer array, dimension (N).
 *                        Block indices (0-based) for each eigenvalue.
 * @param[out]    indexw  Integer array, dimension (N).
 *                        Local index (0-based) within block for each eigenvalue.
 * @param[out]    gers    Double precision array, dimension (2*N).
 *                        The N Gerschgorin intervals: (gers[2*i], gers[2*i+1]).
 * @param[out]    pivmin  The minimum pivot in the Sturm sequence for T.
 * @param[out]    work    Double precision array, dimension (6*N). Workspace.
 * @param[out]    iwork   Integer array, dimension (5*N). Workspace.
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - > 0: A problem occurred in SLARRE.
 *                         - < 0: One of the called subroutines signaled an internal
 *                           problem.
 *                         - = -1: Problem in SLARRD.
 *                         - = 2: No base representation could be found in MAXTRY
 *                           iterations.
 *                         - = -3: Problem in SLARRB when computing the refined root
 *                           representation for SLASQ2.
 *                         - = -4: Problem in SLARRB when performing bisection on the
 *                           desired part of the spectrum.
 *                         - = -5: Problem in SLASQ2.
 *                         - = -6: Problem in SLASQ2.
 */
void slarre(const char* range, const int n, f32* vl, f32* vu,
            const int il, const int iu,
            f32* D, f32* E, f32* E2,
            const f32 rtol1, const f32 rtol2, const f32 spltol,
            int* nsplit, int* isplit, int* m,
            f32* W, f32* werr, f32* wgap,
            int* iblock, int* indexw, f32* gers,
            f32* pivmin, f32* work, int* iwork, int* info)
{
    /* Parameters */
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 FOUR = 4.0f;
    const f32 HNDRD = 100.0f;
    const f32 PERT = 4.0f;
    const f32 HALF = 0.5f;
    const f32 FOURTH = 0.25f;
    const f32 FAC = 0.5f;
    const f32 MAXGROWTH = 64.0f;
    const f32 FUDGE = 2.0f;
    const int MAXTRY = 6;
    const int ALLRNG = 1;
    const int INDRNG = 2;
    const int VALRNG = 3;

    /* Local scalars */
    int forceb, norep, usedqd;
    int cnt, cnt1, cnt2, i, ibegin, idum, iend, iinfo,
        in, indl = 0, indu = 0, irange = 0, j, jblk, mb = 0, mm,
        wbegin, wend = 0;
    f32 avgap, bsrtol, clwdth, dmax_, dpivot, eabs,
           emax, eold, eps, gl, gu, isleft, isrght, rtl,
           rtol, s1, s2, safmin, sgndef, sigma, spdiam,
           tau, tmp, tmp1;

    /* Local arrays */
    int iseed[4];

    *info = 0;
    *nsplit = 0;
    *m = 0;

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    /* Decode RANGE */
    if (range[0] == 'A' || range[0] == 'a') {
        irange = ALLRNG;
    } else if (range[0] == 'V' || range[0] == 'v') {
        irange = VALRNG;
    } else if (range[0] == 'I' || range[0] == 'i') {
        irange = INDRNG;
    }

    /* Get machine constants */
    safmin = slamch("S");
    eps = slamch("P");

    /* Set parameters */
    rtl = HNDRD * eps;
    bsrtol = sqrtf(eps) * (0.5e-3f);

    /* Treat case of 1x1 matrix for quick return */
    if (n == 1) {
        if ((irange == ALLRNG) ||
            ((irange == VALRNG) && (D[0] > *vl) && (D[0] <= *vu)) ||
            ((irange == INDRNG) && (il == 0) && (iu == 0))) {
            *m = 1;
            W[0] = D[0];
            /* The computation error of the eigenvalue is zero */
            werr[0] = ZERO;
            wgap[0] = ZERO;
            iblock[0] = 0;
            indexw[0] = 0;
            gers[0] = D[0];
            gers[1] = D[0];
        }
        /* store the shift for the initial RRR, which is zero in this case */
        E[0] = ZERO;
        return;
    }

    /* General case: tridiagonal matrix of order > 1 */

    /* Init WERR, WGAP. Compute Gerschgorin intervals and spectral diameter. */
    /* Compute maximum off-diagonal entry and pivmin. */
    gl = D[0];
    gu = D[0];
    eold = ZERO;
    emax = ZERO;
    E[n - 1] = ZERO;
    for (i = 0; i < n; i++) {
        werr[i] = ZERO;
        wgap[i] = ZERO;
        eabs = fabsf(E[i]);
        if (eabs >= emax) {
            emax = eabs;
        }
        tmp1 = eabs + eold;
        gers[2 * i] = D[i] - tmp1;
        if (gl > gers[2 * i]) {
            gl = gers[2 * i];
        }
        gers[2 * i + 1] = D[i] + tmp1;
        if (gu < gers[2 * i + 1]) {
            gu = gers[2 * i + 1];
        }
        eold = eabs;
    }
    /* The minimum pivot allowed in the Sturm sequence for T */
    *pivmin = safmin * (ONE > emax * emax ? ONE : emax * emax);
    /* Compute spectral diameter. The Gerschgorin bounds give an */
    /* estimate that is wrong by at most a factor of SQRT(2) */
    spdiam = gu - gl;

    /* Compute splitting points */
    slarra(n, D, E, E2, spltol, spdiam, nsplit, isplit, &iinfo);

    /* Can force use of bisection instead of faster DQDS. */
    /* Option left in the code for future multisection work. */
    forceb = 0;

    /* Initialize USEDQD, DQDS should be used for ALLRNG unless someone */
    /* explicitly wants bisection. */
    usedqd = ((irange == ALLRNG) && (!forceb));

    if ((irange == ALLRNG) && (!forceb)) {
        /* Set interval [VL,VU] that contains all eigenvalues */
        *vl = gl;
        *vu = gu;
    } else {
        /* We call SLARRD to find crude approximations to the eigenvalues */
        /* in the desired range. In case IRANGE = INDRNG, we also obtain the */
        /* interval (VL,VU] that contains all the wanted eigenvalues. */
        /* An interval [LEFT,RIGHT] has converged if */
        /* RIGHT-LEFT < RTOL*MAX(ABS(LEFT),ABS(RIGHT)) */
        /* SLARRD needs a WORK of size 4*N, IWORK of size 3*N */
        slarrd(range, "B", n, *vl, *vu, il, iu, gers,
               bsrtol, D, E, E2, *pivmin, *nsplit, isplit,
               &mm, W, werr, vl, vu, iblock, indexw,
               work, iwork, &iinfo);
        if (iinfo != 0) {
            *info = -1;
            return;
        }
        /* Make sure that the entries M+1 to N in W, WERR, IBLOCK, INDEXW are 0 */
        for (i = mm; i < n; i++) {
            W[i] = ZERO;
            werr[i] = ZERO;
            iblock[i] = 0;
            indexw[i] = 0;
        }
    }

    /* Loop over unreduced blocks */
    ibegin = 0;
    wbegin = 0;
    for (jblk = 0; jblk < *nsplit; jblk++) {
        iend = isplit[jblk];
        in = iend - ibegin + 1;

        /* 1 X 1 block */
        if (in == 1) {
            if ((irange == ALLRNG) ||
                ((irange == VALRNG) &&
                 (D[ibegin] > *vl) && (D[ibegin] <= *vu)) ||
                ((irange == INDRNG) && (iblock[wbegin] == jblk))) {
                (*m)++;
                W[*m - 1] = D[ibegin];
                werr[*m - 1] = ZERO;
                /* The gap for a single block doesn't matter for the later */
                /* algorithm and is assigned an arbitrary large value */
                wgap[*m - 1] = ZERO;
                iblock[*m - 1] = jblk;
                indexw[*m - 1] = 0;
                wbegin++;
            }
            /* E[iend] holds the shift for the initial RRR */
            E[iend] = ZERO;
            ibegin = iend + 1;
            continue;
        }

        /* Blocks of size larger than 1x1 */

        /* E[iend] will hold the shift for the initial RRR, for now set it =0 */
        E[iend] = ZERO;

        /* Find local outer bounds GL,GU for the block */
        gl = D[ibegin];
        gu = D[ibegin];
        for (i = ibegin; i <= iend; i++) {
            if (gers[2 * i] < gl) {
                gl = gers[2 * i];
            }
            if (gers[2 * i + 1] > gu) {
                gu = gers[2 * i + 1];
            }
        }
        spdiam = gu - gl;

        if (!((irange == ALLRNG) && (!forceb))) {
            /* Count the number of eigenvalues in the current block. */
            mb = 0;
            for (i = wbegin; i < mm; i++) {
                if (iblock[i] == jblk) {
                    mb++;
                } else {
                    break;
                }
            }

            if (mb == 0) {
                /* No eigenvalue in the current block lies in the desired range */
                /* E[iend] holds the shift for the initial RRR */
                E[iend] = ZERO;
                ibegin = iend + 1;
                continue;
            } else {
                /* Decide whether dqds or bisection is more efficient */
                usedqd = ((mb > FAC * in) && (!forceb));
                wend = wbegin + mb - 1;
                /* Calculate gaps for the current block */
                /* In later stages, when representations for individual */
                /* eigenvalues are different, we use SIGMA = E[iend]. */
                sigma = ZERO;
                for (i = wbegin; i < wend; i++) {
                    wgap[i] = ZERO > (W[i + 1] - werr[i + 1] - (W[i] + werr[i])) ?
                              ZERO : (W[i + 1] - werr[i + 1] - (W[i] + werr[i]));
                }
                tmp = *vu - sigma - (W[wend] + werr[wend]);
                wgap[wend] = ZERO > tmp ? ZERO : tmp;
                /* Find local index of the first and last desired evalue. */
                indl = indexw[wbegin];
                indu = indexw[wend];
            }
        }

        if (((irange == ALLRNG) && (!forceb)) || usedqd) {
            /* Case of DQDS */
            /* Find approximations to the extremal eigenvalues of the block */
            slarrk(in, 1, gl, gu, &D[ibegin],
                   &E2[ibegin], *pivmin, rtl, &tmp, &tmp1, &iinfo);
            if (iinfo != 0) {
                *info = -1;
                return;
            }
            isleft = gl > (tmp - tmp1 - HNDRD * eps * fabsf(tmp - tmp1)) ?
                     gl : (tmp - tmp1 - HNDRD * eps * fabsf(tmp - tmp1));

            slarrk(in, in, gl, gu, &D[ibegin],
                   &E2[ibegin], *pivmin, rtl, &tmp, &tmp1, &iinfo);
            if (iinfo != 0) {
                *info = -1;
                return;
            }
            isrght = gu < (tmp + tmp1 + HNDRD * eps * fabsf(tmp + tmp1)) ?
                     gu : (tmp + tmp1 + HNDRD * eps * fabsf(tmp + tmp1));
            /* Improve the estimate of the spectral diameter */
            spdiam = isrght - isleft;
        } else {
            /* Case of bisection */
            /* Find approximations to the wanted extremal eigenvalues */
            tmp = W[wbegin] - werr[wbegin] - HNDRD * eps * fabsf(W[wbegin] - werr[wbegin]);
            isleft = gl > tmp ? gl : tmp;
            tmp = W[wend] + werr[wend] + HNDRD * eps * fabsf(W[wend] + werr[wend]);
            isrght = gu < tmp ? gu : tmp;
        }

        /* Decide whether the base representation for the current block */
        /* L_JBLK D_JBLK L_JBLK^T = T_JBLK - sigma_JBLK I */
        /* should be on the left or the right end of the current block. */
        /* The strategy is to shift to the end which is "more populated" */
        /* Furthermore, decide whether to use DQDS for the computation of */
        /* the eigenvalue approximations at the end of SLARRE or bisection. */
        /* dqds is chosen if all eigenvalues are desired or the number of */
        /* eigenvalues to be computed is large compared to the blocksize. */
        if ((irange == ALLRNG) && (!forceb)) {
            /* If all the eigenvalues have to be computed, we use dqd */
            usedqd = 1;
            /* INDL is the local index of the first eigenvalue to compute */
            indl = 0;
            indu = in - 1;
            /* MB = number of eigenvalues to compute */
            mb = in;
            wend = wbegin + mb - 1;
            /* Define 1/4 and 3/4 points of the spectrum */
            s1 = isleft + FOURTH * spdiam;
            s2 = isrght - FOURTH * spdiam;
        } else {
            /* SLARRD has computed IBLOCK and INDEXW for each eigenvalue */
            /* approximation. */
            /* choose sigma */
            if (usedqd) {
                s1 = isleft + FOURTH * spdiam;
                s2 = isrght - FOURTH * spdiam;
            } else {
                tmp = (isrght < *vu ? isrght : *vu) - (isleft > *vl ? isleft : *vl);
                s1 = (isleft > *vl ? isleft : *vl) + FOURTH * tmp;
                s2 = (isrght < *vu ? isrght : *vu) - FOURTH * tmp;
            }
        }

        /* Compute the negcount at the 1/4 and 3/4 points */
        if (mb > 1) {
            slarrc("T", in, s1, s2, &D[ibegin],
                   &E[ibegin], *pivmin, &cnt, &cnt1, &cnt2, &iinfo);
        }

        if (mb == 1) {
            sigma = gl;
            sgndef = ONE;
        } else if ((cnt1 - 1 - indl) >= (indu + 1 - cnt2)) {
            if ((irange == ALLRNG) && (!forceb)) {
                sigma = isleft > gl ? isleft : gl;
            } else if (usedqd) {
                /* use Gerschgorin bound as shift to get pos def matrix */
                /* for dqds */
                sigma = isleft;
            } else {
                /* use approximation of the first desired eigenvalue of the */
                /* block as shift */
                sigma = isleft > *vl ? isleft : *vl;
            }
            sgndef = ONE;
        } else {
            if ((irange == ALLRNG) && (!forceb)) {
                sigma = isrght < gu ? isrght : gu;
            } else if (usedqd) {
                /* use Gerschgorin bound as shift to get neg def matrix */
                /* for dqds */
                sigma = isrght;
            } else {
                /* use approximation of the first desired eigenvalue of the */
                /* block as shift */
                sigma = isrght < *vu ? isrght : *vu;
            }
            sgndef = -ONE;
        }

        /* An initial SIGMA has been chosen that will be used for computing */
        /* T - SIGMA I = L D L^T */
        /* Define the increment TAU of the shift in case the initial shift */
        /* needs to be refined to obtain a factorization with not too much */
        /* element growth. */
        if (usedqd) {
            /* The initial SIGMA was to the outer end of the spectrum */
            /* the matrix is definite and we need not retreat. */
            tau = spdiam * eps * n + TWO * (*pivmin);
            tau = tau > TWO * eps * fabsf(sigma) ? tau : TWO * eps * fabsf(sigma);
        } else {
            if (mb > 1) {
                clwdth = W[wend] + werr[wend] - W[wbegin] - werr[wbegin];
                avgap = fabsf(clwdth / (f32)(wend - wbegin));
                if (sgndef == ONE) {
                    tau = HALF * (wgap[wbegin] > avgap ? wgap[wbegin] : avgap);
                    tau = tau > werr[wbegin] ? tau : werr[wbegin];
                } else {
                    tau = HALF * (wgap[wend - 1] > avgap ? wgap[wend - 1] : avgap);
                    tau = tau > werr[wend] ? tau : werr[wend];
                }
            } else {
                tau = werr[wbegin];
            }
        }

        int found_rrr = 0;
        for (idum = 0; idum < MAXTRY; idum++) {
            /* Compute L D L^T factorization of tridiagonal matrix T - sigma I. */
            /* Store D in WORK[0:in-1], L in WORK[in:2*in-2], and reciprocals of */
            /* pivots in WORK[2*in:3*in-1] */
            dpivot = D[ibegin] - sigma;
            work[0] = dpivot;
            dmax_ = fabsf(work[0]);
            j = ibegin;
            for (i = 0; i < in - 1; i++) {
                work[2 * in + i] = ONE / work[i];
                tmp = E[j] * work[2 * in + i];
                work[in + i] = tmp;
                dpivot = (D[j + 1] - sigma) - tmp * E[j];
                work[i + 1] = dpivot;
                if (fabsf(dpivot) > dmax_) {
                    dmax_ = fabsf(dpivot);
                }
                j++;
            }
            /* check for element growth */
            if (dmax_ > MAXGROWTH * spdiam) {
                norep = 1;
            } else {
                norep = 0;
            }
            if (usedqd && !norep) {
                /* Ensure the definiteness of the representation */
                /* All entries of D (of L D L^T) must have the same sign */
                for (i = 0; i < in; i++) {
                    tmp = sgndef * work[i];
                    if (tmp < ZERO) norep = 1;
                }
            }
            if (norep) {
                /* Note that in the case of IRANGE=ALLRNG, we use the Gerschgorin */
                /* shift which makes the matrix definite. So we should end up */
                /* here really only in the case of IRANGE = VALRNG or INDRNG. */
                if (idum == MAXTRY - 2) {
                    if (sgndef == ONE) {
                        /* The fudged Gerschgorin shift should succeed */
                        sigma = gl - FUDGE * spdiam * eps * n - FUDGE * TWO * (*pivmin);
                    } else {
                        sigma = gu + FUDGE * spdiam * eps * n + FUDGE * TWO * (*pivmin);
                    }
                } else {
                    sigma = sigma - sgndef * tau;
                    tau = TWO * tau;
                }
            } else {
                /* an initial RRR is found */
                found_rrr = 1;
                break;
            }
        }

        if (!found_rrr) {
            /* if the program reaches this point, no base representation could be */
            /* found in MAXTRY iterations. */
            *info = 2;
            return;
        }

        /* At this point, we have found an initial base representation */
        /* T - SIGMA I = L D L^T with not too much element growth. */
        /* Store the shift. */
        E[iend] = sigma;
        /* Store D and L. */
        memcpy(&D[ibegin], work, (size_t)in * sizeof(f32));
        if (in > 1) {
            memcpy(&E[ibegin], &work[in], (size_t)(in - 1) * sizeof(f32));
        }

        if (mb > 1) {
            /* Perturb each entry of the base representation by a small */
            /* (but random) relative amount to overcome difficulties with */
            /* glued matrices. */
            for (i = 0; i < 4; i++) {
                iseed[i] = 1;
            }

            slarnv(2, iseed, 2 * in - 1, work);
            for (i = 0; i < in - 1; i++) {
                D[ibegin + i] = D[ibegin + i] * (ONE + eps * PERT * work[i]);
                E[ibegin + i] = E[ibegin + i] * (ONE + eps * PERT * work[in + i]);
            }
            D[iend] = D[iend] * (ONE + eps * FOUR * work[in - 1]);
        }

        /* Don't update the Gerschgorin intervals because keeping track */
        /* of the updates would be too much work in SLARRV. */
        /* We update W instead and use it to locate the proper Gerschgorin */
        /* intervals. */

        /* Compute the required eigenvalues of L D L' by bisection or dqds */
        if (!usedqd) {
            /* If SLARRD has been used, shift the eigenvalue approximations */
            /* according to their representation. This is necessary for */
            /* a uniform SLARRV since dqds computes eigenvalues of the */
            /* shifted representation. In SLARRV, W will always hold the */
            /* UNshifted eigenvalue approximation. */
            for (j = wbegin; j <= wend; j++) {
                W[j] = W[j] - sigma;
                werr[j] = werr[j] + fabsf(W[j]) * eps;
            }
            /* call SLARRB to reduce eigenvalue error of the approximations */
            /* from SLARRD */
            for (i = ibegin; i < iend; i++) {
                work[i] = D[i] * E[i] * E[i];
            }
            /* use bisection to find EV from INDL to INDU */
            slarrb(in, &D[ibegin], &work[ibegin],
                   indl, indu, rtol1, rtol2, indl,
                   &W[wbegin], &wgap[wbegin], &werr[wbegin],
                   &work[2 * n], iwork, *pivmin, spdiam,
                   in, &iinfo);
            if (iinfo != 0) {
                *info = -4;
                return;
            }
            /* SLARRB computes all gaps correctly except for the last one */
            /* Record distance to VU/GU */
            tmp = (*vu - sigma) - (W[wend] + werr[wend]);
            wgap[wend] = ZERO > tmp ? ZERO : tmp;
            for (i = indl; i <= indu; i++) {
                (*m)++;
                iblock[*m - 1] = jblk;
                indexw[*m - 1] = i;
            }
        } else {
            /* Call dqds to get all eigs (and then possibly delete unwanted */
            /* eigenvalues). */
            /* Note that dqds finds the eigenvalues of the L D L^T representation */
            /* of T to high relative accuracy. High relative accuracy */
            /* might be lost when the shift of the RRR is subtracted to obtain */
            /* the eigenvalues of T. However, T is not guaranteed to define its */
            /* eigenvalues to high relative accuracy anyway. */
            /* Set RTOL to the order of the tolerance used in SLASQ2 */
            /* This is an ESTIMATED error, the worst case bound is 4*N*EPS */
            /* which is usually too large and requires unnecessary work to be */
            /* done by bisection when computing the eigenvectors */
            rtol = logf((f32)in) * FOUR * eps;
            j = ibegin;
            for (i = 0; i < in - 1; i++) {
                work[2 * i] = fabsf(D[j]);
                work[2 * i + 1] = E[j] * E[j] * work[2 * i];
                j++;
            }
            work[2 * in - 2] = fabsf(D[iend]);
            work[2 * in - 1] = ZERO;
            slasq2(in, work, &iinfo);
            if (iinfo != 0) {
                /* If IINFO = -5 then an index is part of a tight cluster */
                /* and should be changed. The index is in IWORK[0] and the */
                /* gap is in WORK[N] */
                *info = -5;
                return;
            } else {
                /* Test that all eigenvalues are positive as expected */
                for (i = 0; i < in; i++) {
                    if (work[i] < ZERO) {
                        *info = -6;
                        return;
                    }
                }
            }
            if (sgndef > ZERO) {
                for (i = indl; i <= indu; i++) {
                    (*m)++;
                    W[*m - 1] = work[in - 1 - i];
                    iblock[*m - 1] = jblk;
                    indexw[*m - 1] = i;
                }
            } else {
                for (i = indl; i <= indu; i++) {
                    (*m)++;
                    W[*m - 1] = -work[i];
                    iblock[*m - 1] = jblk;
                    indexw[*m - 1] = i;
                }
            }

            for (i = *m - mb; i < *m; i++) {
                /* the value of RTOL below should be the tolerance in SLASQ2 */
                werr[i] = rtol * fabsf(W[i]);
            }
            for (i = *m - mb; i < *m - 1; i++) {
                /* compute the right gap between the intervals */
                tmp = W[i + 1] - werr[i + 1] - (W[i] + werr[i]);
                wgap[i] = ZERO > tmp ? ZERO : tmp;
            }
            tmp = (*vu - sigma) - (W[*m - 1] + werr[*m - 1]);
            wgap[*m - 1] = ZERO > tmp ? ZERO : tmp;
        }
        /* proceed with next block */
        ibegin = iend + 1;
        wbegin = wend + 1;
    }

    return;
}
