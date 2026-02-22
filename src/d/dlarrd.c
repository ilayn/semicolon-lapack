/**
 * @file dlarrd.c
 * @brief DLARRD computes the eigenvalues of a symmetric tridiagonal
 *        matrix to suitable accuracy.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DLARRD computes the eigenvalues of a symmetric tridiagonal
 * matrix T to suitable accuracy. This is an auxiliary code to be
 * called from DSTEMR.
 * The user may ask for all eigenvalues, all eigenvalues
 * in the half-open interval (VL, VU], or the IL-th through IU-th
 * eigenvalues.
 *
 * To avoid overflow, the matrix must be scaled so that its
 * largest element is no greater than overflow**(1/2) * underflow**(1/4)
 * in absolute value, and for greatest accuracy, it should not be much
 * smaller than that.
 *
 * See W. Kahan "Accurate Eigenvalues of a Symmetric Tridiagonal
 * Matrix", Report CS41, Computer Science Dept., Stanford
 * University, July 21, 1966.
 *
 * @param[in]     range   = 'A': all eigenvalues will be found.
 *                        = 'V': all eigenvalues in (VL, VU] will be found.
 *                        = 'I': the IL-th through IU-th eigenvalues will be found.
 * @param[in]     order   = 'B': eigenvalues grouped by split-off block.
 *                        = 'E': eigenvalues ordered smallest to largest.
 * @param[in]     n       The order of the tridiagonal matrix T. n >= 0.
 * @param[in]     vl      Lower bound of interval (if range='V').
 * @param[in]     vu      Upper bound of interval (if range='V'). vl < vu.
 * @param[in]     il      Index (0-based) of smallest eigenvalue to return (if range='I').
 * @param[in]     iu      Index (0-based) of largest eigenvalue to return (if range='I').
 * @param[in]     gers    Double precision array, dimension (2*n).
 *                        The N Gerschgorin intervals: (gers[2*i], gers[2*i+1]).
 * @param[in]     reltol  The minimum relative width of an interval.
 * @param[in]     D       Double precision array, dimension (n).
 *                        The diagonal elements of the tridiagonal matrix T.
 * @param[in]     E       Double precision array, dimension (n-1).
 *                        The off-diagonal elements of T.
 * @param[in]     E2      Double precision array, dimension (n-1).
 *                        The squared off-diagonal elements of T.
 * @param[in]     pivmin  The minimum pivot allowed in the Sturm sequence.
 * @param[in]     nsplit  The number of diagonal blocks in T.
 * @param[in]     isplit  Integer array, dimension (n).
 *                        The splitting points (0-based block boundaries).
 * @param[out]    m       The actual number of eigenvalues found.
 * @param[out]    W       Double precision array, dimension (n).
 *                        The eigenvalue approximations.
 * @param[out]    werr    Double precision array, dimension (n).
 *                        The error bounds on eigenvalues.
 * @param[out]    wl      Lower bound of interval containing wanted eigenvalues.
 * @param[out]    wu      Upper bound of interval containing wanted eigenvalues.
 * @param[out]    iblock  Integer array, dimension (n).
 *                        Block number (0-based) for each eigenvalue.
 * @param[out]    indexw  Integer array, dimension (n).
 *                        Local index (0-based) within block for each eigenvalue.
 * @param[out]    work    Double precision workspace, dimension (4*n).
 * @param[out]    iwork   Integer workspace, dimension (3*n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: some eigenvalues failed to converge or were not computed.
 */
void dlarrd(const char* range, const char* order, const INT n,
            const f64 vl, const f64 vu, const INT il, const INT iu,
            const f64* gers, const f64 reltol,
            const f64* D, const f64* E, const f64* E2,
            const f64 pivmin, const INT nsplit, const INT* isplit,
            INT* m, f64* W, f64* werr,
            f64* wl, f64* wu,
            INT* iblock, INT* indexw,
            f64* work, INT* iwork, INT* info)
{
    const f64 ZERO = 0.0;
    const f64 TWO = 2.0;
    const f64 HALF = 0.5;
    const f64 FUDGE = 2.0;

    const INT ALLRNG = 1;
    const INT VALRNG = 2;
    const INT INDRNG = 3;

    INT ncnvrg, toofew;
    INT i, ib, ibegin, idiscl, idiscu, ie, iend, iinfo,
        im, in, ioff, iout, irange, itmax, itmp1,
        itmp2, iw, iwoff, j, jblk, jdisc, je, jee, nb,
        nwl, nwu;
    f64 atoli, eps, gl, gu, rtoli, tmp1, tmp2,
           tnorm, uflow, wkill, wlu = 0.0, wul = 0.0;
    INT idumma[1];

    *info = 0;
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
    } else {
        irange = 0;
    }

    /* Check for Errors */
    if (irange <= 0) {
        *info = -1;
    } else if (!(order[0] == 'B' || order[0] == 'b' ||
                 order[0] == 'E' || order[0] == 'e')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (irange == VALRNG) {
        if (vl >= vu)
            *info = -5;
    } else if (irange == INDRNG &&
               (il < 0 || il > (1 > n ? 1 : n) - 1)) {
        *info = -6;
    } else if (irange == INDRNG &&
               (iu < ((n - 1) < il ? (n - 1) : il) || iu > n - 1)) {
        *info = -7;
    }

    if (*info != 0) {
        return;
    }

    /* Initialize error flags */
    ncnvrg = 0;
    toofew = 0;

    /* Simplification: */
    if (irange == INDRNG && il == 0 && iu == n - 1) irange = ALLRNG;

    /* Get machine constants */
    eps = dlamch("P");
    uflow = dlamch("U");

    /* Special Case when N=1 */
    if (n == 1) {
        if ((irange == ALLRNG) ||
            ((irange == VALRNG) && (D[0] > vl) && (D[0] <= vu)) ||
            ((irange == INDRNG) && (il == 0) && (iu == 0))) {
            *m = 1;
            W[0] = D[0];
            /* The computation error of the eigenvalue is zero */
            werr[0] = ZERO;
            iblock[0] = 0;
            indexw[0] = 0;
        }
        return;
    }

    /* NB is the minimum vector length for vector bisection, or 0
     * if only scalar is to be done.
     * NB = ILAENV(1, 'DSTEBZ', ' ', N, -1, -1, -1) => default NB=1 */
    nb = 1;
    if (nb <= 1) nb = 0;

    /* Find global spectral radius */
    gl = D[0];
    gu = D[0];
    for (i = 0; i < n; i++) {
        gl = gl < gers[2 * i] ? gl : gers[2 * i];
        gu = gu > gers[2 * i + 1] ? gu : gers[2 * i + 1];
    }
    /* Compute global Gerschgorin bounds and spectral diameter */
    tnorm = fabs(gl) > fabs(gu) ? fabs(gl) : fabs(gu);
    gl = gl - FUDGE * tnorm * eps * n - FUDGE * TWO * pivmin;
    gu = gu + FUDGE * tnorm * eps * n + FUDGE * TWO * pivmin;

    /* Input arguments for DLAEBZ:
     * The relative tolerance. An interval (a,b] lies within
     * "relative tolerance" if b-a < RELTOL*max(|a|,|b|) */
    rtoli = reltol;
    /* Set the absolute tolerance for interval convergence to zero to force
     * interval convergence based on relative size of the interval.
     * This is dangerous because intervals might not converge when RELTOL is
     * small. But at least a very small number should be selected so that for
     * strongly graded matrices, the code can get relatively accurate
     * eigenvalues. */
    atoli = FUDGE * TWO * uflow + FUDGE * TWO * pivmin;

    if (irange == INDRNG) {

        /* RANGE='I': Compute an interval containing eigenvalues
         * IL through IU. The initial interval [GL,GU] from the global
         * Gerschgorin bounds GL and GU is refined by DLAEBZ. */
        itmax = (INT)((log(tnorm + pivmin) - log(pivmin)) /
                log(TWO)) + 2;
        /* 0-based: work[n+0..n+5], iwork[0..5]
         * Fortran: WORK(N+1)..WORK(N+6), IWORK(1)..IWORK(6) */
        work[n + 0] = gl;
        work[n + 1] = gl;
        work[n + 2] = gu;
        work[n + 3] = gu;
        work[n + 4] = gl;
        work[n + 5] = gu;
        iwork[0] = -1;
        iwork[1] = -1;
        iwork[2] = n + 1;
        iwork[3] = n + 1;
        iwork[4] = il;
        iwork[5] = iu + 1;

        dlaebz(3, itmax, n, 2, 2, nb, atoli, rtoli, pivmin,
               D, E, E2, &iwork[4], &work[n], &work[n + 4], &iout,
               iwork, W, iblock, &iinfo);
        if (iinfo != 0) {
            *info = iinfo;
            return;
        }
        /* On exit, output intervals may not be ordered by ascending negcount */
        if (iwork[5] == iu + 1) {
            *wl = work[n + 0];
            wlu = work[n + 2];
            nwl = iwork[0];
            *wu = work[n + 3];
            wul = work[n + 1];
            nwu = iwork[3];
        } else {
            *wl = work[n + 1];
            wlu = work[n + 3];
            nwl = iwork[1];
            *wu = work[n + 2];
            wul = work[n + 0];
            nwu = iwork[2];
        }
        /* On exit, the interval [WL, WLU] contains a value with negcount NWL,
         * and [WUL, WU] contains a value with negcount NWU. */
        if (nwl < 0 || nwl >= n || nwu < 1 || nwu > n) {
            *info = 4;
            return;
        }

    } else if (irange == VALRNG) {
        *wl = vl;
        *wu = vu;

    } else if (irange == ALLRNG) {
        *wl = gl;
        *wu = gu;
    }

    /* Find Eigenvalues -- Loop Over blocks and recompute NWL and NWU.
     * NWL accumulates the number of eigenvalues .le. WL,
     * NWU accumulates the number of eigenvalues .le. WU */
    *m = 0;
    iend = -1;
    *info = 0;
    nwl = 0;
    nwu = 0;

    for (jblk = 0; jblk < nsplit; jblk++) {
        ioff = iend;
        ibegin = ioff + 1;
        iend = isplit[jblk];
        in = iend - ioff;

        if (in == 1) {
            /* 1x1 block */
            if (*wl >= D[ibegin] - pivmin)
                nwl = nwl + 1;
            if (*wu >= D[ibegin] - pivmin)
                nwu = nwu + 1;
            if (irange == ALLRNG ||
                (*wl < D[ibegin] - pivmin &&
                 *wu >= D[ibegin] - pivmin)) {
                W[*m] = D[ibegin];
                werr[*m] = ZERO;
                /* The gap for a single block doesn't matter for the later
                 * algorithm and is assigned an arbitrary large value */
                iblock[*m] = jblk;
                indexw[*m] = 0;
                *m = *m + 1;
            }

        } else {
            /* General Case - block of size IN >= 2
             * Compute local Gerschgorin interval and use it as the initial
             * interval for DLAEBZ */
            gu = D[ibegin];
            gl = D[ibegin];

            for (j = ibegin; j <= iend; j++) {
                gl = gl < gers[2 * j] ? gl : gers[2 * j];
                gu = gu > gers[2 * j + 1] ? gu : gers[2 * j + 1];
            }
            /* [JAN/28/2009]
             * change SPDIAM by TNORM in lines 2 and 3 thereafter */
            gl = gl - FUDGE * tnorm * eps * in - FUDGE * pivmin;
            gu = gu + FUDGE * tnorm * eps * in + FUDGE * pivmin;

            if (irange > 1) {
                if (gu < *wl) {
                    /* the local block contains none of the wanted eigenvalues */
                    nwl = nwl + in;
                    nwu = nwu + in;
                    continue;
                }
                /* refine search interval if possible, only range (WL,WU] matters */
                gl = gl > *wl ? gl : *wl;
                gu = gu < *wu ? gu : *wu;
                if (gl >= gu)
                    continue;
            }

            /* Find negcount of initial interval boundaries GL and GU */
            work[n + 0] = gl;
            work[n + in] = gu;
            dlaebz(1, 0, in, in, 1, nb, atoli, rtoli, pivmin,
                   &D[ibegin], &E[ibegin], &E2[ibegin],
                   idumma, &work[n], &work[n + 2 * in], &im,
                   iwork, &W[*m], &iblock[*m], &iinfo);
            if (iinfo != 0) {
                *info = iinfo;
                return;
            }

            nwl = nwl + iwork[0];
            nwu = nwu + iwork[in];
            iwoff = *m - iwork[0];

            /* Compute Eigenvalues */
            itmax = (INT)((log(gu - gl + pivmin) - log(pivmin)) /
                    log(TWO)) + 2;
            dlaebz(2, itmax, in, in, 1, nb, atoli, rtoli,
                   pivmin,
                   &D[ibegin], &E[ibegin], &E2[ibegin],
                   idumma, &work[n], &work[n + 2 * in], &iout,
                   iwork, &W[*m], &iblock[*m], &iinfo);
            if (iinfo != 0) {
                *info = iinfo;
                return;
            }

            /* Copy eigenvalues into W and IBLOCK
             * Use -JBLK for block number for unconverged eigenvalues.
             * Loop over the number of output intervals from DLAEBZ */
            for (j = 0; j < iout; j++) {
                /* eigenvalue approximation is middle point of interval */
                tmp1 = HALF * (work[j + n] + work[j + in + n]);
                /* semi length of error interval */
                tmp2 = HALF * fabs(work[j + n] - work[j + in + n]);
                if (j > iout - 1 - iinfo) {
                    /* Flag non-convergence. */
                    ncnvrg = 1;
                    ib = -(jblk + 1);
                } else {
                    ib = jblk;
                }
                for (je = iwork[j] + iwoff;
                     je < iwork[j + in] + iwoff; je++) {
                    W[je] = tmp1;
                    werr[je] = tmp2;
                    indexw[je] = je - iwoff;
                    iblock[je] = ib;
                }
            }

            *m = *m + im;
        }
    }

    /* If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU
     * If NWL+1 < IL or NWU > IU, discard extra eigenvalues. */
    if (irange == INDRNG) {
        idiscl = il - nwl;
        idiscu = nwu - iu - 1;

        if (idiscl > 0) {
            im = 0;
            for (je = 0; je < *m; je++) {
                /* Remove some of the smallest eigenvalues from the left so that
                 * at the end IDISCL =0. Move all eigenvalues up to the left. */
                if (W[je] <= wlu && idiscl > 0) {
                    idiscl = idiscl - 1;
                } else {
                    W[im] = W[je];
                    werr[im] = werr[je];
                    indexw[im] = indexw[je];
                    iblock[im] = iblock[je];
                    im = im + 1;
                }
            }
            *m = im;
        }
        if (idiscu > 0) {
            /* Remove some of the largest eigenvalues from the right so that
             * at the end IDISCU =0. Move all eigenvalues up to the left. */
            im = *m;
            for (je = *m - 1; je >= 0; je--) {
                if (W[je] >= wul && idiscu > 0) {
                    idiscu = idiscu - 1;
                } else {
                    im = im - 1;
                    W[im] = W[je];
                    werr[im] = werr[je];
                    indexw[im] = indexw[je];
                    iblock[im] = iblock[je];
                }
            }
            jee = 0;
            for (je = im; je < *m; je++) {
                W[jee] = W[je];
                werr[jee] = werr[je];
                indexw[jee] = indexw[je];
                iblock[jee] = iblock[je];
                jee = jee + 1;
            }
            *m = *m - im;
        }

        if (idiscl > 0 || idiscu > 0) {
            /* Code to deal with effects of bad arithmetic. (If N(w) is
             * monotone non-decreasing, this should never happen.)
             * Some low eigenvalues to be discarded are not in (WL,WLU],
             * or high eigenvalues to be discarded are not in (WUL,WU]
             * so just kill off the smallest IDISCL/largest IDISCU
             * eigenvalues, by marking the corresponding IBLOCK with a
             * deletion sentinel (n is never a valid 0-based block number) */
            if (idiscl > 0) {
                wkill = *wu;
                for (jdisc = 0; jdisc < idiscl; jdisc++) {
                    iw = -1;
                    for (je = 0; je < *m; je++) {
                        if (iblock[je] != n &&
                            (W[je] < wkill || iw < 0)) {
                            iw = je;
                            wkill = W[je];
                        }
                    }
                    iblock[iw] = n;
                }
            }
            if (idiscu > 0) {
                wkill = *wl;
                for (jdisc = 0; jdisc < idiscu; jdisc++) {
                    iw = -1;
                    for (je = 0; je < *m; je++) {
                        if (iblock[je] != n &&
                            (W[je] >= wkill || iw < 0)) {
                            iw = je;
                            wkill = W[je];
                        }
                    }
                    iblock[iw] = n;
                }
            }
            /* Now erase all eigenvalues with IBLOCK set to the sentinel */
            im = 0;
            for (je = 0; je < *m; je++) {
                if (iblock[je] != n) {
                    W[im] = W[je];
                    werr[im] = werr[je];
                    indexw[im] = indexw[je];
                    iblock[im] = iblock[je];
                    im = im + 1;
                }
            }
            *m = im;
        }
        if (idiscl < 0 || idiscu < 0) {
            toofew = 1;
        }
    }

    if ((irange == ALLRNG && *m != n) ||
        (irange == INDRNG && *m != iu - il + 1)) {
        toofew = 1;
    }

    /* If ORDER='B', do nothing the eigenvalues are already sorted by
     *    block.
     * If ORDER='E', sort the eigenvalues from smallest to largest */
    if ((order[0] == 'E' || order[0] == 'e') && nsplit > 1) {
        for (je = 0; je < *m - 1; je++) {
            ie = -1;
            tmp1 = W[je];
            for (j = je + 1; j < *m; j++) {
                if (W[j] < tmp1) {
                    ie = j;
                    tmp1 = W[j];
                }
            }
            if (ie >= 0) {
                tmp2 = werr[ie];
                itmp1 = iblock[ie];
                itmp2 = indexw[ie];
                W[ie] = W[je];
                werr[ie] = werr[je];
                iblock[ie] = iblock[je];
                indexw[ie] = indexw[je];
                W[je] = tmp1;
                werr[je] = tmp2;
                iblock[je] = itmp1;
                indexw[je] = itmp2;
            }
        }
    }

    *info = 0;
    if (ncnvrg)
        *info = *info + 1;
    if (toofew)
        *info = *info + 2;
    return;
}
