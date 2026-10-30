/**
 * @file sstebz.c
 * @brief SSTEBZ computes the eigenvalues of a symmetric tridiagonal matrix T.
 */

#include <math.h>
#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"

/**
 * SSTEBZ computes the eigenvalues of a symmetric tridiagonal
 * matrix T.  The user may ask for all eigenvalues, all eigenvalues
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
 *                          = 'V': all eigenvalues in (VL, VU] will be found.
 *                          = 'I': the IL-th through IU-th eigenvalues will be found.
 * @param[in]     order   = 'B': eigenvalues grouped by split-off block.
 *                          = 'E': eigenvalues ordered from smallest to largest.
 * @param[in]     n       The order of the tridiagonal matrix T. n >= 0.
 * @param[in]     vl      If range='V', the lower bound of the interval.
 * @param[in]     vu      If range='V', the upper bound of the interval.
 * @param[in]     il      If range='I', the index of the smallest eigenvalue (0-based).
 * @param[in]     iu      If range='I', the index of the largest eigenvalue (0-based).
 * @param[in]     abstol  The absolute tolerance for the eigenvalues.
 * @param[in]     D       Double precision array, dimension (n). The diagonal elements.
 * @param[in]     E       Double precision array, dimension (n-1). The off-diagonal elements.
 * @param[out]    m       The actual number of eigenvalues found.
 * @param[out]    nsplit  The number of diagonal blocks in T.
 * @param[out]    W       Double precision array, dimension (n). The eigenvalues.
 * @param[out]    iblock  Integer array, dimension (n). Block number (0-based) for each eigenvalue.
 * @param[out]    isplit  Integer array, dimension (n). The splitting points (0-based endpoints).
 * @param[out]    work    Double precision array, dimension (4*n).
 * @param[out]    iwork   Integer array, dimension (3*n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: some eigenvalues failed to converge or were not computed.
 */
void sstebz(const char* range, const char* order, const INT n,
            const f32 vl, const f32 vu, const INT il, const INT iu,
            const f32 abstol,
            const f32* restrict D,
            const f32* restrict E,
            INT* m, INT* nsplit,
            f32* restrict W,
            INT* restrict iblock,
            INT* restrict isplit,
            f32* restrict work,
            INT* restrict iwork,
            INT* info)
{
    const f32 FUDGE = 2.3f;
    const f32 RELFAC = 2.0f;

    INT ncnvrg, toofew;
    INT ib, ibegin, idiscl, idiscu, ie, iend, iinfo,
        im, in, iorder, iout, irange, itmax,
        itmp1, iw, iwoff, j, jb, jdisc, je, nb, nwl, nwu;
    f32 atoli, bnorm, gl, gu, pivmin, rtoli, safemn,
           tmp1, tmp2, tnorm, ulp, wkill, wl, wlu = 0.0f, wu, wul = 0.0f;
    INT idumma[1];
    INT m_val;

    *info = 0;

    /* Decode RANGE */
    if (range[0] == 'A' || range[0] == 'a') {
        irange = 1;
    } else if (range[0] == 'V' || range[0] == 'v') {
        irange = 2;
    } else if (range[0] == 'I' || range[0] == 'i') {
        irange = 3;
    } else {
        irange = 0;
    }

    /* Decode ORDER */
    if (order[0] == 'B' || order[0] == 'b') {
        iorder = 2;
    } else if (order[0] == 'E' || order[0] == 'e') {
        iorder = 1;
    } else {
        iorder = 0;
    }

    /* Check for errors */
    if (irange <= 0) {
        *info = -1;
    } else if (iorder <= 0) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (irange == 2) {
        if (vl >= vu)
            *info = -5;
    } else if (irange == 3 && (il < 0 || il > (0 > n - 1 ? 0 : n - 1))) {
        *info = -6;
    } else if (irange == 3 && (iu < ((n - 1) < il ? (n - 1) : il) || iu > n - 1)) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("SSTEBZ", -(*info));
        return;
    }

    /* Initialize error flags */
    *info = 0;
    ncnvrg = 0;
    toofew = 0;

    /* Quick return if possible */
    *m = 0;
    if (n == 0)
        return;

    /* Simplifications: */
    if (irange == 3 && il == 0 && iu == n - 1)
        irange = 1;

    /* Get machine constants
     * NB is the minimum vector length for vector bisection, or 0
     * if only scalar is to be done. */
    safemn = slamch("S");
    ulp = slamch("P");
    rtoli = ulp * RELFAC;
    nb = lapack_get_nb("STEBZ");
    if (nb <= 1)
        nb = 0;

    /* Special case when N=1 */
    if (n == 1) {
        *nsplit = 1;
        isplit[0] = 0;
        if (irange == 2 && (vl >= D[0] || vu < D[0])) {
            *m = 0;
        } else {
            W[0] = D[0];
            iblock[0] = 0;
            *m = 1;
        }
        return;
    }

    /* Compute Splitting Points */
    *nsplit = 1;
    work[n - 1] = 0.0f;
    pivmin = 1.0f;

    for (j = 1; j < n; j++) {
        tmp1 = E[j - 1] * E[j - 1];
        /* Split where e(j)**2 < ulp**2 |d(j) d(j+1)| (Kahan's relative
         * criterion, as in xLARRA) or e(j) = 0.  No absolute floor: the
         * drivers scale small matrices to sqrt(SAFEMN/ULP), where a floor
         * of SAFEMN on e(j)**2 would discard off-diagonals as large as
         * sqrt(ULP) relative to the matrix.  The threshold is formed as
         * a product of scaled factors so that it cannot overflow before
         * e(j)**2 does. */
        tmp2 = (fabsf(D[j]) * ulp) * (fabsf(D[j - 1]) * ulp);
        if (tmp1 < tmp2 || tmp1 == 0.0f) {
            isplit[*nsplit - 1] = j - 1;
            *nsplit = *nsplit + 1;
            work[j - 1] = 0.0f;
        } else {
            work[j - 1] = tmp1;
            pivmin = (pivmin > tmp1) ? pivmin : tmp1;
        }
    }
    isplit[*nsplit - 1] = n - 1;
    pivmin = pivmin * safemn;

    /* Compute Interval and ATOLI */
    if (irange == 3) {
        /*
         * RANGE='I': Compute the interval containing eigenvalues
         *            IL through IU.
         *
         * Compute Gershgorin interval for entire (split) matrix
         * and use it as the initial interval.
         */
        gu = D[0];
        gl = D[0];
        tmp1 = 0.0f;

        for (j = 0; j < n - 1; j++) {
            tmp2 = sqrtf(work[j]);
            gu = (gu > D[j] + tmp1 + tmp2) ? gu : D[j] + tmp1 + tmp2;
            gl = (gl < D[j] - tmp1 - tmp2) ? gl : D[j] - tmp1 - tmp2;
            tmp1 = tmp2;
        }

        gu = (gu > D[n - 1] + tmp1) ? gu : D[n - 1] + tmp1;
        gl = (gl < D[n - 1] - tmp1) ? gl : D[n - 1] - tmp1;

        tnorm = fabsf(gl) > fabsf(gu) ? fabsf(gl) : fabsf(gu);
        gl = gl - FUDGE * tnorm * ulp * n - FUDGE * 2.0f * pivmin;
        gu = gu + FUDGE * tnorm * ulp * n + FUDGE * pivmin;

        /* Compute iteration parameters */
        itmax = (INT)((logf(tnorm + pivmin) - logf(pivmin)) / logf(2.0f)) + 2;
        if (abstol <= 0.0f) {
            atoli = ulp * tnorm;
        } else {
            atoli = abstol;
        }

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

        slaebz(3, itmax, n, 2, 2, nb, atoli, rtoli, pivmin,
               D, E, work,
               &iwork[4], &work[n], &work[n + 4],
               &iout, iwork, W, iblock, &iinfo);

        if (iwork[5] == iu + 1) {
            wl  = work[n + 0];
            wlu = work[n + 2];
            nwl = iwork[0];
            wu  = work[n + 3];
            wul = work[n + 1];
            nwu = iwork[3];
        } else {
            wl  = work[n + 1];
            wlu = work[n + 3];
            nwl = iwork[1];
            wu  = work[n + 2];
            wul = work[n + 0];
            nwu = iwork[2];
        }

        if (nwl < 0 || nwl >= n || nwu < 1 || nwu > n) {
            *info = 4;
            return;
        }
    } else {
        /*
         * RANGE='A' or 'V' -- Set ATOLI
         */
        tnorm = fabsf(D[0]) + fabsf(E[0]);
        {
            f32 t = fabsf(D[n - 1]) + fabsf(E[n - 2]);
            tnorm = (tnorm > t) ? tnorm : t;
        }

        for (j = 1; j < n - 1; j++) {
            tmp1 = fabsf(D[j]) + fabsf(E[j - 1]) + fabsf(E[j]);
            tnorm = (tnorm > tmp1) ? tnorm : tmp1;
        }

        if (abstol <= 0.0f) {
        } else {
        }

        if (irange == 2) {
            wl = vl;
            wu = vu;
        } else {
            wl = 0.0f;
            wu = 0.0f;
        }
    }

    /*
     * Find Eigenvalues -- Loop Over Blocks and recompute NWL and NWU.
     * NWL accumulates the number of eigenvalues <= WL,
     * NWU accumulates the number of eigenvalues <= WU.
     */
    m_val = 0;
    iend = -1;
    *info = 0;
    nwl = 0;
    nwu = 0;

    for (jb = 0; jb < *nsplit; jb++) {
        ibegin = iend + 1;
        iend = isplit[jb];
        in = iend - ibegin + 1;

        if (in == 1) {
            /*
             * Special Case -- IN=1
             */
            if (irange == 1 || wl >= D[ibegin] - pivmin)
                nwl = nwl + 1;
            if (irange == 1 || wu >= D[ibegin] - pivmin)
                nwu = nwu + 1;
            if (irange == 1 || (wl < D[ibegin] - pivmin && wu >= D[ibegin] - pivmin)) {
                W[m_val] = D[ibegin];
                iblock[m_val] = jb;
                m_val = m_val + 1;
            }
        } else {
            /*
             * General Case -- IN > 1
             *
             * Compute Gershgorin Interval
             * and use it as the initial interval.
             */
            gu = D[ibegin];
            gl = D[ibegin];
            tmp1 = 0.0f;

            for (j = ibegin; j < iend; j++) {
                tmp2 = fabsf(E[j]);
                gu = (gu > D[j] + tmp1 + tmp2) ? gu : D[j] + tmp1 + tmp2;
                gl = (gl < D[j] - tmp1 - tmp2) ? gl : D[j] - tmp1 - tmp2;
                tmp1 = tmp2;
            }

            gu = (gu > D[iend] + tmp1) ? gu : D[iend] + tmp1;
            gl = (gl < D[iend] - tmp1) ? gl : D[iend] - tmp1;

            bnorm = fabsf(gl) > fabsf(gu) ? fabsf(gl) : fabsf(gu);
            gl = gl - FUDGE * bnorm * ulp * in - FUDGE * pivmin;
            gu = gu + FUDGE * bnorm * ulp * in + FUDGE * pivmin;

            /* Compute ATOLI for the current submatrix */
            if (abstol <= 0.0f) {
                f32 agl = fabsf(gl), agu = fabsf(gu);
                atoli = ulp * (agl > agu ? agl : agu);
            } else {
                atoli = abstol;
            }

            if (irange > 1) {
                if (gu <= wl) {
                    nwl = nwl + in;
                    nwu = nwu + in;
                    continue;
                }
                gl = (gl > wl) ? gl : wl;
                gu = (gu < wu) ? gu : wu;
                if (gl >= gu)
                    continue;
            }

            /*
             * Set Up Initial Interval
             */
            work[n] = gl;
            work[n + in] = gu;

            slaebz(1, 0, in, in, 1, nb, atoli, rtoli, pivmin,
                   &D[ibegin], &E[ibegin], &work[ibegin],
                   idumma, &work[n], &work[n + 2 * in],
                   &im, iwork, &W[m_val], &iblock[m_val], &iinfo);

            nwl = nwl + iwork[0];
            nwu = nwu + iwork[in];
            iwoff = m_val - iwork[0];

            /*
             * Compute Eigenvalues
             */
            itmax = (INT)((logf(gu - gl + pivmin) - logf(pivmin)) / logf(2.0f)) + 2;

            slaebz(2, itmax, in, in, 1, nb, atoli, rtoli, pivmin,
                   &D[ibegin], &E[ibegin], &work[ibegin],
                   idumma, &work[n], &work[n + 2 * in],
                   &iout, iwork, &W[m_val], &iblock[m_val], &iinfo);

            /*
             * Copy Eigenvalues Into W and IBLOCK
             * Use -(jb+2) for block number for unconverged eigenvalues.
             */
            for (j = 0; j < iout; j++) {
                tmp1 = 0.5f * (work[n + j] + work[n + in + j]);

                /* Flag non-convergence */
                if (j >= iout - iinfo) {
                    ncnvrg = 1;
                    ib = -(jb + 2);
                } else {
                    ib = jb;
                }

                for (je = iwork[j] + iwoff; je < iwork[in + j] + iwoff; je++) {
                    W[je] = tmp1;
                    iblock[je] = ib;
                }
            }

            m_val = m_val + im;
        }
    }

    /*
     * If RANGE='I', then (WL,WU) contains eigenvalues NWL+1,...,NWU.
     * If NWL+1 < IL or NWU > IU, discard extra eigenvalues.
     */
    if (irange == 3) {
        im = 0;
        idiscl = il - nwl;
        idiscu = nwu - iu - 1;

        if (idiscl > 0 || idiscu > 0) {
            for (je = 0; je < m_val; je++) {
                if (W[je] <= wlu && idiscl > 0) {
                    idiscl = idiscl - 1;
                } else if (W[je] >= wul && idiscu > 0) {
                    idiscu = idiscu - 1;
                } else {
                    W[im] = W[je];
                    iblock[im] = iblock[je];
                    im = im + 1;
                }
            }
            m_val = im;
        }

        if (idiscl > 0 || idiscu > 0) {
            /*
             * Code to deal with effects of bad arithmetic:
             * Some low eigenvalues to be discarded are not in (WL,WLU],
             * or high eigenvalues to be discarded are not in (WUL,WU]
             * so just kill off the smallest IDISCL/largest IDISCU
             * eigenvalues, by simply finding the smallest/largest
             * eigenvalue(s).
             *
             * (If N(w) is monotone non-decreasing, this should never
             * happen.)
             */
            if (idiscl > 0) {
                wkill = wu;
                for (jdisc = 0; jdisc < idiscl; jdisc++) {
                    iw = -1;
                    for (je = 0; je < m_val; je++) {
                        if (iblock[je] != -1 && (W[je] < wkill || iw < 0)) {
                            iw = je;
                            wkill = W[je];
                        }
                    }
                    if (iw != -1) iblock[iw] = -1;
                }
            }
            if (idiscu > 0) {
                wkill = wl;
                for (jdisc = 0; jdisc < idiscu; jdisc++) {
                    iw = -1;
                    for (je = 0; je < m_val; je++) {
                        if (iblock[je] != -1 && (W[je] > wkill || iw < 0)) {
                            iw = je;
                            wkill = W[je];
                        }
                    }
                    if (iw != -1) iblock[iw] = -1;
                }
            }
            im = 0;
            for (je = 0; je < m_val; je++) {
                if (iblock[je] != -1) {
                    W[im] = W[je];
                    iblock[im] = iblock[je];
                    im = im + 1;
                }
            }
            m_val = im;
        }
        if (idiscl < 0 || idiscu < 0) {
            toofew = 1;
        }
    }

    /*
     * If ORDER='B', do nothing -- the eigenvalues are already sorted by block.
     * If ORDER='E', sort the eigenvalues from smallest to largest.
     */
    if (iorder == 1 && *nsplit > 1) {
        for (je = 0; je < m_val - 1; je++) {
            ie = -1;
            tmp1 = W[je];
            for (j = je + 1; j < m_val; j++) {
                if (W[j] < tmp1) {
                    ie = j;
                    tmp1 = W[j];
                }
            }
            if (ie >= 0) {
                itmp1 = iblock[ie];
                W[ie] = W[je];
                iblock[ie] = iblock[je];
                W[je] = tmp1;
                iblock[je] = itmp1;
            }
        }
    }

    *m = m_val;
    *info = 0;
    if (ncnvrg)
        *info = *info + 1;
    if (toofew)
        *info = *info + 2;
}
