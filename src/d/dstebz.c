/**
 * @file dstebz.c
 * @brief DSTEBZ computes the eigenvalues of a symmetric tridiagonal matrix T.
 */

#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * DSTEBZ computes the eigenvalues of a symmetric tridiagonal
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
 * @param[in]     il      If range='I', the index of the smallest eigenvalue (1-based).
 * @param[in]     iu      If range='I', the index of the largest eigenvalue (1-based).
 * @param[in]     abstol  The absolute tolerance for the eigenvalues.
 * @param[in]     D       Double precision array, dimension (n). The diagonal elements.
 * @param[in]     E       Double precision array, dimension (n-1). The off-diagonal elements.
 * @param[out]    m       The actual number of eigenvalues found.
 * @param[out]    nsplit  The number of diagonal blocks in T.
 * @param[out]    W       Double precision array, dimension (n). The eigenvalues.
 * @param[out]    iblock  Integer array, dimension (n). Block number for each eigenvalue.
 * @param[out]    isplit  Integer array, dimension (n). The splitting points (0-based endpoints).
 * @param[out]    work    Double precision array, dimension (4*n).
 * @param[out]    iwork   Integer array, dimension (3*n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: some eigenvalues failed to converge or were not computed.
 */
void dstebz(const char* range, const char* order, const int n,
            const double vl, const double vu, const int il, const int iu,
            const double abstol,
            const double* const restrict D,
            const double* const restrict E,
            int* m, int* nsplit,
            double* const restrict W,
            int* const restrict iblock,
            int* const restrict isplit,
            double* const restrict work,
            int* const restrict iwork,
            int* info)
{
    /* Internal parameters from the Fortran source */
    const double FUDGE = 2.1;
    const double RELFAC = 2.0;

    int ncnvrg, toofew;
    int ib, ibegin, idiscl, idiscu, ie, iend, iinfo,
        im, in, iorder, iout, irange, itmax,
        itmp1, iw, iwoff, j, jb, jdisc, je, nb, nwl, nwu;
    double atoli, bnorm, gl, gu, pivmin, rtoli, safemn,
           tmp1, tmp2, tnorm, ulp, wkill, wl, wlu = 0.0, wu, wul = 0.0;
    int idumma[1];
    int m_val;

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
    } else if (irange == 3 && (il < 1 || il > (1 > n ? 1 : n))) {
        *info = -6;
    } else if (irange == 3 && (iu < (n < il ? n : il) || iu > n)) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("DSTEBZ", -(*info));
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

    /* Simplification: if RANGE='I' and IL=1 and IU=N, treat as RANGE='A' */
    if (irange == 3 && il == 1 && iu == n)
        irange = 1;

    /* Get machine constants
     * NB is the minimum vector length for vector bisection, or 0
     * if only scalar is to be done.
     * DSTEBZ has no special case in ILAENV, so NB defaults to 1,
     * and since NB<=1 implies NB=0, we use scalar only. */
    safemn = dlamch("S");
    ulp = dlamch("P");
    rtoli = ulp * RELFAC;
    nb = 0;  /* scalar bisection only */

    /* Special case when N=1 */
    if (n == 1) {
        *nsplit = 1;
        isplit[0] = 0;  /* 0-based: last index of only block is 0 */
        if (irange == 2 && (vl >= D[0] || vu < D[0])) {
            *m = 0;
        } else {
            W[0] = D[0];
            iblock[0] = 1;
            *m = 1;
        }
        return;
    }

    /* Compute Splitting Points */
    *nsplit = 1;
    work[n - 1] = 0.0;  /* Fortran: WORK(N) = ZERO */
    pivmin = 1.0;

    for (j = 1; j < n; j++) {
        /* j is 0-based index into E: E[j-1] is E(J) in Fortran where J goes 2..N
         * Actually: Fortran loop is DO J=2,N with E(J-1).
         * In 0-based: j goes 1..n-1, E[j-1] is the (j-1)-th off-diagonal.
         * D[j]*D[j-1] corresponds to Fortran D(J)*D(J-1). */
        tmp1 = E[j - 1] * E[j - 1];
        if (fabs(D[j] * D[j - 1]) * ulp * ulp + safemn > tmp1) {
            isplit[*nsplit - 1] = j - 1;  /* 0-based: end of current block */
            *nsplit = *nsplit + 1;
            work[j - 1] = 0.0;
        } else {
            work[j - 1] = tmp1;
            if (tmp1 > pivmin)
                pivmin = tmp1;
        }
    }
    isplit[*nsplit - 1] = n - 1;  /* 0-based: last block ends at n-1 */
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
        tmp1 = 0.0;

        for (j = 0; j < n - 1; j++) {
            /* Fortran: DO J=1,N-1 with TMP2 = SQRT(WORK(J))
             * work[j] = E[j]^2 (stored above for j=0..n-2) */
            tmp2 = sqrt(work[j]);
            if (D[j] + tmp1 + tmp2 > gu)
                gu = D[j] + tmp1 + tmp2;
            if (D[j] - tmp1 - tmp2 < gl)
                gl = D[j] - tmp1 - tmp2;
            tmp1 = tmp2;
        }

        if (D[n - 1] + tmp1 > gu)
            gu = D[n - 1] + tmp1;
        if (D[n - 1] - tmp1 < gl)
            gl = D[n - 1] - tmp1;

        tnorm = fabs(gl) > fabs(gu) ? fabs(gl) : fabs(gu);
        gl = gl - FUDGE * tnorm * ulp * n - FUDGE * 2.0 * pivmin;
        gu = gu + FUDGE * tnorm * ulp * n + FUDGE * pivmin;

        /* Compute iteration parameters */
        itmax = (int)((log(tnorm + pivmin) - log(pivmin)) / log(2.0)) + 2;
        if (abstol <= 0.0) {
            atoli = ulp * tnorm;
        } else {
            atoli = abstol;
        }

        /*
         * Set up the initial intervals for DLAEBZ IJOB=3 call.
         * AB is stored at work[n..] with mmax=2 (column-major, 2 rows).
         * In Fortran: WORK(N+1..N+6) maps to C work[n..n+5].
         * AB layout (mmax=2, column-major):
         *   AB[0 + 0*2] = work[n+0], AB[1 + 0*2] = work[n+1]  (column 0: lower bounds)
         *   AB[0 + 1*2] = work[n+2], AB[1 + 1*2] = work[n+3]  (column 1: upper bounds)
         * C is stored at work[n+4..n+5]:
         *   C[0] = work[n+4], C[1] = work[n+5]
         *
         * Fortran sets:
         *   WORK(N+1) = GL  -> AB[0,0]
         *   WORK(N+2) = GL  -> AB[1,0]
         *   WORK(N+3) = GU  -> AB[0,1]
         *   WORK(N+4) = GU  -> AB[1,1]
         *   WORK(N+5) = GL  -> C[0]
         *   WORK(N+6) = GU  -> C[1]
         */
        work[n + 0] = gl;  /* AB[0 + 0*2] */
        work[n + 1] = gl;  /* AB[1 + 0*2] */
        work[n + 2] = gu;  /* AB[0 + 1*2] */
        work[n + 3] = gu;  /* AB[1 + 1*2] */
        work[n + 4] = gl;  /* C[0] */
        work[n + 5] = gu;  /* C[1] */

        /*
         * NAB is stored in iwork[0..3] with mmax=2 (column-major, 2 rows).
         * NVAL is stored in iwork[4..5].
         *
         * Fortran sets:
         *   IWORK(1) = -1   -> NAB[0 + 0*2] = iwork[0]
         *   IWORK(2) = -1   -> NAB[1 + 0*2] = iwork[1]
         *   IWORK(3) = N+1  -> NAB[0 + 1*2] = iwork[2]
         *   IWORK(4) = N+1  -> NAB[1 + 1*2] = iwork[3]
         *   IWORK(5) = IL-1 -> NVAL[0] = iwork[4]
         *   IWORK(6) = IU   -> NVAL[1] = iwork[5]
         */
        iwork[0] = -1;
        iwork[1] = -1;
        iwork[2] = n + 1;
        iwork[3] = n + 1;
        iwork[4] = il - 1;
        iwork[5] = iu;

        /*
         * DLAEBZ(3, ITMAX, N, 2, 2, NB, ATOLI, RTOLI, PIVMIN,
         *         D, E, WORK, IWORK(5), WORK(N+1), WORK(N+5),
         *         IOUT, IWORK, W, IBLOCK, IINFO)
         *
         * In our C interface:
         *   ijob=3, nitmax=itmax, n=n, mmax=2, minp=2, nbmin=nb
         *   D=D, E=E, E2=work (the E^2 values stored in work[0..n-2])
         *   nval=&iwork[4], AB=&work[n], C=&work[n+4]
         *   mout=&iout, NAB=iwork, work=W, iwork=iblock
         *   info=&iinfo
         */
        dlaebz(3, itmax, n, 2, 2, nb, atoli, rtoli, pivmin,
               D, E, work,
               &iwork[4], &work[n], &work[n + 4],
               &iout, iwork, W, iblock, &iinfo);

        /*
         * Extract WL, WLU, NWL, WU, WUL, NWU.
         * Fortran checks IWORK(6)==IU; we check iwork[5]==iu.
         *
         * AB layout after dlaebz (mmax=2, column-major):
         *   AB[0+0*2] = work[n+0]  (row 0, col 0 = lower bound of interval 0)
         *   AB[1+0*2] = work[n+1]  (row 1, col 0 = lower bound of interval 1)
         *   AB[0+1*2] = work[n+2]  (row 0, col 1 = upper bound of interval 0)
         *   AB[1+1*2] = work[n+3]  (row 1, col 1 = upper bound of interval 1)
         *
         * NAB layout (mmax=2, column-major):
         *   NAB[0+0*2] = iwork[0]  (row 0, col 0)
         *   NAB[1+0*2] = iwork[1]  (row 1, col 0)
         *   NAB[0+1*2] = iwork[2]  (row 0, col 1)
         *   NAB[1+1*2] = iwork[3]  (row 1, col 1)
         *
         * Fortran WORK(N+1) = AB(1,1) = work[n+0]
         * Fortran WORK(N+2) = AB(2,1) = work[n+1]
         * Fortran WORK(N+3) = AB(1,2) = work[n+2]
         * Fortran WORK(N+4) = AB(2,2) = work[n+3]
         *
         * Fortran IWORK(1) = NAB(1,1) = iwork[0]
         * Fortran IWORK(2) = NAB(2,1) = iwork[1]
         * Fortran IWORK(3) = NAB(1,2) = iwork[2]
         * Fortran IWORK(4) = NAB(2,2) = iwork[3]
         */
        if (iwork[5] == iu) {
            /* Fortran: IWORK(6).EQ.IU */
            wl  = work[n + 0];  /* WORK(N+1) = AB(1,1) */
            wlu = work[n + 2];  /* WORK(N+3) = AB(1,2) */
            nwl = iwork[0];     /* IWORK(1) = NAB(1,1) */
            wu  = work[n + 3];  /* WORK(N+4) = AB(2,2) */
            wul = work[n + 1];  /* WORK(N+2) = AB(2,1) */
            nwu = iwork[3];     /* IWORK(4) = NAB(2,2) */
        } else {
            wl  = work[n + 1];  /* WORK(N+2) = AB(2,1) */
            wlu = work[n + 3];  /* WORK(N+4) = AB(2,2) */
            nwl = iwork[1];     /* IWORK(2) = NAB(2,1) */
            wu  = work[n + 2];  /* WORK(N+3) = AB(1,2) */
            wul = work[n + 0];  /* WORK(N+1) = AB(1,1) */
            nwu = iwork[2];     /* IWORK(3) = NAB(1,2) */
        }

        if (nwl < 0 || nwl >= n || nwu < 1 || nwu > n) {
            *info = 4;
            return;
        }
    } else {
        /*
         * RANGE='A' or 'V' -- Set ATOLI
         *
         * Compute 1-norm of the tridiagonal matrix.
         * Fortran: TNORM = MAX(|D(1)|+|E(1)|, |D(N)|+|E(N-1)|)
         * In 0-based: D[0]+|E[0]|, |D[n-1]|+|E[n-2]|
         */
        tnorm = fabs(D[0]) + fabs(E[0]);
        {
            double t = fabs(D[n - 1]) + fabs(E[n - 2]);
            if (t > tnorm) tnorm = t;
        }

        for (j = 1; j < n - 1; j++) {
            /* Fortran: DO J=2,N-1; TNORM = MAX(TNORM, |D(J)|+|E(J-1)|+|E(J)|)
             * 0-based: j=1..n-2; |D[j]| + |E[j-1]| + |E[j]| */
            tmp1 = fabs(D[j]) + fabs(E[j - 1]) + fabs(E[j]);
            if (tmp1 > tnorm) tnorm = tmp1;
        }

        if (abstol <= 0.0) {
            atoli = ulp * tnorm;
        } else {
            atoli = abstol;
        }

        if (irange == 2) {
            wl = vl;
            wu = vu;
        } else {
            wl = 0.0;
            wu = 0.0;
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
        ibegin = iend + 1;          /* 0-based start of block */
        iend = isplit[jb];           /* 0-based last index of block */
        in = iend - ibegin + 1;     /* block size */

        if (in == 1) {
            /*
             * Special Case -- IN=1
             * D[ibegin] is the single diagonal element of this block.
             */
            if (irange == 1 || wl >= D[ibegin] - pivmin)
                nwl = nwl + 1;
            if (irange == 1 || wu >= D[ibegin] - pivmin)
                nwu = nwu + 1;
            if (irange == 1 || (wl < D[ibegin] - pivmin && wu >= D[ibegin] - pivmin)) {
                W[m_val] = D[ibegin];
                iblock[m_val] = jb + 1;  /* 1-based block number */
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
            tmp1 = 0.0;

            for (j = ibegin; j < iend; j++) {
                /* Fortran: DO J=IBEGIN,IEND-1; TMP2 = ABS(E(J))
                 * In 0-based: E[j] is the off-diagonal between D[j] and D[j+1] */
                tmp2 = fabs(E[j]);
                if (D[j] + tmp1 + tmp2 > gu) gu = D[j] + tmp1 + tmp2;
                if (D[j] - tmp1 - tmp2 < gl) gl = D[j] - tmp1 - tmp2;
                tmp1 = tmp2;
            }

            if (D[iend] + tmp1 > gu) gu = D[iend] + tmp1;
            if (D[iend] - tmp1 < gl) gl = D[iend] - tmp1;

            bnorm = fabs(gl) > fabs(gu) ? fabs(gl) : fabs(gu);
            gl = gl - FUDGE * bnorm * ulp * in - FUDGE * pivmin;
            gu = gu + FUDGE * bnorm * ulp * in + FUDGE * pivmin;

            /* Compute ATOLI for the current submatrix */
            if (abstol <= 0.0) {
                double agl = fabs(gl), agu = fabs(gu);
                atoli = ulp * (agl > agu ? agl : agu);
            } else {
                atoli = abstol;
            }

            if (irange > 1) {
                if (gu < wl) {
                    nwl = nwl + in;
                    nwu = nwu + in;
                    continue;  /* GO TO 70 */
                }
                if (gl < wl) gl = wl;
                if (gu > wu) gu = wu;
                if (gl >= gu)
                    continue;  /* GO TO 70 */
            }

            /*
             * Set Up Initial Interval for DLAEBZ IJOB=1 call.
             *
             * AB is stored at work[n..] with mmax=in (column-major, in rows).
             * AB[0 + 0*in] = work[n + 0] = GL
             * AB[0 + 1*in] = work[n + in] = GU
             *
             * Fortran: WORK(N+1) = GL, WORK(N+IN+1) = GU
             */
            work[n] = gl;
            work[n + in] = gu;

            /*
             * DLAEBZ(1, 0, IN, IN, 1, NB, ATOLI, RTOLI, PIVMIN,
             *         D(IBEGIN), E(IBEGIN), WORK(IBEGIN),
             *         IDUMMA, WORK(N+1), WORK(N+2*IN+1),
             *         IM, IWORK, W(M+1), IBLOCK(M+1), IINFO)
             *
             * In our C interface:
             *   ijob=1, nitmax=0, n=in, mmax=in, minp=1, nbmin=nb
             *   D=&D[ibegin], E=&E[ibegin], E2=&work[ibegin]
             *   nval=idumma, AB=&work[n], C=&work[n+2*in]
             *   mout=&im, NAB=iwork, work=&W[m_val], iwork=&iblock[m_val]
             *   info=&iinfo
             */
            dlaebz(1, 0, in, in, 1, nb, atoli, rtoli, pivmin,
                   &D[ibegin], &E[ibegin], &work[ibegin],
                   idumma, &work[n], &work[n + 2 * in],
                   &im, iwork, &W[m_val], &iblock[m_val], &iinfo);

            nwl = nwl + iwork[0];
            nwu = nwu + iwork[in];
            iwoff = m_val - iwork[0];

            /*
             * Compute Eigenvalues via DLAEBZ IJOB=2 call.
             *
             * ITMAX = INT((LOG(GU-GL+PIVMIN)-LOG(PIVMIN))/LOG(2)) + 2
             */
            itmax = (int)((log(gu - gl + pivmin) - log(pivmin)) / log(2.0)) + 2;

            /*
             * DLAEBZ(2, ITMAX, IN, IN, 1, NB, ATOLI, RTOLI, PIVMIN,
             *         D(IBEGIN), E(IBEGIN), WORK(IBEGIN),
             *         IDUMMA, WORK(N+1), WORK(N+2*IN+1),
             *         IOUT, IWORK, W(M+1), IBLOCK(M+1), IINFO)
             */
            dlaebz(2, itmax, in, in, 1, nb, atoli, rtoli, pivmin,
                   &D[ibegin], &E[ibegin], &work[ibegin],
                   idumma, &work[n], &work[n + 2 * in],
                   &iout, iwork, &W[m_val], &iblock[m_val], &iinfo);

            /*
             * Copy Eigenvalues Into W and IBLOCK.
             * Use -(jb+1) for block number for unconverged eigenvalues.
             *
             * Fortran loop: DO J=1,IOUT
             *   TMP1 = HALF*(WORK(J+N) + WORK(J+IN+N))
             *   ...
             *   DO JE = IWORK(J)+1+IWOFF, IWORK(J+IN)+IWOFF
             *
             * In 0-based:
             *   work[n + j] = AB[j + 0*in] (lower bound for interval j, 0-based)
             *   work[n + in + j] = AB[j + 1*in] (upper bound for interval j)
             *   iwork[j] = NAB[j + 0*in] (count at lower bound)
             *   iwork[in + j] = NAB[j + 1*in] (count at upper bound)
             *
             * Fortran: WORK(J+N) is AB(J,1) in 1-based. In 0-based: work[n + (j-1)]
             *          for J=1..IOUT => work[n+0..n+iout-1]
             * But dlaebz uses 0-based internally: AB[j + 0*mmax] = work[n + j]
             * and AB[j + 1*mmax] = work[n + in + j], for j=0..iout-1.
             *
             * Fortran: IWORK(J) is NAB(J,1) in 1-based = iwork[j-1] in 0-based.
             *          IWORK(J+IN) is NAB(J,2) = iwork[j-1+in] in 0-based.
             *
             * Actually, our dlaebz uses 0-based indexing throughout, so:
             *   AB[j + 0*in] = work[n + j]     for j=0..iout-1
             *   AB[j + 1*in] = work[n + in + j] for j=0..iout-1
             *   NAB[j + 0*in] = iwork[j]       for j=0..iout-1
             *   NAB[j + 1*in] = iwork[in + j]  for j=0..iout-1
             *
             * The Fortran loop "DO J=1,IOUT" with WORK(J+N) means:
             *   J=1: WORK(1+N) = work[n+0] in 0-based (since Fortran arrays 1-based)
             *   => j_c = 0..iout-1 in C
             */
            for (j = 0; j < iout; j++) {
                tmp1 = 0.5 * (work[n + j] + work[n + in + j]);

                /* Flag non-convergence */
                if (j >= iout - iinfo) {
                    ncnvrg = 1;
                    ib = -(jb + 1);  /* negative 1-based block number */
                } else {
                    ib = jb + 1;     /* positive 1-based block number */
                }

                /*
                 * Fortran: DO JE = IWORK(J)+1+IWOFF, IWORK(J+IN)+IWOFF
                 * In 0-based: iwork[j] = NAB(j,1), iwork[in+j] = NAB(j,2)
                 *
                 * Fortran IWORK(J) is 1-based subscript into IWORK, so
                 * IWORK(J) with J=1..IOUT maps to iwork[0..iout-1] in C.
                 *
                 * The loop bounds in Fortran:
                 *   start: IWORK(J) + 1 + IWOFF
                 *   end:   IWORK(J+IN) + IWOFF
                 * These are 1-based W/IBLOCK indices.
                 *
                 * In 0-based C:
                 *   start: iwork[j] + iwoff      (already 0-based count + 0-based offset)
                 *   end:   iwork[in + j] - 1 + iwoff
                 *
                 * Actually, NAB values are eigenvalue counts (0-based: 0..n).
                 * IWOFF = m_val - iwork[0] (matching Fortran: IWOFF = M - IWORK(1))
                 *
                 * Fortran JE loop: JE = IWORK(J)+1+IWOFF to IWORK(J+IN)+IWOFF
                 * These are 1-based indices into W and IBLOCK.
                 * In 0-based: JE = IWORK(J)+IWOFF to IWORK(J+IN)+IWOFF-1
                 *
                 * Wait - let's be more careful. In Fortran:
                 *   start: IWORK(J) + 1 + IWOFF  (1-based index)
                 *   end:   IWORK(J+IN) + IWOFF   (1-based index)
                 * Convert to 0-based: subtract 1 from each:
                 *   start: IWORK(J) + IWOFF       (0-based)
                 *   end:   IWORK(J+IN) + IWOFF - 1 (0-based)
                 *
                 * But IWORK here is 1-based subscript: IWORK(J) = our iwork[j-1].
                 * For J=1..IOUT: iwork[0..iout-1] and iwork[in..in+iout-1].
                 * Since our j goes 0..iout-1: iwork[j] and iwork[in+j].
                 *
                 * IWOFF = m_val - iwork[0] corresponds to Fortran IWOFF = M - IWORK(1).
                 *
                 * So 0-based loop:
                 *   for (je = iwork[j] + iwoff; je <= iwork[in+j] + iwoff - 1; je++)
                 *   = for (je = iwork[j] + iwoff; je < iwork[in+j] + iwoff; je++)
                 */
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
        idiscl = il - 1 - nwl;
        idiscu = nwu - iu;

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
             * eigenvalues.
             */
            if (idiscl > 0) {
                wkill = wu;
                for (jdisc = 0; jdisc < idiscl; jdisc++) {
                    iw = -1;
                    for (je = 0; je < m_val; je++) {
                        if (iblock[je] != 0 && (W[je] < wkill || iw < 0)) {
                            iw = je;
                            wkill = W[je];
                        }
                    }
                    iblock[iw] = 0;
                }
            }
            if (idiscu > 0) {
                wkill = wl;
                for (jdisc = 0; jdisc < idiscu; jdisc++) {
                    iw = -1;
                    for (je = 0; je < m_val; je++) {
                        if (iblock[je] != 0 && (W[je] > wkill || iw < 0)) {
                            iw = je;
                            wkill = W[je];
                        }
                    }
                    iblock[iw] = 0;
                }
            }
            im = 0;
            for (je = 0; je < m_val; je++) {
                if (iblock[je] != 0) {
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
