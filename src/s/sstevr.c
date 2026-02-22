/**
 * @file sstevr.c
 * @brief SSTEVR computes selected eigenvalues and, optionally, eigenvectors
 *        of a real symmetric tridiagonal matrix T.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SSTEVR computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric tridiagonal matrix T.  Eigenvalues and
 * eigenvectors can be selected by specifying either a range of values
 * or a range of indices for the desired eigenvalues.
 *
 * Whenever possible, SSTEVR calls SSTEMR to compute the
 * eigenspectrum using Relatively Robust Representations.  SSTEMR
 * computes eigenvalues by the dqds algorithm, while orthogonal
 * eigenvectors are computed from various "good" L D L^T representations
 * (also known as Relatively Robust Representations). Gram-Schmidt
 * orthogonalization is avoided as far as possible. More specifically,
 * the various steps of the algorithm are as follows. For the i-th
 * unreduced block of T,
 *    (a) Compute T - sigma_i = L_i D_i L_i^T, such that L_i D_i L_i^T
 *         is a relatively robust representation,
 *    (b) Compute the eigenvalues, lambda_j, of L_i D_i L_i^T to high
 *        relative accuracy by the dqds algorithm,
 *    (c) If there is a cluster of close eigenvalues, "choose" sigma_i
 *        close to the cluster, and go to step (a),
 *    (d) Given the approximate eigenvalue lambda_j of L_i D_i L_i^T,
 *        compute the corresponding eigenvector by forming a
 *        rank-revealing twisted factorization.
 * The desired accuracy of the output can be specified by the input
 * parameter ABSTOL.
 *
 * For more details, see "A new O(n^2) algorithm for the symmetric
 * tridiagonal eigenvalue/eigenvector problem", by Inderjit Dhillon,
 * Computer Science Division Technical Report No. UCB//CSD-97-971,
 * UC Berkeley, May 1997.
 *
 * Note 1 : SSTEVR calls SSTEMR when the full spectrum is requested
 * on machines which conform to the ieee-754 floating point standard.
 * SSTEVR calls SSTEBZ and SSTEIN on non-ieee machines and
 * when partial spectrum requests are made.
 *
 * Normal execution of SSTEMR may create NaNs and infinities and
 * hence may abort due to a floating point exception in environments
 * which do not handle NaNs and infinities in the ieee standard default
 * manner.
 *
 * @param[in]     jobz    = 'N':  Compute eigenvalues only;
 *                          = 'V':  Compute eigenvalues and eigenvectors.
 * @param[in]     range   = 'A': all eigenvalues will be found.
 *                          = 'V': all eigenvalues in the half-open interval (VL,VU]
 *                                 will be found.
 *                          = 'I': the IL-th through IU-th eigenvalues will be found
 *                                 (0-based indices).
 * @param[in]     n       The order of the matrix. n >= 0.
 * @param[in,out] D       Double precision array, dimension (n).
 *                        On entry, the n diagonal elements of the tridiagonal matrix.
 *                        On exit, D may be multiplied by a constant factor chosen
 *                        to avoid over/underflow in computing the eigenvalues.
 * @param[in,out] E       Double precision array, dimension (max(1,n-1)).
 *                        On entry, the (n-1) subdiagonal elements of the tridiagonal
 *                        matrix in elements 0 to n-2.
 *                        On exit, E may be multiplied by a constant factor chosen
 *                        to avoid over/underflow in computing the eigenvalues.
 * @param[in]     vl      If range='V', the lower bound of the interval to
 *                        be searched for eigenvalues. vl < vu.
 *                        Not referenced if range = 'A' or 'I'.
 * @param[in]     vu      If range='V', the upper bound of the interval to
 *                        be searched for eigenvalues. vl < vu.
 *                        Not referenced if range = 'A' or 'I'.
 * @param[in]     il      If range='I', the index of the smallest eigenvalue to be
 *                        returned (0-based). 0 <= il <= iu < n, if n > 0.
 *                        Not referenced if range = 'A' or 'V'.
 * @param[in]     iu      If range='I', the index of the largest eigenvalue to be
 *                        returned (0-based). 0 <= il <= iu < n, if n > 0.
 *                        Not referenced if range = 'A' or 'V'.
 * @param[in]     abstol  The absolute error tolerance for the eigenvalues.
 *                        An approximate eigenvalue is accepted as converged
 *                        when it is determined to lie in an interval [a,b]
 *                        of width less than or equal to
 *                            ABSTOL + EPS * max( |a|,|b| ),
 *                        where EPS is the machine precision. If ABSTOL is less than
 *                        or equal to zero, then EPS*|T| will be used in its place.
 * @param[out]    m       The total number of eigenvalues found. 0 <= m <= n.
 *                        If range = 'A', m = n, and if range = 'I', m = iu-il+1.
 * @param[out]    W       Double precision array, dimension (n).
 *                        The first m elements contain the selected eigenvalues in
 *                        ascending order.
 * @param[out]    Z       Double precision array, dimension (ldz, max(1,m)).
 *                        If jobz = 'V', then if info = 0, the first m columns of Z
 *                        contain the orthonormal eigenvectors of the matrix
 *                        corresponding to the selected eigenvalues.
 * @param[in]     ldz     The leading dimension of the array Z. ldz >= 1, and if
 *                        jobz = 'V', ldz >= max(1,n).
 * @param[out]    isuppz  Integer array, dimension (2*max(1,m)).
 *                        The support of the eigenvectors in Z, i.e., the indices
 *                        indicating the nonzero elements in Z (0-based).
 * @param[out]    work    Double precision array, dimension (max(1,lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of the array work. lwork >= max(1,20*n).
 *                        If lwork = -1, then a workspace query is assumed.
 * @param[out]    iwork   Integer array, dimension (max(1,liwork)).
 *                        On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork  The dimension of the array iwork. liwork >= max(1,10*n).
 *                        If liwork = -1, then a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: Internal error
 */
void sstevr(const char* jobz, const char* range, const INT n,
            f32* D, f32* E,
            const f32 vl, const f32 vu,
            const INT il, const INT iu,
            const f32 abstol,
            INT* m, f32* W, f32* Z, const INT ldz,
            INT* isuppz, f32* work, const INT lwork,
            INT* iwork, const INT liwork, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    INT wantz, alleig, valeig, indeig, lquery, test;
    INT tryrac;
    INT i, imax, iscale, j, jj, lwmin, liwmin, nsplit, itmp1;
    f32 bignum, eps, rmax, rmin, safmin, sigma, smlnum,
           tmp1, tnrm, vll, vuu;

    /* Workspace indices for fallback path (SSTEBZ/SSTEIN) */
    INT indibl, indisp, indifl, indiwo;

    /* ILAENV(10, 'SSTEVR', ...) = 1: IEEE arithmetic is trusted */
    INT ieee = 1;

    /* Decode JOBZ */
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');

    /* Decode RANGE */
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    /* Workspace query? */
    lquery = (lwork == -1 || liwork == -1);

    /* Minimum workspace sizes */
    lwmin = 1 > 20 * n ? 1 : 20 * n;
    liwmin = 1 > 10 * n ? 1 : 10 * n;

    /* Test the input parameters */
    *info = 0;
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(alleig || valeig || indeig)) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -7;
        } else if (indeig) {
            /* 0-based: 0 <= il <= iu < n */
            if (il < 0 || il > (0 > n - 1 ? 0 : n - 1)) {
                *info = -8;
            } else if (iu < (n - 1 < il ? n - 1 : il) || iu > n - 1) {
                *info = -9;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -14;
        }
    }

    if (*info == 0) {
        work[0] = (f32)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -17;
        } else if (liwork < liwmin && !lquery) {
            *info = -19;
        }
    }

    if (*info != 0) {
        xerbla("SSTEVR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    *m = 0;
    if (n == 0)
        return;

    if (n == 1) {
        if (alleig || indeig) {
            *m = 1;
            W[0] = D[0];
        } else {
            if (vl < D[0] && vu >= D[0]) {
                *m = 1;
                W[0] = D[0];
            }
        }
        if (wantz)
            Z[0] = ONE;
        return;
    }

    /* Get machine constants */
    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum) < ONE / sqrtf(sqrtf(safmin)) ?
           sqrtf(bignum) : ONE / sqrtf(sqrtf(safmin));

    /* Scale matrix to allowable range, if necessary */
    iscale = 0;
    vll = ZERO;
    vuu = ZERO;
    if (valeig) {
        vll = vl;
        vuu = vu;
    }

    tnrm = slanst("M", n, D, E);
    if (tnrm > ZERO && tnrm < rmin) {
        iscale = 1;
        sigma = rmin / tnrm;
    } else if (tnrm > rmax) {
        iscale = 1;
        sigma = rmax / tnrm;
    }
    if (iscale == 1) {
        cblas_sscal(n, sigma, D, 1);
        cblas_sscal(n - 1, sigma, E, 1);
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    /* Initialize indices into workspaces. Note: These indices are used only
     * if SSTERF or SSTEMR fail.
     *
     * IWORK(INDIBL:INDIBL+M-1) corresponds to IBLOCK in SSTEBZ and
     * stores the block indices of each of the M<=N eigenvalues. */
    indibl = 0;
    /* IWORK(INDISP:INDISP+NSPLIT-1) corresponds to ISPLIT in SSTEBZ and
     * stores the starting and finishing indices of each block. */
    indisp = indibl + n;
    /* IWORK(INDIFL:INDIFL+N-1) stores the indices of eigenvectors
     * that corresponding to eigenvectors that fail to converge in
     * SSTEIN.  This information is discarded; if any fail, the driver
     * returns INFO > 0. */
    indifl = indisp + n;
    /* INDIWO is the offset of the remaining integer workspace. */
    indiwo = indisp + n;

    /* If all eigenvalues are desired, then
     * call SSTERF or SSTEMR.  If this fails for some eigenvalue, then
     * try SSTEBZ. */

    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && ieee == 1) {
        cblas_scopy(n - 1, E, 1, work, 1);
        if (!wantz) {
            cblas_scopy(n, D, 1, W, 1);
            ssterf(n, W, work, info);
        } else {
            cblas_scopy(n, D, 1, &work[n], 1);
            if (abstol <= TWO * n * eps) {
                tryrac = 1;
            } else {
                tryrac = 0;
            }
            sstemr(jobz, "A", n, &work[n], work, vl, vu, il,
                   iu, m, W, Z, ldz, n, isuppz, &tryrac,
                   &work[2 * n], lwork - 2 * n, iwork, liwork, info);
        }
        if (*info == 0) {
            *m = n;
            goto L10;
        }
        *info = 0;
    }

    /* Otherwise, call SSTEBZ and, if eigenvectors are desired, SSTEIN. */

    {
        const char* order;
        if (wantz) {
            order = "B";
        } else {
            order = "E";
        }

        sstebz(range, order, n, vll, vuu, il, iu, abstol, D, E,
               m, &nsplit, W, &iwork[indibl], &iwork[indisp], work,
               &iwork[indiwo], info);
    }

    if (wantz) {
        sstein(n, D, E, *m, W, &iwork[indibl],
               &iwork[indisp],
               Z, ldz, work, &iwork[indiwo], &iwork[indifl],
               info);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
L10:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    /* If eigenvalues are not in order, then sort them, along with
     * eigenvectors. */
    if (wantz) {
        for (j = 0; j < *m - 1; j++) {
            i = 0;
            tmp1 = W[j];
            for (jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    i = jj;
                    tmp1 = W[jj];
                }
            }

            if (i != 0) {
                itmp1 = iwork[i];
                W[i] = W[j];
                iwork[i] = iwork[j];
                W[j] = tmp1;
                iwork[j] = itmp1;
                cblas_sswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
            }
        }
    }

    work[0] = (f32)lwmin;
    iwork[0] = liwmin;
    return;
}
