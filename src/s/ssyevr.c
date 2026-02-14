/**
 * @file ssyevr.c
 * @brief SSYEVR computes selected eigenvalues and, optionally, eigenvectors of a
 *        real symmetric matrix using the Relatively Robust Representations (RRR).
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "../include/lapack_tuning.h"

/**
 * SSYEVR computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric matrix A. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of indices
 * for the desired eigenvalues.
 *
 * SSYEVR first reduces the matrix A to tridiagonal form T with a call
 * to SSYTRD. Then, whenever possible, SSYEVR calls SSTEMR to compute
 * the eigenspectrum using Relatively Robust Representations.
 *
 * @param[in]     jobz    = 'N': Compute eigenvalues only;
 *                          = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     range   = 'A': all eigenvalues will be found.
 *                          = 'V': all eigenvalues in (VL,VU] will be found.
 *                          = 'I': the IL-th through IU-th eigenvalues will be found.
 * @param[in]     uplo    = 'U': Upper triangle of A is stored;
 *                          = 'L': Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in,out] A       On entry, the symmetric matrix A.
 *                        On exit, the triangle is destroyed.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     vl      If RANGE='V', lower bound of interval. Not referenced otherwise.
 * @param[in]     vu      If RANGE='V', upper bound of interval. VL < VU. Not referenced otherwise.
 * @param[in]     il      If RANGE='I', index of smallest eigenvalue (1-based). Not referenced otherwise.
 * @param[in]     iu      If RANGE='I', index of largest eigenvalue (1-based). Not referenced otherwise.
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       The total number of eigenvalues found.
 * @param[out]    W       Array of dimension (n). The first M elements contain
 *                        the selected eigenvalues in ascending order.
 * @param[out]    Z       If JOBZ = 'V', the first M columns contain the orthonormal
 *                        eigenvectors. If JOBZ = 'N', Z is not referenced.
 * @param[in]     ldz     Leading dimension of Z. LDZ >= 1, and if JOBZ = 'V', LDZ >= N.
 * @param[out]    isuppz  Support of eigenvectors in Z. Dimension (2*max(1,M)).
 * @param[out]    work    Workspace array, dimension (max(1, lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of work. If N <= 1, LWORK >= 1, else LWORK >= 26*N.
 *                        If lwork = -1, workspace query only.
 * @param[out]    iwork   Integer workspace array, dimension (max(1, liwork)).
 *                        On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork  The dimension of iwork. If N <= 1, LIWORK >= 1, else LIWORK >= 10*N.
 *                        If liwork = -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: Internal error
 */
void ssyevr(const char* jobz, const char* range, const char* uplo,
            const int n, f32* const restrict A, const int lda,
            const f32 vl, const f32 vu, const int il, const int iu,
            const f32 abstol, int* m,
            f32* const restrict W,
            f32* const restrict Z, const int ldz,
            int* const restrict isuppz,
            f32* const restrict work, const int lwork,
            int* const restrict iwork, const int liwork,
            int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    int alleig, indeig, lower, lquery, valeig, wantz, tryrac;
    int i, ieeeok, iinfo, imax, indd, inddd, inde, indee;
    int indibl, indifl, indisp, indiwo, indtau, indwk, indwkn;
    int iscale, j, jj, liwmin, llwork, llwrkn, lwkopt, lwmin, nb, nsplit;
    f32 abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1, vll = 0.0f, vuu = 0.0f;

    /* Test the input parameters */
    /* IEEEOK = 1 means IEEE arithmetic is assumed (NaN/Inf handled properly) */
    ieeeok = 1;  /* Assume IEEE arithmetic on modern systems */

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    lquery = (lwork == -1 || liwork == -1);

    if (n <= 1) {
        lwmin = 1;
        liwmin = 1;
    } else {
        lwmin = 26 * n;
        liwmin = 10 * n;
    }

    *info = 0;
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(alleig || valeig || indeig)) {
        *info = -2;
    } else if (!(lower || uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 1 || il > (n > 1 ? n : 1)) {
                *info = -9;
            } else if (iu < (n < il ? n : il) || iu > n) {
                *info = -10;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -15;
        } else if (lwork < lwmin && !lquery) {
            *info = -18;
        } else if (liwork < liwmin && !lquery) {
            *info = -20;
        }
    }

    if (*info == 0) {
        nb = lapack_get_nb("SYTRD");
        nb = nb > lapack_get_nb("ORMTR") ? nb : lapack_get_nb("ORMTR");
        lwkopt = (nb + 1) * n > lwmin ? (nb + 1) * n : lwmin;
        work[0] = (f32)lwkopt;
        iwork[0] = liwmin;
    }

    if (*info != 0) {
        xerbla("SSYEVR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    *m = 0;
    if (n == 0) {
        work[0] = 1.0f;
        return;
    }

    if (n == 1) {
        work[0] = 1.0f;
        if (alleig || indeig) {
            *m = 1;
            W[0] = A[0];
        } else {
            if (vl < A[0] && vu >= A[0]) {
                *m = 1;
                W[0] = A[0];
            }
        }
        if (wantz) {
            Z[0] = ONE;
            isuppz[0] = 1;
            isuppz[1] = 1;
        }
        return;
    }

    /* Get machine constants */
    safmin = slamch("S");
    eps = slamch("E");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);
    if (rmax > ONE / sqrtf(sqrtf(safmin))) {
        rmax = ONE / sqrtf(sqrtf(safmin));
    }

    /* Scale matrix to allowable range, if necessary */
    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    }
    anrm = slansy("M", uplo, n, A, lda, work);
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            for (j = 0; j < n; j++) {
                cblas_sscal(n - j, sigma, &A[j + j * lda], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                cblas_sscal(j + 1, sigma, &A[0 + j * lda], 1);
            }
        }
        if (abstol > 0) {
            abstll = abstol * sigma;
        }
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    /* Initialize indices into workspaces */
    indtau = 0;          /* tau at work[0] */
    indd = indtau + n;   /* D at work[n] */
    inde = indd + n;     /* E at work[2*n] */
    inddd = inde + n;    /* DD at work[3*n] */
    indee = inddd + n;   /* EE at work[4*n] */
    indwk = indee + n;   /* remaining work at work[5*n] */
    llwork = lwork - indwk;

    /* IWORK indices */
    indibl = 0;
    indisp = indibl + n;
    indifl = indisp + n;
    indiwo = indifl + n;

    /* Call SSYTRD to reduce symmetric matrix to tridiagonal form */
    ssytrd(uplo, n, A, lda, &work[indd], &work[inde], &work[indtau],
           &work[indwk], llwork, &iinfo);

    /* If all eigenvalues are desired and ABSTOL is less than or equal
     * to zero, then call SSTERF or SSTEMR and SORMTR. */
    if ((alleig || (indeig && il == 1 && iu == n)) && ieeeok == 1) {
        if (!wantz) {
            cblas_scopy(n, &work[indd], 1, W, 1);
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            ssterf(n, W, &work[indee], info);
        } else {
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            cblas_scopy(n, &work[indd], 1, &work[inddd], 1);

            if (abstol <= TWO * n * eps) {
                tryrac = 1;
            } else {
                tryrac = 0;
            }
            sstemr(jobz, "A", n, &work[inddd], &work[indee],
                   vl, vu, il, iu, m, W, Z, ldz, n, isuppz,
                   &tryrac, &work[indwk], lwork, iwork, liwork, info);

            /* Apply orthogonal matrix used in reduction to tridiagonal
             * form to eigenvectors returned by SSTEMR. */
            if (wantz && *info == 0) {
                indwkn = inde;
                llwrkn = lwork - indwkn;
                sormtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
                       Z, ldz, &work[indwkn], llwrkn, &iinfo);
            }
        }

        if (*info == 0) {
            /* Everything worked. Skip SSTEBZ/SSTEIN. */
            *m = n;
            goto L30;
        }
        *info = 0;
    }

    /* Otherwise, call SSTEBZ and, if eigenvectors are desired, SSTEIN.
     * Also call SSTEBZ and SSTEIN if SSTEMR fails. */
    const char* order;
    if (wantz) {
        order = "B";
    } else {
        order = "E";
    }

    sstebz(range, order, n, vll, vuu, il, iu, abstll,
           &work[indd], &work[inde], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &work[indwk],
           &iwork[indiwo], info);

    if (wantz) {
        sstein(n, &work[indd], &work[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &work[indwk], &iwork[indiwo], &iwork[indifl], info);

        /* Apply orthogonal matrix used in reduction to tridiagonal
         * form to eigenvectors returned by SSTEIN. */
        indwkn = inde;
        llwrkn = lwork - indwkn;
        sormtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
L30:
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
            i = -1;
            tmp1 = W[j];
            for (jj = j + 1; jj < *m; jj++) {
                if (W[jj] < tmp1) {
                    i = jj;
                    tmp1 = W[jj];
                }
            }

            if (i >= 0) {
                W[i] = W[j];
                W[j] = tmp1;
                cblas_sswap(n, &Z[0 + i * ldz], 1, &Z[0 + j * ldz], 1);
            }
        }
    }

    /* Set WORK(1) to optimal workspace size */
    work[0] = (f32)lwkopt;
    iwork[0] = liwmin;
}
