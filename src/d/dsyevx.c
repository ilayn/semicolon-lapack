/**
 * @file dsyevx.c
 * @brief DSYEVX computes selected eigenvalues and, optionally, eigenvectors of a
 *        real symmetric matrix using bisection and inverse iteration.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "../include/lapack_tuning.h"

/**
 * DSYEVX computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric matrix A. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of indices
 * for the desired eigenvalues.
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
 * @param[out]    work    Workspace array, dimension (max(1, lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of work. LWORK >= 1 when N <= 1, otherwise 8*N.
 *                        If lwork = -1, workspace query only.
 * @param[out]    iwork   Integer workspace array, dimension (5*N).
 * @param[out]    ifail   If JOBZ = 'V', indices of eigenvectors that failed to converge.
 *                        If JOBZ = 'N', IFAIL is not referenced.
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal value
 *                        > 0: if info = i, then i eigenvectors failed to converge.
 */
void dsyevx(const char* jobz, const char* range, const char* uplo,
            const int n, double* const restrict A, const int lda,
            const double vl, const double vu, const int il, const int iu,
            const double abstol, int* m,
            double* const restrict W,
            double* const restrict Z, const int ldz,
            double* const restrict work, const int lwork,
            int* const restrict iwork,
            int* const restrict ifail,
            int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int alleig, indeig, lower, lquery, test, valeig, wantz;
    int i, iinfo, imax, indd, inde, indee, indibl, indisp, indiwo;
    int indtau, indwkn, indwrk, iscale, itmp1, j, jj, llwork;
    int llwrkn, lwkmin, lwkopt, nb, nsplit;
    double abstll, anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum, tmp1, vll = 0.0, vuu = 0.0;

    /* Test the input parameters */
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lquery = (lwork == -1);

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
        }
    }

    if (*info == 0) {
        if (n <= 1) {
            lwkmin = 1;
            lwkopt = 1;
        } else {
            lwkmin = 8 * n;
            nb = lapack_get_nb("SYTRD");
            nb = nb > lapack_get_nb("ORMTR") ? nb : lapack_get_nb("ORMTR");
            lwkopt = (nb + 3) * n > lwkmin ? (nb + 3) * n : lwkmin;
        }
        work[0] = (double)lwkopt;

        if (lwork < lwkmin && !lquery) {
            *info = -17;
        }
    }

    if (*info != 0) {
        xerbla("DSYEVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    *m = 0;
    if (n == 0) {
        return;
    }

    if (n == 1) {
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
        }
        return;
    }

    /* Get machine constants */
    safmin = dlamch("S");
    eps = dlamch("E");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);
    if (rmax > ONE / sqrt(sqrt(safmin))) {
        rmax = ONE / sqrt(sqrt(safmin));
    }

    /* Scale matrix to allowable range, if necessary */
    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    }
    anrm = dlansy("M", uplo, n, A, lda, work);
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
                cblas_dscal(n - j, sigma, &A[j + j * lda], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                cblas_dscal(j + 1, sigma, &A[0 + j * lda], 1);
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

    /* Call DSYTRD to reduce symmetric matrix to tridiagonal form */
    indtau = 0;          /* tau at work[0] */
    inde = indtau + n;   /* E at work[n] */
    indd = inde + n;     /* D at work[2*n] */
    indwrk = indd + n;   /* remaining work at work[3*n] */
    llwork = lwork - indwrk;

    dsytrd(uplo, n, A, lda, &work[indd], &work[inde], &work[indtau],
           &work[indwrk], llwork, &iinfo);

    /* If all eigenvalues are desired and ABSTOL is less than or equal to
     * zero, then call DSTERF or DORGTR and DSTEQR. If this fails for
     * some eigenvalue, then try DSTEBZ. */
    test = 0;
    if (indeig) {
        if (il == 1 && iu == n) {
            test = 1;
        }
    }
    if ((alleig || test) && abstol <= ZERO) {
        cblas_dcopy(n, &work[indd], 1, W, 1);
        indee = indwrk + 2 * n;
        if (!wantz) {
            cblas_dcopy(n - 1, &work[inde], 1, &work[indee], 1);
            dsterf(n, W, &work[indee], info);
        } else {
            dlacpy("A", n, n, A, lda, Z, ldz);
            dorgtr(uplo, n, Z, ldz, &work[indtau], &work[indwrk], llwork, &iinfo);
            cblas_dcopy(n - 1, &work[inde], 1, &work[indee], 1);
            dsteqr(jobz, n, W, &work[indee], Z, ldz, &work[indwrk], info);
            if (*info == 0) {
                for (i = 0; i < n; i++) {
                    ifail[i] = 0;
                }
            }
        }
        if (*info == 0) {
            *m = n;
            goto L40;
        }
        *info = 0;
    }

    /* Otherwise, call DSTEBZ and, if eigenvectors are desired, DSTEIN. */
    const char* order;
    if (wantz) {
        order = "B";
    } else {
        order = "E";
    }
    indibl = 0;
    indisp = indibl + n;
    indiwo = indisp + n;
    dstebz(range, order, n, vll, vuu, il, iu, abstll,
           &work[indd], &work[inde], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &work[indwrk],
           &iwork[indiwo], info);

    if (wantz) {
        dstein(n, &work[indd], &work[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &work[indwrk], &iwork[indiwo], ifail, info);

        /* Apply orthogonal matrix used in reduction to tridiagonal
         * form to eigenvectors returned by DSTEIN. */
        indwkn = inde;
        llwrkn = lwork - indwkn;
        dormtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
L40:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, W, 1);
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
                itmp1 = iwork[indibl + i];
                W[i] = W[j];
                iwork[indibl + i] = iwork[indibl + j];
                W[j] = tmp1;
                iwork[indibl + j] = itmp1;
                cblas_dswap(n, &Z[0 + i * ldz], 1, &Z[0 + j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }

    /* Set WORK(1) to optimal workspace size */
    work[0] = (double)lwkopt;
}
