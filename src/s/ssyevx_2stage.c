/**
 * @file ssyevx_2stage.c
 * @brief SSYEVX_2STAGE computes selected eigenvalues and optionally eigenvectors
 *        of a real symmetric matrix using 2-stage reduction.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>
#include <math.h>

/**
 * SSYEVX_2STAGE computes selected eigenvalues and, optionally, eigenvectors
 * of a real symmetric matrix A using the 2-stage technique for
 * the reduction to tridiagonal. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of indices
 * for the desired eigenvalues.
 *
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     range   = 'A': all eigenvalues; = 'V': eigenvalues in (vl,vu]; = 'I': il-th through iu-th
 * @param[in]     uplo    = 'U': upper triangle stored; = 'L': lower triangle stored
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in,out] A       Symmetric matrix A. On exit, destroyed.
 * @param[in]     lda     Leading dimension of A. lda >= max(1, n).
 * @param[in]     vl      Lower bound if range='V'.
 * @param[in]     vu      Upper bound if range='V'. vl < vu.
 * @param[in]     il      Smallest eigenvalue index if range='I'.
 * @param[in]     iu      Largest eigenvalue index if range='I'.
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       Number of eigenvalues found.
 * @param[out]    W       Selected eigenvalues in ascending order.
 * @param[out]    Z       Eigenvectors if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work    Workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    iwork   Integer workspace, dimension (5*n).
 * @param[out]    ifail   Indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: convergence failure.
 */
void ssyevx_2stage(
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    f32* restrict A,
    const int lda,
    const f32 vl,
    const f32 vu,
    const int il,
    const int iu,
    const f32 abstol,
    int* m,
    f32* restrict W,
    f32* restrict Z,
    const int ldz,
    f32* restrict work,
    const int lwork,
    int* restrict iwork,
    int* restrict ifail,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int alleig, indeig, lower, lquery, test, valeig, wantz;
    int i, iinfo, imax, indd, inde, indee, indibl;
    int indisp, indiwo, indtau, indwkn, indwrk, iscale;
    int itmp1, j, jj, llwork, llwrkn;
    int nsplit, lwmin, lhtrd = 0, lwtrd, kd, ib, indhous;
    f32 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f32 sigma, smlnum, tmp1, vll, vuu;

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lquery = (lwork == -1);

    *info = 0;
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!alleig && !valeig && !indeig) {
        *info = -2;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 0 || il > (0 > n - 1 ? 0 : n - 1)) {
                *info = -9;
            } else if (iu < ((n - 1) < il ? (n - 1) : il) || iu > n - 1) {
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
            lwmin = 1;
            work[0] = (f32)lwmin;
        } else {
            kd = ilaenv2stage(1, "SSYTRD_2STAGE", jobz, n, -1, -1, -1);
            ib = ilaenv2stage(2, "SSYTRD_2STAGE", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
            lwmin = (8 * n > 3 * n + lhtrd + lwtrd) ? 8 * n : 3 * n + lhtrd + lwtrd;
            work[0] = (f32)lwmin;
        }

        if (lwork < lwmin && !lquery) {
            *info = -17;
        }
    }

    if (*info != 0) {
        xerbla("SSYEVX_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

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

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);
    if (rmax > ONE / sqrtf(sqrtf(safmin))) {
        rmax = ONE / sqrtf(sqrtf(safmin));
    }

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
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
                cblas_sscal(j + 1, sigma, &A[j * lda], 1);
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

    indtau = 0;
    inde = indtau + n;
    indd = inde + n;
    indhous = indd + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    ssytrd_2stage(jobz, uplo, n, A, lda, &work[indd],
                  &work[inde], &work[indtau], &work[indhous],
                  lhtrd, &work[indwrk], llwork, &iinfo);

    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && abstol <= ZERO) {
        cblas_scopy(n, &work[indd], 1, W, 1);
        indee = indwrk + 2 * n;
        if (!wantz) {
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            ssterf(n, W, &work[indee], info);
        } else {
            slacpy("A", n, n, A, lda, Z, ldz);
            sorgtr(uplo, n, Z, ldz, &work[indtau],
                   &work[indwrk], llwork, &iinfo);
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            ssteqr(jobz, n, W, &work[indee], Z, ldz,
                   &work[indwrk], info);
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

    indibl = 0;
    indisp = indibl + n;
    indiwo = indisp + n;
    sstebz(range, wantz ? "B" : "E", n, vll, vuu, il, iu, abstll,
           &work[indd], &work[inde], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &work[indwrk],
           &iwork[indiwo], info);

    if (wantz) {
        sstein(n, &work[indd], &work[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &work[indwrk], &iwork[indiwo], ifail, info);

        indwkn = inde;
        llwrkn = lwork - indwkn;
        sormtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

L40:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

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
                cblas_sswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }

    work[0] = (f32)lwmin;
}
