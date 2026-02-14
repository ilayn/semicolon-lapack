/**
 * @file zheevr_2stage.c
 * @brief ZHEEVR_2STAGE computes selected eigenvalues and optionally eigenvectors
 *        of a complex Hermitian matrix using the 2-stage technique and RRR algorithm.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHEEVR_2STAGE computes selected eigenvalues and, optionally, eigenvectors
 * of a complex Hermitian matrix A using the 2stage technique for
 * the reduction to tridiagonal. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of indices
 * for the desired eigenvalues.
 *
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     range   = 'A': all eigenvalues; = 'V': eigenvalues in (vl,vu];
 *                          = 'I': il-th through iu-th eigenvalues.
 * @param[in]     uplo    = 'U': upper triangle stored; = 'L': lower triangle stored
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in,out] A       Hermitian matrix A. On exit, destroyed.
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
 * @param[out]    isuppz  Support of eigenvectors in Z. Dimension (2*max(1,M)).
 * @param[out]    work    Complex workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    rwork   Double precision workspace. On exit, rwork[0] = optimal LRWORK.
 * @param[in]     lrwork  Length of rwork. If -1, workspace query.
 * @param[out]    iwork   Integer workspace. On exit, iwork[0] = optimal LIWORK.
 * @param[in]     liwork  Length of iwork. If -1, workspace query.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: internal error.
 */
void zheevr_2stage(
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    c128* restrict A,
    const int lda,
    const f64 vl,
    const f64 vu,
    const int il,
    const int iu,
    const f64 abstol,
    int* m,
    f64* restrict W,
    c128* restrict Z,
    const int ldz,
    int* restrict isuppz,
    c128* restrict work,
    const int lwork,
    f64* restrict rwork,
    const int lrwork,
    int* restrict iwork,
    const int liwork,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;

    int alleig, indeig, lower, lquery, valeig, wantz, tryrac;
    int i, ieeeok, iinfo, imax, indibl, indifl, indisp, indiwo;
    int indrd, indrdd, indre, indree, indrwk, indtau, indwk, indwkn;
    int iscale, itmp1, j, jj, liwmin, llwork, llrwork, llwrkn;
    int lrwmin, lwmin, nsplit;
    int lhtrd = 0, lwtrd, kd, ib, indhous;
    f64 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f64 sigma, smlnum, tmp1, vll = 0.0, vuu = 0.0;

    ieeeok = 1;

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    kd = ilaenv2stage(1, "ZHETRD_2STAGE", jobz, n, -1, -1, -1);
    ib = ilaenv2stage(2, "ZHETRD_2STAGE", jobz, n, kd, -1, -1);
    lhtrd = ilaenv2stage(3, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);
    lwtrd = ilaenv2stage(4, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);

    if (n <= 1) {
        lwmin = 1;
        lrwmin = 1;
        liwmin = 1;
    } else {
        lwmin = n + lhtrd + lwtrd;
        lrwmin = 24 * n;
        liwmin = 10 * n;
    }

    *info = 0;
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(alleig || valeig || indeig)) {
        *info = -2;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl)
                *info = -8;
        } else if (indeig) {
            if (il < 1 || il > (1 > n ? 1 : n)) {
                *info = -9;
            } else if (iu < (n < il ? n : il) || iu > n) {
                *info = -10;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n))
            *info = -15;
    }

    if (*info == 0) {
        work[0] = CMPLX((f64)lwmin, 0.0);
        rwork[0] = (f64)lrwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -18;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -20;
        } else if (liwork < liwmin && !lquery) {
            *info = -22;
        }
    }

    if (*info != 0) {
        xerbla("ZHEEVR_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0) {
        work[0] = CMPLX(1.0, 0.0);
        return;
    }

    if (n == 1) {
        work[0] = CMPLX(1.0, 0.0);
        if (alleig || indeig) {
            *m = 1;
            W[0] = creal(A[0]);
        } else {
            if (vl < creal(A[0]) && vu >= creal(A[0])) {
                *m = 1;
                W[0] = creal(A[0]);
            }
        }
        if (wantz) {
            Z[0] = CMPLX(1.0, 0.0);
            isuppz[0] = 1;
            isuppz[1] = 1;
        }
        return;
    }

    safmin = dlamch("S");
    eps = dlamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = (sqrt(bignum) < ONE / sqrt(sqrt(safmin))) ? sqrt(bignum) : ONE / sqrt(sqrt(safmin));

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    }
    anrm = zlanhe("M", uplo, n, A, lda, rwork);
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
                cblas_zdscal(n - j, sigma, &A[j + j * lda], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                cblas_zdscal(j + 1, sigma, &A[j * lda], 1);
            }
        }
        if (abstol > 0)
            abstll = abstol * sigma;
        if (valeig) {
            vll = vl * sigma;
            vuu = vu * sigma;
        }
    }

    indtau = 0;
    indhous = indtau + n;
    indwk = indhous + lhtrd;
    llwork = lwork - indwk;

    indrd = 0;
    indre = indrd + n;
    indrdd = indre + n;
    indree = indrdd + n;
    indrwk = indree + n;
    llrwork = lrwork - indrwk;

    indibl = 0;
    indisp = indibl + n;
    indifl = indisp + n;
    indiwo = indifl + n;

    zhetrd_2stage(jobz, uplo, n, A, lda, &rwork[indrd],
                  &rwork[indre], &work[indtau], &work[indhous],
                  lhtrd, &work[indwk], llwork, &iinfo);

    if ((alleig || (indeig && il == 1 && iu == n)) && ieeeok == 1) {
        if (!wantz) {
            cblas_dcopy(n, &rwork[indrd], 1, W, 1);
            cblas_dcopy(n - 1, &rwork[indre], 1, &rwork[indree], 1);
            dsterf(n, W, &rwork[indree], info);
        } else {
            cblas_dcopy(n - 1, &rwork[indre], 1, &rwork[indree], 1);
            cblas_dcopy(n, &rwork[indrd], 1, &rwork[indrdd], 1);

            if (abstol <= TWO * n * eps) {
                tryrac = 1;
            } else {
                tryrac = 0;
            }
            zstemr(jobz, "A", n, &rwork[indrdd], &rwork[indree],
                   vl, vu, il, iu, m, W, Z, ldz, n, isuppz,
                   &tryrac, &rwork[indrwk], llrwork,
                   iwork, liwork, info);

            if (wantz && *info == 0) {
                indwkn = indwk;
                llwrkn = lwork - indwkn;
                zunmtr("L", uplo, "N", n, *m, A, lda,
                       &work[indtau], Z, ldz, &work[indwkn],
                       llwrkn, &iinfo);
            }
        }

        if (*info == 0) {
            *m = n;
            goto L30;
        }
        *info = 0;
    }

    const char* order;
    if (wantz) {
        order = "B";
    } else {
        order = "E";
    }

    dstebz(range, order, n, vll, vuu, il, iu, abstll,
           &rwork[indrd], &rwork[indre], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwo], info);

    if (wantz) {
        zstein(n, &rwork[indrd], &rwork[indre], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &rwork[indrwk], &iwork[indiwo], &iwork[indifl], info);

        indwkn = indwk;
        llwrkn = lwork - indwkn;
        zunmtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

L30:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, W, 1);
    }

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
                itmp1 = iwork[indibl + i];
                W[i] = W[j];
                iwork[indibl + i] = iwork[indibl + j];
                W[j] = tmp1;
                iwork[indibl + j] = itmp1;
                cblas_zswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
            }
        }
    }

    work[0] = CMPLX((f64)lwmin, 0.0);
    rwork[0] = (f64)lrwmin;
    iwork[0] = liwmin;
}
