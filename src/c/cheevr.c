/**
 * @file cheevr.c
 * @brief CHEEVR computes selected eigenvalues and optionally eigenvectors
 *        of a complex Hermitian matrix using the RRR algorithm.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"
#include "../include/lapack_tuning.h"

/**
 * CHEEVR computes selected eigenvalues and, optionally, eigenvectors
 * of a complex Hermitian matrix A. Eigenvalues and eigenvectors can be
 * selected by specifying either a range of values or a range of indices
 * for the desired eigenvalues.
 *
 * CHEEVR first reduces the matrix A to tridiagonal form T with a call
 * to CHETRD. Then, whenever possible, CHEEVR calls CSTEMR to compute
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
 * @param[in,out] A       On entry, the Hermitian matrix A.
 *                        On exit, the lower triangle (if uplo='L') or
 *                        the upper triangle (if uplo='U') is destroyed.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in]     vl      If RANGE='V', lower bound of interval.
 * @param[in]     vu      If RANGE='V', upper bound of interval. VL < VU.
 * @param[in]     il      If RANGE='I', index of smallest eigenvalue.
 * @param[in]     iu      If RANGE='I', index of largest eigenvalue.
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       The total number of eigenvalues found.
 * @param[out]    W       The first M elements contain the selected eigenvalues
 *                        in ascending order.
 * @param[out]    Z       If JOBZ = 'V', the first M columns contain the orthonormal
 *                        eigenvectors. If JOBZ = 'N', Z is not referenced.
 * @param[in]     ldz     Leading dimension of Z. LDZ >= 1, and if JOBZ = 'V', LDZ >= N.
 * @param[out]    isuppz  Support of eigenvectors in Z. Dimension (2*max(1,M)).
 * @param[out]    work    Complex workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    rwork   Single precision workspace. On exit, rwork[0] = optimal LRWORK.
 * @param[in]     lrwork  Length of rwork. If -1, workspace query.
 * @param[out]    iwork   Integer workspace. On exit, iwork[0] = optimal LIWORK.
 * @param[in]     liwork  Length of iwork. If -1, workspace query.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: Internal error
 */
void cheevr(
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    c64* restrict A,
    const int lda,
    const f32 vl,
    const f32 vu,
    const int il,
    const int iu,
    const f32 abstol,
    int* m,
    f32* restrict W,
    c64* restrict Z,
    const int ldz,
    int* restrict isuppz,
    c64* restrict work,
    const int lwork,
    f32* restrict rwork,
    const int lrwork,
    int* restrict iwork,
    const int liwork,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;

    int alleig, indeig, lower, lquery, valeig, wantz, tryrac;
    int i, ieeeok, iinfo, imax, indibl, indifl, indisp, indiwo;
    int indrd, indrdd, indre, indree, indrwk, indtau, indwk, indwkn;
    int iscale, itmp1, j, jj, liwmin, llwork, llrwork, llwrkn;
    int lrwmin, lwkopt, lwmin, nb, nsplit;
    f32 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f32 sigma, smlnum, tmp1, vll = 0.0f, vuu = 0.0f;

    ieeeok = 1;

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    if (n <= 1) {
        lwmin = 1;
        lrwmin = 1;
        liwmin = 1;
    } else {
        lwmin = 2 * n;
        lrwmin = 24 * n;
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
        nb = lapack_get_nb("HETRD");
        nb = nb > lapack_get_nb("UNMTR") ? nb : lapack_get_nb("UNMTR");
        lwkopt = ((nb + 1) * n > lwmin) ? (nb + 1) * n : lwmin;
        work[0] = CMPLXF((f32)lwkopt, 0.0f);
        rwork[0] = (f32)lrwmin;
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
        xerbla("CHEEVR", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0) {
        work[0] = CMPLXF(1.0f, 0.0f);
        return;
    }

    if (n == 1) {
        work[0] = CMPLXF(1.0f, 0.0f);
        if (alleig || indeig) {
            *m = 1;
            W[0] = crealf(A[0]);
        } else {
            if (vl < crealf(A[0]) && vu >= crealf(A[0])) {
                *m = 1;
                W[0] = crealf(A[0]);
            }
        }
        if (wantz) {
            Z[0] = CMPLXF(1.0f, 0.0f);
            isuppz[0] = 1;
            isuppz[1] = 1;
        }
        return;
    }

    safmin = slamch("S");
    eps = slamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);
    if (rmax > ONE / sqrtf(sqrtf(safmin)))
        rmax = ONE / sqrtf(sqrtf(safmin));

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    }
    anrm = clanhe("M", uplo, n, A, lda, rwork);
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
                cblas_csscal(n - j, sigma, &A[j + j * lda], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                cblas_csscal(j + 1, sigma, &A[j * lda], 1);
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
    indwk = indtau + n;
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

    chetrd(uplo, n, A, lda, &rwork[indrd], &rwork[indre],
           &work[indtau], &work[indwk], llwork, &iinfo);

    if ((alleig || (indeig && il == 1 && iu == n)) && ieeeok == 1) {
        if (!wantz) {
            cblas_scopy(n, &rwork[indrd], 1, W, 1);
            cblas_scopy(n - 1, &rwork[indre], 1, &rwork[indree], 1);
            ssterf(n, W, &rwork[indree], info);
        } else {
            cblas_scopy(n - 1, &rwork[indre], 1, &rwork[indree], 1);
            cblas_scopy(n, &rwork[indrd], 1, &rwork[indrdd], 1);

            if (abstol <= TWO * n * eps) {
                tryrac = 1;
            } else {
                tryrac = 0;
            }
            cstemr(jobz, "A", n, &rwork[indrdd], &rwork[indree],
                   vl, vu, il, iu, m, W, Z, ldz, n, isuppz,
                   &tryrac, &rwork[indrwk], llrwork,
                   iwork, liwork, info);

            if (wantz && *info == 0) {
                indwkn = indwk;
                llwrkn = lwork - indwkn;
                cunmtr("L", uplo, "N", n, *m, A, lda,
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

    sstebz(range, order, n, vll, vuu, il, iu, abstll,
           &rwork[indrd], &rwork[indre], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwo], info);

    if (wantz) {
        cstein(n, &rwork[indrd], &rwork[indre], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &rwork[indrwk], &iwork[indiwo], &iwork[indifl], info);

        indwkn = indwk;
        llwrkn = lwork - indwkn;
        cunmtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

L30:
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
                cblas_cswap(n, &Z[i * ldz], 1, &Z[j * ldz], 1);
            }
        }
    }

    work[0] = CMPLXF((f32)lwkopt, 0.0f);
    rwork[0] = (f32)lrwmin;
    iwork[0] = liwmin;
}
