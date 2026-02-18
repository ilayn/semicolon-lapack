/**
 * @file zheevx_2stage.c
 * @brief ZHEEVX_2STAGE computes selected eigenvalues and optionally eigenvectors
 *        of a complex Hermitian matrix using 2-stage reduction.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHEEVX_2STAGE computes selected eigenvalues and, optionally, eigenvectors
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
 * @param[out]    work    Complex workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    rwork   Double precision workspace, dimension (7*n).
 * @param[out]    iwork   Integer workspace, dimension (5*n).
 * @param[out]    ifail   Indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: convergence failure.
 */
void zheevx_2stage(
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
    c128* restrict work,
    const int lwork,
    f64* restrict rwork,
    int* restrict iwork,
    int* restrict ifail,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);

    int alleig, indeig, lower, lquery, test, valeig, wantz;
    int i, iinfo, imax, indd, inde, indee, indibl;
    int indisp, indiwk, indtau, indwrk, indrwk, iscale;
    int itmp1, j, jj, llwork;
    int nsplit, lwmin, lhtrd = 0, lwtrd, kd, ib, indhous;
    f64 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f64 sigma, smlnum, tmp1, vll, vuu;

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
            if (n > 0 && vu <= vl)
                *info = -8;
        } else if (indeig) {
            if (il < 0 || il > (0 > n - 1 ? 0 : n - 1)) {
                *info = -9;
            } else if (iu < ((n - 1) < il ? (n - 1) : il) || iu > n - 1) {
                *info = -10;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n))
            *info = -15;
    }

    if (*info == 0) {
        if (n <= 1) {
            lwmin = 1;
            work[0] = CMPLX((f64)lwmin, 0.0);
        } else {
            kd = ilaenv2stage(1, "ZHETRD_2STAGE", jobz, n, -1, -1, -1);
            ib = ilaenv2stage(2, "ZHETRD_2STAGE", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);
            lwmin = n + lhtrd + lwtrd;
            work[0] = CMPLX((f64)lwmin, 0.0);
        }

        if (lwork < lwmin && !lquery)
            *info = -17;
    }

    if (*info != 0) {
        xerbla("ZHEEVX_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0)
        return;

    if (n == 1) {
        if (alleig || indeig) {
            *m = 1;
            W[0] = creal(A[0]);
        } else if (valeig) {
            if (vl < creal(A[0]) && vu >= creal(A[0])) {
                *m = 1;
                W[0] = creal(A[0]);
            }
        }
        if (wantz)
            Z[0] = CONE;
        return;
    }

    safmin = dlamch("S");
    eps = dlamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);
    if (rmax > ONE / sqrt(sqrt(safmin)))
        rmax = ONE / sqrt(sqrt(safmin));

    iscale = 0;
    abstll = abstol;
    if (valeig) {
        vll = vl;
        vuu = vu;
    } else {
        vll = ZERO;
        vuu = ZERO;
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

    indd = 0;
    inde = indd + n;
    indrwk = inde + n;
    indtau = 0;
    indhous = indtau + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    zhetrd_2stage(jobz, uplo, n, A, lda, &rwork[indd],
                  &rwork[inde], &work[indtau], &work[indhous],
                  lhtrd, &work[indwrk], llwork, &iinfo);

    test = 0;
    if (indeig) {
        if (il == 0 && iu == n - 1) {
            test = 1;
        }
    }
    if ((alleig || test) && abstol <= ZERO) {
        cblas_dcopy(n, &rwork[indd], 1, W, 1);
        indee = indrwk + 2 * n;
        if (!wantz) {
            cblas_dcopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            dsterf(n, W, &rwork[indee], info);
        } else {
            zlacpy("A", n, n, A, lda, Z, ldz);
            zungtr(uplo, n, Z, ldz, &work[indtau],
                   &work[indwrk], llwork, &iinfo);
            cblas_dcopy(n - 1, &rwork[inde], 1, &rwork[indee], 1);
            zsteqr(jobz, n, W, &rwork[indee], Z, ldz,
                   &rwork[indrwk], info);
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
    indiwk = indisp + n;
    dstebz(range, wantz ? "B" : "E", n, vll, vuu, il, iu, abstll,
           &rwork[indd], &rwork[inde], m, &nsplit, W,
           &iwork[indibl], &iwork[indisp], &rwork[indrwk],
           &iwork[indiwk], info);

    if (wantz) {
        zstein(n, &rwork[indd], &rwork[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &rwork[indrwk], &iwork[indiwk], ifail, info);

        zunmtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwrk], llwork, &iinfo);
    }

L40:
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
                if (*info != 0) {
                    itmp1 = ifail[i];
                    ifail[i] = ifail[j];
                    ifail[j] = itmp1;
                }
            }
        }
    }

    work[0] = CMPLX((f64)lwmin, 0.0);
}
