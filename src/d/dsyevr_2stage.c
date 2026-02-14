/**
 * @file dsyevr_2stage.c
 * @brief DSYEVR_2STAGE computes selected eigenvalues and optionally eigenvectors using RRR.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>
#include <math.h>

void dsyevr_2stage(const char* jobz, const char* range, const char* uplo,
                   const int n, f64* A, const int lda,
                   const f64 vl, const f64 vu,
                   const int il, const int iu,
                   const f64 abstol, int* m,
                   f64* W, f64* Z, const int ldz, int* isuppz,
                   f64* work, const int lwork,
                   int* iwork, const int liwork, int* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const f64 two = 2.0;

    int alleig, indeig, lower, lquery, valeig, wantz, tryrac;
    char order;
    int i, ieeeok, iinfo, imax, indd, inddd, inde;
    int indee, indibl, indifl, indisp, indiwo, indtau;
    int indwk, indwkn, iscale, j, jj, liwmin;
    int llwork, llwrkn, lwmin, nsplit;
    int lhtrd, lwtrd, kd, ib, indhous;
    f64 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f64 sigma, smlnum, tmp1, vll, vuu;

    /* IEEEOK = 1 means IEEE arithmetic is assumed (NaN/Inf handled properly) */
    ieeeok = 1;

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    lquery = (lwork == -1) || (liwork == -1);

    kd = ilaenv2stage(1, "DSYTRD_2STAGE", jobz, n, -1, -1, -1);
    ib = ilaenv2stage(2, "DSYTRD_2STAGE", jobz, n, kd, -1, -1);
    lhtrd = ilaenv2stage(3, "DSYTRD_2STAGE", jobz, n, kd, ib, -1);
    lwtrd = ilaenv2stage(4, "DSYTRD_2STAGE", jobz, n, kd, ib, -1);

    if (n <= 1) {
        lwmin = 1;
        liwmin = 1;
    } else {
        lwmin = (26 * n > 5 * n + lhtrd + lwtrd) ? 26 * n : 5 * n + lhtrd + lwtrd;
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
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -6;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -8;
            }
        } else if (indeig) {
            if (il < 1 || il > ((1 > n) ? 1 : n)) {
                *info = -9;
            } else if (iu < ((n < il) ? n : il) || iu > n) {
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
        work[0] = (f64)lwmin;
        iwork[0] = liwmin;
    }

    if (*info != 0) {
        xerbla("DSYEVR_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0) {
        work[0] = 1.0;
        return;
    }

    if (n == 1) {
        work[0] = 1.0;
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
            Z[0] = one;
            isuppz[0] = 1;
            isuppz[1] = 1;
        }
        return;
    }

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrt(smlnum);
    rmax = (sqrt(bignum) < one / sqrt(sqrt(safmin))) ? sqrt(bignum) : one / sqrt(sqrt(safmin));

    iscale = 0;
    abstll = abstol;
    vll = vl;
    vuu = vu;
    anrm = dlansy("M", uplo, n, A, lda, work);
    if (anrm > zero && anrm < rmin) {
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

    indtau = 0;
    indd = indtau + n;
    inde = indd + n;
    inddd = inde + n;
    indee = inddd + n;
    indhous = indee + n;
    indwk = indhous + lhtrd;
    llwork = lwork - indwk;

    indibl = 0;
    indisp = indibl + n;
    indifl = indisp + n;
    indiwo = indifl + n;

    dsytrd_2stage(jobz, uplo, n, A, lda, &work[indd],
                  &work[inde], &work[indtau], &work[indhous],
                  lhtrd, &work[indwk], llwork, &iinfo);

    if ((alleig || (indeig && il == 1 && iu == n)) && ieeeok == 1) {
        if (!wantz) {
            cblas_dcopy(n, &work[indd], 1, W, 1);
            cblas_dcopy(n - 1, &work[inde], 1, &work[indee], 1);
            dsterf(n, W, &work[indee], info);
        } else {
            cblas_dcopy(n - 1, &work[inde], 1, &work[indee], 1);
            cblas_dcopy(n, &work[indd], 1, &work[inddd], 1);

            if (abstol <= two * n * eps) {
                tryrac = 1;
            } else {
                tryrac = 0;
            }
            dstemr(jobz, "A", n, &work[inddd], &work[indee],
                   vl, vu, il, iu, m, W, Z, ldz, n, isuppz,
                   &tryrac, &work[indwk], lwork, iwork, liwork, info);

            if (wantz && *info == 0) {
                indwkn = inde;
                llwrkn = lwork - indwkn;
                dormtr("L", uplo, "N", n, *m, A, lda,
                       &work[indtau], Z, ldz, &work[indwkn],
                       llwrkn, &iinfo);
            }
        }

        if (*info == 0) {
            *m = n;
            goto label30;
        }
        *info = 0;
    }

    if (wantz) {
        order = 'B';
    } else {
        order = 'E';
    }

    {
        char order_str[2] = {order, '\0'};
        dstebz(range, order_str, n, vll, vuu, il, iu, abstll,
               &work[indd], &work[inde], m, &nsplit, W,
               &iwork[indibl], &iwork[indisp], &work[indwk],
               &iwork[indiwo], info);
    }

    if (wantz) {
        dstein(n, &work[indd], &work[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &work[indwk], &iwork[indiwo], &iwork[indifl], info);

        indwkn = inde;
        llwrkn = lwork - indwkn;
        dormtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

label30:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, one / sigma, W, 1);
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
                W[i] = W[j];
                W[j] = tmp1;
                cblas_dswap(n, &Z[0 + i * ldz], 1, &Z[0 + j * ldz], 1);
            }
        }
    }

    work[0] = (f64)lwmin;
    iwork[0] = liwmin;
}
