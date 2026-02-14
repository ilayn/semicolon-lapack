/**
 * @file ssyevr_2stage.c
 * @brief SSYEVR_2STAGE computes selected eigenvalues and optionally eigenvectors using RRR.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>
#include <math.h>

void ssyevr_2stage(const char* jobz, const char* range, const char* uplo,
                   const int n, f32* A, const int lda,
                   const f32 vl, const f32 vu,
                   const int il, const int iu,
                   const f32 abstol, int* m,
                   f32* W, f32* Z, const int ldz, int* isuppz,
                   f32* work, const int lwork,
                   int* iwork, const int liwork, int* info)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;
    const f32 two = 2.0f;

    int alleig, indeig, lower, lquery, valeig, wantz, tryrac;
    char order;
    int i, ieeeok, iinfo, imax, indd, inddd, inde;
    int indee, indibl, indifl, indisp, indiwo, indtau;
    int indwk, indwkn, iscale, j, jj, liwmin;
    int llwork, llwrkn, lwmin, nsplit;
    int lhtrd, lwtrd, kd, ib, indhous;
    f32 abstll, anrm, bignum, eps, rmax, rmin, safmin;
    f32 sigma, smlnum, tmp1, vll, vuu;

    /* IEEEOK = 1 means IEEE arithmetic is assumed (NaN/Inf handled properly) */
    ieeeok = 1;

    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');

    lquery = (lwork == -1) || (liwork == -1);

    kd = ilaenv2stage(1, "SSYTRD_2STAGE", jobz, n, -1, -1, -1);
    ib = ilaenv2stage(2, "SSYTRD_2STAGE", jobz, n, kd, -1, -1);
    lhtrd = ilaenv2stage(3, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
    lwtrd = ilaenv2stage(4, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);

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
        work[0] = (f32)lwmin;
        iwork[0] = liwmin;
    }

    if (*info != 0) {
        xerbla("SSYEVR_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

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
            Z[0] = one;
            isuppz[0] = 1;
            isuppz[1] = 1;
        }
        return;
    }

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrtf(smlnum);
    rmax = (sqrtf(bignum) < one / sqrtf(sqrtf(safmin))) ? sqrtf(bignum) : one / sqrtf(sqrtf(safmin));

    iscale = 0;
    abstll = abstol;
    vll = vl;
    vuu = vu;
    anrm = slansy("M", uplo, n, A, lda, work);
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

    ssytrd_2stage(jobz, uplo, n, A, lda, &work[indd],
                  &work[inde], &work[indtau], &work[indhous],
                  lhtrd, &work[indwk], llwork, &iinfo);

    if ((alleig || (indeig && il == 1 && iu == n)) && ieeeok == 1) {
        if (!wantz) {
            cblas_scopy(n, &work[indd], 1, W, 1);
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            ssterf(n, W, &work[indee], info);
        } else {
            cblas_scopy(n - 1, &work[inde], 1, &work[indee], 1);
            cblas_scopy(n, &work[indd], 1, &work[inddd], 1);

            if (abstol <= two * n * eps) {
                tryrac = 1;
            } else {
                tryrac = 0;
            }
            sstemr(jobz, "A", n, &work[inddd], &work[indee],
                   vl, vu, il, iu, m, W, Z, ldz, n, isuppz,
                   &tryrac, &work[indwk], lwork, iwork, liwork, info);

            if (wantz && *info == 0) {
                indwkn = inde;
                llwrkn = lwork - indwkn;
                sormtr("L", uplo, "N", n, *m, A, lda,
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
        sstebz(range, order_str, n, vll, vuu, il, iu, abstll,
               &work[indd], &work[inde], m, &nsplit, W,
               &iwork[indibl], &iwork[indisp], &work[indwk],
               &iwork[indiwo], info);
    }

    if (wantz) {
        sstein(n, &work[indd], &work[inde], *m, W,
               &iwork[indibl], &iwork[indisp], Z, ldz,
               &work[indwk], &iwork[indiwo], &iwork[indifl], info);

        indwkn = inde;
        llwrkn = lwork - indwkn;
        sormtr("L", uplo, "N", n, *m, A, lda, &work[indtau],
               Z, ldz, &work[indwkn], llwrkn, &iinfo);
    }

label30:
    if (iscale == 1) {
        if (*info == 0) {
            imax = *m;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, one / sigma, W, 1);
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
                cblas_sswap(n, &Z[0 + i * ldz], 1, &Z[0 + j * ldz], 1);
            }
        }
    }

    work[0] = (f32)lwmin;
    iwork[0] = liwmin;
}
