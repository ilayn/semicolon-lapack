/**
 * @file ssyevd_2stage.c
 * @brief SSYEVD_2STAGE computes eigenvalues and optionally eigenvectors using divide-and-conquer.
 */

#include "semicolon_lapack_single.h"
#include <cblas.h>
#include <math.h>

void ssyevd_2stage(const char* jobz, const char* uplo, const int n,
                   f32* A, const int lda,
                   f32* W,
                   f32* work, const int lwork,
                   int* iwork, const int liwork, int* info)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;

    int lower, lquery, wantz;
    int iinfo, inde, indtau, indwrk, iscale;
    int liwmin, llwork, lwmin;
    /* int indwk2, llwrk2; - used only in eigenvector path (disabled) */
    int lhtrd = 0, lwtrd, kd, ib, indhous;
    f32 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1) || (liwork == -1);

    *info = 0;
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < ((1 > n) ? 1 : n)) {
        *info = -5;
    }

    if (*info == 0) {
        if (n <= 1) {
            liwmin = 1;
            lwmin = 1;
        } else {
            kd = ilaenv2stage(1, "SSYTRD_2STAGE", jobz, n, -1, -1, -1);
            ib = ilaenv2stage(2, "SSYTRD_2STAGE", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
            if (wantz) {
                liwmin = 3 + 5 * n;
                lwmin = 1 + 6 * n + 2 * n * n;
            } else {
                liwmin = 1;
                lwmin = 2 * n + 1 + lhtrd + lwtrd;
            }
        }
        work[0] = (f32)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (liwork < liwmin && !lquery) {
            *info = -10;
        }
    }

    if (*info != 0) {
        xerbla("SSYEVD_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = A[0];
        if (wantz) {
            A[0] = one;
        }
        return;
    }

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    anrm = slansy("M", uplo, n, A, lda, work);
    iscale = 0;
    if (anrm > zero && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        slascl(uplo, 0, 0, one, sigma, n, n, A, lda, info);
    }

    inde = 0;
    indtau = inde + n;
    indhous = indtau + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;
    /* indwk2 = indwrk + n * n; - used only in eigenvector path (disabled) */
    /* llwrk2 = lwork - indwk2; - used only in eigenvector path (disabled) */

    ssytrd_2stage(jobz, uplo, n, A, lda, W, &work[inde],
                  &work[indtau], &work[indhous], lhtrd,
                  &work[indwrk], llwork, &iinfo);

    if (!wantz) {
        ssterf(n, W, &work[inde], info);
    } else {
        /* Eigenvector computation not available in 2-stage algorithm;
           argument checking should prevent reaching here */
        return;
        /* TODO: Enable when eigenvector support is added
        sstedc("I", n, W, &work[inde], &work[indwrk], n,
               &work[indwk2], llwrk2, iwork, liwork, info);
        sormtr("L", uplo, "N", n, n, A, lda, &work[indtau],
               &work[indwrk], n, &work[indwk2], llwrk2, &iinfo);
        slacpy("A", n, n, &work[indwrk], n, A, lda);
        */
    }

    if (iscale == 1) {
        cblas_sscal(n, one / sigma, W, 1);
    }

    work[0] = (f32)lwmin;
    iwork[0] = liwmin;
}
