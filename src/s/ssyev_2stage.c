/**
 * @file ssyev_2stage.c
 * @brief SSYEV_2STAGE computes eigenvalues and optionally eigenvectors using 2-stage reduction.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include <math.h>

void ssyev_2stage(const char* jobz, const char* uplo, const INT n,
                  f32* A, const INT lda,
                  f32* W,
                  f32* work, const INT lwork, INT* info)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;

    INT lower, lquery, wantz;
    INT iinfo, imax, inde, indtau, indwrk, iscale;
    INT llwork, lwmin, lhtrd, lwtrd, kd, ib, indhous;
    f32 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1);

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
        kd = ilaenv2stage(1, "SSYTRD_2STAGE", jobz, n, -1, -1, -1);
        ib = ilaenv2stage(2, "SSYTRD_2STAGE", jobz, n, kd, -1, -1);
        lhtrd = ilaenv2stage(3, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
        lwtrd = ilaenv2stage(4, "SSYTRD_2STAGE", jobz, n, kd, ib, -1);
        lwmin = 2 * n + lhtrd + lwtrd;
        work[0] = (f32)lwmin;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("SSYEV_2STAGE ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = A[0];
        work[0] = 2.0f;
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
        sorgtr(uplo, n, A, lda, &work[indtau],
               &work[indwrk], llwork, &iinfo);
        ssteqr(jobz, n, W, &work[inde], A, lda,
               &work[indtau], info);
        */
    }

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, one / sigma, W, 1);
    }

    work[0] = (f32)lwmin;
}
