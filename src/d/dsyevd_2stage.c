/**
 * @file dsyevd_2stage.c
 * @brief DSYEVD_2STAGE computes eigenvalues and optionally eigenvectors using divide-and-conquer.
 */

#include "semicolon_lapack_double.h"
#include "semicolon_cblas.h"
#include <math.h>

void dsyevd_2stage(const char* jobz, const char* uplo, const INT n,
                   f64* A, const INT lda,
                   f64* W,
                   f64* work, const INT lwork,
                   INT* iwork, const INT liwork, INT* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;

    INT lower, lquery, wantz;
    INT iinfo, inde, indtau, indwrk, iscale;
    INT liwmin, llwork, lwmin;
    /* INT indwk2, llwrk2; - used only in eigenvector path (disabled) */
    INT lhtrd = 0, lwtrd, kd, ib, indhous;
    f64 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

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
            kd = ilaenv2stage(1, "DSYTRD_2STAGE", jobz, n, -1, -1, -1);
            ib = ilaenv2stage(2, "DSYTRD_2STAGE", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "DSYTRD_2STAGE", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "DSYTRD_2STAGE", jobz, n, kd, ib, -1);
            if (wantz) {
                liwmin = 3 + 5 * n;
                lwmin = 1 + 6 * n + 2 * n * n;
            } else {
                liwmin = 1;
                lwmin = 2 * n + 1 + lhtrd + lwtrd;
            }
        }
        work[0] = (f64)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (liwork < liwmin && !lquery) {
            *info = -10;
        }
    }

    if (*info != 0) {
        xerbla("DSYEVD_2STAGE", -(*info));
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

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

    anrm = dlansy("M", uplo, n, A, lda, work);
    iscale = 0;
    if (anrm > zero && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        dlascl(uplo, 0, 0, one, sigma, n, n, A, lda, info);
    }

    inde = 0;
    indtau = inde + n;
    indhous = indtau + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;
    /* indwk2 = indwrk + n * n; - used only in eigenvector path (disabled) */
    /* llwrk2 = lwork - indwk2; - used only in eigenvector path (disabled) */

    dsytrd_2stage(jobz, uplo, n, A, lda, W, &work[inde],
                  &work[indtau], &work[indhous], lhtrd,
                  &work[indwrk], llwork, &iinfo);

    if (!wantz) {
        dsterf(n, W, &work[inde], info);
    } else {
        /* Eigenvector computation not available in 2-stage algorithm;
           argument checking should prevent reaching here */
        return;
        /* TODO: Enable when eigenvector support is added
        dstedc("I", n, W, &work[inde], &work[indwrk], n,
               &work[indwk2], llwrk2, iwork, liwork, info);
        dormtr("L", uplo, "N", n, n, A, lda, &work[indtau],
               &work[indwrk], n, &work[indwk2], llwrk2, &iinfo);
        dlacpy("A", n, n, &work[indwrk], n, A, lda);
        */
    }

    if (iscale == 1) {
        cblas_dscal(n, one / sigma, W, 1);
    }

    work[0] = (f64)lwmin;
    iwork[0] = liwmin;
}
