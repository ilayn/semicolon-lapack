/**
 * @file cheev_2stage.c
 * @brief CHEEV_2STAGE computes eigenvalues and optionally eigenvectors using 2-stage reduction.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

void cheev_2stage(const char* jobz, const char* uplo, const INT n,
                  c64* A, const INT lda,
                  f32* W,
                  c64* work, const INT lwork,
                  f32* rwork, INT* info)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;
    const c64 cone = CMPLXF(1.0f, 0.0f);

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
        kd = ilaenv2stage(1, "CHETRD_2STAGE", jobz, n, -1, -1, -1);
        ib = ilaenv2stage(2, "CHETRD_2STAGE", jobz, n, kd, -1, -1);
        lhtrd = ilaenv2stage(3, "CHETRD_2STAGE", jobz, n, kd, ib, -1);
        lwtrd = ilaenv2stage(4, "CHETRD_2STAGE", jobz, n, kd, ib, -1);
        lwmin = n + lhtrd + lwtrd;
        work[0] = CMPLXF((f32)lwmin, 0.0f);

        if (lwork < lwmin && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("CHEEV_2STAGE ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = crealf(A[0]);
        work[0] = CMPLXF(1.0f, 0.0f);
        if (wantz) {
            A[0] = cone;
        }
        return;
    }

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    anrm = clanhe("M", uplo, n, A, lda, rwork);
    iscale = 0;
    if (anrm > zero && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        clascl(uplo, 0, 0, one, sigma, n, n, A, lda, info);
    }

    inde = 0;
    indtau = 0;
    indhous = indtau + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    chetrd_2stage(jobz, uplo, n, A, lda, W, &rwork[inde],
                  &work[indtau], &work[indhous], lhtrd,
                  &work[indwrk], llwork, &iinfo);

    if (!wantz) {
        ssterf(n, W, &rwork[inde], info);
    } else {
        cungtr(uplo, n, A, lda, &work[indtau],
               &work[indwrk], llwork, &iinfo);
        indwrk = inde + n;
        csteqr(jobz, n, W, &rwork[inde], A, lda,
               &rwork[indwrk], info);
    }

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, one / sigma, W, 1);
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
