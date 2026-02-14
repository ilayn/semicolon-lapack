/**
 * @file zheev_2stage.c
 * @brief ZHEEV_2STAGE computes eigenvalues and optionally eigenvectors using 2-stage reduction.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

void zheev_2stage(const char* jobz, const char* uplo, const int n,
                  double complex* A, const int lda,
                  double* W,
                  double complex* work, const int lwork,
                  double* rwork, int* info)
{
    const double zero = 0.0;
    const double one = 1.0;
    const double complex cone = CMPLX(1.0, 0.0);

    int lower, lquery, wantz;
    int iinfo, imax, inde, indtau, indwrk, iscale;
    int llwork, lwmin, lhtrd, lwtrd, kd, ib, indhous;
    double anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

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
        kd = ilaenv2stage(1, "ZHETRD_2STAGE", jobz, n, -1, -1, -1);
        ib = ilaenv2stage(2, "ZHETRD_2STAGE", jobz, n, kd, -1, -1);
        lhtrd = ilaenv2stage(3, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);
        lwtrd = ilaenv2stage(4, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);
        lwmin = n + lhtrd + lwtrd;
        work[0] = CMPLX((double)lwmin, 0.0);

        if (lwork < lwmin && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("ZHEEV_2STAGE ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = creal(A[0]);
        work[0] = CMPLX(1.0, 0.0);
        if (wantz) {
            A[0] = cone;
        }
        return;
    }

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

    anrm = zlanhe("M", uplo, n, A, lda, rwork);
    iscale = 0;
    if (anrm > zero && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        zlascl(uplo, 0, 0, one, sigma, n, n, A, lda, info);
    }

    inde = 0;
    indtau = 0;
    indhous = indtau + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    zhetrd_2stage(jobz, uplo, n, A, lda, W, &rwork[inde],
                  &work[indtau], &work[indhous], lhtrd,
                  &work[indwrk], llwork, &iinfo);

    if (!wantz) {
        dsterf(n, W, &rwork[inde], info);
    } else {
        zungtr(uplo, n, A, lda, &work[indtau],
               &work[indwrk], llwork, &iinfo);
        indwrk = inde + n;
        zsteqr(jobz, n, W, &rwork[inde], A, lda,
               &rwork[indwrk], info);
    }

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, one / sigma, W, 1);
    }

    work[0] = CMPLX((double)lwmin, 0.0);
}
