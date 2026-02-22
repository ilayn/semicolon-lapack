/**
 * @file chbev_2stage.c
 * @brief CHBEV_2STAGE computes all eigenvalues and optionally eigenvectors of a
 *        complex Hermitian band matrix using 2-stage reduction to tridiagonal.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

/**
 * CHBEV_2STAGE computes all the eigenvalues and, optionally, eigenvectors of
 * a complex Hermitian band matrix A using the 2stage technique for
 * the reduction to tridiagonal.
 *
 * @param[in]     jobz   = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     uplo   = 'U': upper triangle stored; = 'L': lower triangle stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     Number of super/sub-diagonals. kd >= 0.
 * @param[in,out] AB     Complex*16 array, dimension (ldab, n). Band matrix,
 *                       overwritten on exit.
 * @param[in]     ldab   Leading dimension of AB. ldab >= kd+1.
 * @param[out]    W      Single precision array, dimension (n). Eigenvalues in
 *                       ascending order.
 * @param[out]    Z      Complex*16 array, dimension (ldz, n). Eigenvectors if
 *                       jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz    Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work   Complex*16 workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork  Length of work. If -1, workspace query.
 * @param[out]    rwork  Single precision array, dimension (max(1, 3*n-2)).
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: convergence failure.
 */
void chbev_2stage(
    const char* jobz,
    const char* uplo,
    const INT n,
    const INT kd,
    c64* restrict AB,
    const INT ldab,
    f32* restrict W,
    c64* restrict Z,
    const INT ldz,
    c64* restrict work,
    const INT lwork,
    f32* restrict rwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT lower, wantz, lquery;
    INT iinfo, imax, inde, indwrk, indrwk, iscale;
    INT llwork, lwmin, lhtrd = 0, lwtrd, ib, indhous;
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
    } else if (kd < 0) {
        *info = -4;
    } else if (ldab < kd + 1) {
        *info = -6;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -9;
    }

    if (*info == 0) {
        if (n <= 1) {
            lwmin = 1;
            work[0] = CMPLXF((f32)lwmin, 0.0f);
        } else {
            ib = ilaenv2stage(2, "CHETRD_HB2ST", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "CHETRD_HB2ST", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "CHETRD_HB2ST", jobz, n, kd, ib, -1);
            lwmin = lhtrd + lwtrd;
            work[0] = CMPLXF((f32)lwmin, 0.0f);
        }

        if (lwork < lwmin && !lquery) {
            *info = -11;
        }
    }

    if (*info != 0) {
        xerbla("CHBEV_2STAGE ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        if (lower) {
            W[0] = crealf(AB[0]);
        } else {
            W[0] = crealf(AB[kd]);
        }
        if (wantz) {
            Z[0] = CMPLXF(ONE, 0.0f);
        }
        return;
    }

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    anrm = clanhb("M", uplo, n, kd, AB, ldab, rwork);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        if (lower) {
            clascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            clascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
    }

    inde = 0;
    indhous = 0;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    chetrd_hb2st("N", jobz, uplo, n, kd, AB, ldab, W,
                 &rwork[inde], &work[indhous], lhtrd,
                 &work[indwrk], llwork, &iinfo);

    if (!wantz) {
        ssterf(n, W, &rwork[inde], info);
    } else {
        indrwk = inde + n;
        csteqr(jobz, n, W, &rwork[inde], Z, ldz,
               &rwork[indrwk], info);
    }

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
