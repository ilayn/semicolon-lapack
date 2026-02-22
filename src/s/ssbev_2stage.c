/**
 * @file ssbev_2stage.c
 * @brief SSBEV_2STAGE computes all eigenvalues and optionally eigenvectors of a
 *        real symmetric band matrix using 2-stage reduction to tridiagonal.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_single.h"
#include <cblas.h>
#include <math.h>

/**
 * SSBEV_2STAGE computes all the eigenvalues and, optionally, eigenvectors of
 * a real symmetric band matrix A using the 2-stage technique for
 * the reduction to tridiagonal.
 *
 * @param[in]     jobz   = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     uplo   = 'U': upper triangle stored; = 'L': lower triangle stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     Number of super/sub-diagonals. kd >= 0.
 * @param[in,out] AB     Band matrix, overwritten on exit.
 * @param[in]     ldab   Leading dimension of AB. ldab >= kd+1.
 * @param[out]    W      Eigenvalues in ascending order.
 * @param[out]    Z      Eigenvectors if jobz='V'; not referenced if jobz='N'.
 * @param[in]     ldz    Leading dimension of Z. ldz >= 1, or ldz >= n if jobz='V'.
 * @param[out]    work   Workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork  Length of work. If -1, workspace query.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: convergence failure.
 */
void ssbev_2stage(
    const char* jobz,
    const char* uplo,
    const INT n,
    const INT kd,
    f32* restrict AB,
    const INT ldab,
    f32* restrict W,
    f32* restrict Z,
    const INT ldz,
    f32* restrict work,
    const INT lwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT lower, wantz, lquery;
    INT iinfo, imax, inde, indwrk, iscale;
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
            work[0] = (f32)lwmin;
        } else {
            ib = ilaenv2stage(2, "SSYTRD_SB2ST", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "SSYTRD_SB2ST", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "SSYTRD_SB2ST", jobz, n, kd, ib, -1);
            lwmin = n + lhtrd + lwtrd;
            work[0] = (f32)lwmin;
        }

        if (lwork < lwmin && !lquery) {
            *info = -11;
        }
    }

    if (*info != 0) {
        xerbla("SSBEV_2STAGE ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        if (lower) {
            W[0] = AB[0];
        } else {
            W[0] = AB[kd];
        }
        if (wantz) {
            Z[0] = ONE;
        }
        return;
    }

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    anrm = slansb("M", uplo, n, kd, AB, ldab, work);
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
            slascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            slascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
    }

    inde = 0;
    indhous = inde + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;

    ssytrd_sb2st("N", jobz, uplo, n, kd, AB, ldab, W,
                 &work[inde], &work[indhous], lhtrd,
                 &work[indwrk], llwork, &iinfo);

    if (!wantz) {
        ssterf(n, W, &work[inde], info);
    } else {
        ssteqr(jobz, n, W, &work[inde], Z, ldz, &work[indwrk], info);
    }

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    work[0] = (f32)lwmin;
}
