/**
 * @file ssyev.c
 * @brief SSYEV computes all eigenvalues and, optionally, eigenvectors of a
 *        real symmetric matrix using QR iteration.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "../include/lapack_tuning.h"

/**
 * SSYEV computes all eigenvalues and, optionally, eigenvectors of a
 * real symmetric matrix A.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only;
 *                         = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                         = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      On entry, the symmetric matrix A. If UPLO = 'U', the
 *                       leading N-by-N upper triangular part of A contains the
 *                       upper triangular part of the matrix A. If UPLO = 'L',
 *                       the leading N-by-N lower triangular part of A contains
 *                       the lower triangular part of the matrix A.
 *                       On exit, if JOBZ = 'V', then A contains the orthonormal
 *                       eigenvectors of the matrix A.
 *                       If JOBZ = 'N', then on exit the lower/upper triangle
 *                       of A, including the diagonal, is destroyed.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[out]    W      Array of dimension (n). The eigenvalues in ascending order.
 * @param[out]    work   Workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The dimension of work. lwork >= max(1, 3*n-1).
 *                       For optimal efficiency, lwork >= (NB+2)*N where NB is
 *                       the blocksize for SSYTRD.
 *                       If lwork = -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the algorithm failed to converge; i
 *                           off-diagonal elements did not converge to zero.
 */
void ssyev(const char* jobz, const char* uplo, const INT n,
           f32* restrict A, const INT lda,
           f32* restrict W,
           f32* restrict work, const INT lwork,
           INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT lower, wantz, lquery;
    INT iinfo, imax, inde, indtau, indwrk, iscale, llwork, lwkopt, nb;
    f32 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    /* Test the input parameters */
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1);

    *info = 0;
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(lower || uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    }

    if (*info == 0) {
        nb = lapack_get_nb("SYTRD");
        lwkopt = (n > 1) ? (nb + 2) * n : 1;
        work[0] = (f32)lwkopt;

        if (lwork < (3 * n - 1 > 1 ? 3 * n - 1 : 1) && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("SSYEV ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = A[0];
        work[0] = 2.0f;
        if (wantz) {
            A[0] = ONE;
        }
        return;
    }

    /* Get machine constants */
    safmin = slamch("S");
    eps = slamch("E");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    /* Scale matrix to allowable range, if necessary */
    anrm = slansy("M", uplo, n, A, lda, work);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        slascl(uplo, 0, 0, ONE, sigma, n, n, A, lda, info);
    }

    /* Call SSYTRD to reduce symmetric matrix to tridiagonal form */
    inde = 0;           /* E starts at work[0] */
    indtau = inde + n;  /* tau starts at work[n] */
    indwrk = indtau + n; /* remaining work at work[2*n] */
    llwork = lwork - indwrk;

    ssytrd(uplo, n, A, lda, W, &work[inde], &work[indtau],
           &work[indwrk], llwork, &iinfo);

    /* For eigenvalues only, call SSTERF. For eigenvectors, first call
     * SORGTR to generate the orthogonal matrix, then call SSTEQR. */
    if (!wantz) {
        ssterf(n, W, &work[inde], info);
    } else {
        sorgtr(uplo, n, A, lda, &work[indtau], &work[indwrk], llwork, &iinfo);
        ssteqr(jobz, n, W, &work[inde], A, lda, &work[indtau], info);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately */
    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    /* Set WORK(1) to optimal workspace size */
    work[0] = (f32)lwkopt;
}
