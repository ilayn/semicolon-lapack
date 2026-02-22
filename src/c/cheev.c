/**
 * @file cheev.c
 * @brief CHEEV computes all eigenvalues and, optionally, eigenvectors of a
 *        complex Hermitian matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"
#include "../include/lapack_tuning.h"

/**
 * CHEEV computes all eigenvalues and, optionally, eigenvectors of a
 * complex Hermitian matrix A.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only;
 *                         = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                         = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] A      On entry, the Hermitian matrix A. If UPLO = 'U', the
 *                       leading N-by-N upper triangular part of A contains the
 *                       upper triangular part of the matrix A. If UPLO = 'L',
 *                       the leading N-by-N lower triangular part of A contains
 *                       the lower triangular part of the matrix A.
 *                       On exit, if JOBZ = 'V', then if INFO = 0, A contains
 *                       the orthonormal eigenvectors of the matrix A.
 *                       If JOBZ = 'N', then on exit the lower triangle (if
 *                       UPLO='L') or the upper triangle (if UPLO='U') of A,
 *                       including the diagonal, is destroyed.
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[out]    W      Array of dimension (n). If INFO = 0, the eigenvalues
 *                       in ascending order.
 * @param[out]    work   Complex workspace array, dimension (max(1, lwork)).
 *                       On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork  The length of the array work. lwork >= max(1, 2*n-1).
 *                       For optimal efficiency, lwork >= (NB+1)*N where NB is
 *                       the blocksize for CHETRD.
 *                       If lwork = -1, workspace query only.
 * @param[out]    rwork  Single precision workspace array, dimension (max(1, 3*n-2)).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the algorithm failed to converge; i
 *                           off-diagonal elements of an intermediate tridiagonal
 *                           form did not converge to zero.
 */
void cheev(const char* jobz, const char* uplo, const INT n,
           c64* restrict A, const INT lda,
           f32* restrict W,
           c64* restrict work, const INT lwork,
           f32* restrict rwork,
           INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);

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
        nb = lapack_get_nb("HETRD");
        lwkopt = (n > 1) ? (nb + 1) * n : 1;
        work[0] = CMPLXF((f32)lwkopt, 0.0f);

        if (lwork < (2 * n - 1 > 1 ? 2 * n - 1 : 1) && !lquery) {
            *info = -8;
        }
    }

    if (*info != 0) {
        xerbla("CHEEV ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = crealf(A[0]);
        work[0] = CMPLXF(1.0f, 0.0f);
        if (wantz) {
            A[0] = CONE;
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
    anrm = clanhe("M", uplo, n, A, lda, rwork);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        clascl(uplo, 0, 0, ONE, sigma, n, n, A, lda, info);
    }

    /* Call CHETRD to reduce Hermitian matrix to tridiagonal form */
    inde = 0;
    indtau = 0;
    indwrk = indtau + n;
    llwork = lwork - indwrk;

    chetrd(uplo, n, A, lda, W, &rwork[inde], &work[indtau],
           &work[indwrk], llwork, &iinfo);

    /* For eigenvalues only, call SSTERF. For eigenvectors, first call
     * CUNGTR to generate the unitary matrix, then call CSTEQR. */
    if (!wantz) {
        ssterf(n, W, &rwork[inde], info);
    } else {
        cungtr(uplo, n, A, lda, &work[indtau],
               &work[indwrk], llwork, &iinfo);
        indwrk = inde + n;
        csteqr(jobz, n, W, &rwork[inde], A, lda,
               &rwork[indwrk], info);
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

    /* Set WORK(1) to optimal complex workspace size */
    work[0] = CMPLXF((f32)lwkopt, 0.0f);
}
