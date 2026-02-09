/**
 * @file ssyevd.c
 * @brief SSYEVD computes all eigenvalues and, optionally, eigenvectors of a
 *        real symmetric matrix using divide and conquer algorithm.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"
#include "../include/lapack_tuning.h"

/**
 * SSYEVD computes all eigenvalues and, optionally, eigenvectors of a
 * real symmetric matrix A. If eigenvectors are desired, it uses a
 * divide and conquer algorithm.
 *
 * Because of large use of BLAS of level 3, SSYEVD needs N**2 more
 * workspace than SSYEVX.
 *
 * @param[in]     jobz    = 'N': Compute eigenvalues only;
 *                          = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo    = 'U': Upper triangle of A is stored;
 *                          = 'L': Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in,out] A       On entry, the symmetric matrix A.
 *                        On exit, if JOBZ = 'V', then A contains the orthonormal
 *                        eigenvectors of the matrix A.
 *                        If JOBZ = 'N', then the triangle is destroyed.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[out]    W       Array of dimension (n). The eigenvalues in ascending order.
 * @param[out]    work    Workspace array, dimension (max(1, lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of work.
 *                        If N <= 1, LWORK >= 1.
 *                        If JOBZ = 'N' and N > 1, LWORK >= 2*N+1.
 *                        If JOBZ = 'V' and N > 1, LWORK >= 1 + 6*N + 2*N**2.
 *                        If lwork = -1, workspace query only.
 * @param[out]    iwork   Integer workspace array, dimension (max(1, liwork)).
 *                        On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork  The dimension of iwork.
 *                        If N <= 1, LIWORK >= 1.
 *                        If JOBZ = 'N' and N > 1, LIWORK >= 1.
 *                        If JOBZ = 'V' and N > 1, LIWORK >= 3 + 5*N.
 *                        If liwork = -1, workspace query only.
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal value
 *                        > 0: if info = i and JOBZ = 'N', the algorithm failed to
 *                             converge; i off-diagonal elements did not converge
 *                             to zero; if info = i and JOBZ = 'V', the algorithm
 *                             failed to compute an eigenvalue while working on
 *                             the submatrix lying in rows and columns INFO/(N+1)
 *                             through mod(INFO,N+1).
 */
void ssyevd(const char* jobz, const char* uplo, const int n,
            float* const restrict A, const int lda,
            float* const restrict W,
            float* const restrict work, const int lwork,
            int* const restrict iwork, const int liwork,
            int* info)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    int lower, wantz, lquery;
    int iinfo, inde, indtau, indwrk, indwk2, iscale;
    int liopt, liwmin, llwork, llwrk2, lopt, lwmin;
    float anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;
    int nb;

    /* Test the input parameters */
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1 || liwork == -1);

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
        if (n <= 1) {
            liwmin = 1;
            lwmin = 1;
            lopt = lwmin;
            liopt = liwmin;
        } else {
            if (wantz) {
                liwmin = 3 + 5 * n;
                lwmin = 1 + 6 * n + 2 * n * n;
            } else {
                liwmin = 1;
                lwmin = 2 * n + 1;
            }
            nb = lapack_get_nb("SYTRD");
            lopt = lwmin > 2 * n + n * nb ? lwmin : 2 * n + n * nb;
            liopt = liwmin;
        }
        work[0] = (float)lopt;
        iwork[0] = liopt;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (liwork < liwmin && !lquery) {
            *info = -10;
        }
    }

    if (*info != 0) {
        xerbla("SSYEVD", -(*info));
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
    inde = 0;            /* E starts at work[0] */
    indtau = inde + n;   /* tau starts at work[n] */
    indwrk = indtau + n; /* temporary work at work[2*n] */
    llwork = lwork - indwrk;
    indwk2 = indwrk + n * n; /* second temporary at work[2*n + n*n] */
    llwrk2 = lwork - indwk2;

    ssytrd(uplo, n, A, lda, W, &work[inde], &work[indtau],
           &work[indwrk], llwork, &iinfo);

    /* For eigenvalues only, call SSTERF. For eigenvectors, first call
     * SSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
     * tridiagonal matrix, then call SORMTR to multiply it by the
     * Householder transformations stored in A. */
    if (!wantz) {
        ssterf(n, W, &work[inde], info);
    } else {
        sstedc("I", n, W, &work[inde], &work[indwrk], n,
               &work[indwk2], llwrk2, iwork, liwork, info);
        sormtr("L", uplo, "N", n, n, A, lda, &work[indtau],
               &work[indwrk], n, &work[indwk2], llwrk2, &iinfo);
        slacpy("A", n, n, &work[indwrk], n, A, lda);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately */
    if (iscale == 1) {
        cblas_sscal(n, ONE / sigma, W, 1);
    }

    work[0] = (float)lopt;
    iwork[0] = liopt;
}
