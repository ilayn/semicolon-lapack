/**
 * @file ssyevd_2stage.c
 * @brief SSYEVD_2STAGE computes eigenvalues and optionally eigenvectors using divide-and-conquer.
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include <math.h>

/**
 * SSYEVD_2STAGE computes all eigenvalues and, optionally, eigenvectors of a
 * real symmetric matrix `A` using the 2stage technique for the reduction to
 * tridiagonal. If eigenvectors are desired, it uses a divide and conquer
 * algorithm.
 *
 * @param[in]     jobz   `'N'`: Compute eigenvalues only;
 *                       `'V'`: Compute eigenvalues and eigenvectors.
 *                              Not available in this release.
 * @param[in]     uplo   `'U'`: Upper triangle of `A` is stored;
 *                       `'L'`: Lower triangle of `A` is stored.
 * @param[in]     n      The order of the matrix `A`. `n>=0`.
 * @param[in,out] A      On entry, the symmetric matrix `A`. If `uplo='U'`, the
 *                       leading `n`-by-`n` upper triangular part of `A` contains the
 *                       upper triangular part of the matrix `A`. If `uplo='L'`,
 *                       the leading `n`-by-`n` lower triangular part of `A` contains
 *                       the lower triangular part of the matrix `A`.
 *                       On exit, if `jobz='V'`, then if `info=0`, `A` contains the
 *                       orthonormal eigenvectors of the matrix `A`.
 *                       If `jobz='N'`, then on exit the lower triangle (if `uplo='L'`)
 *                       or the upper triangle (if `uplo='U'`) of `A`, including the
 *                       diagonal, is destroyed.
 * @param[in]     lda    The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[out]    W      Array of dimension (`n`). If `info=0`, the eigenvalues in
 *                       ascending order.
 * @param[out]    work   Array of dimension (`lwork`). On exit, if `info=0`, `work[0]`
 *                       returns the optimal `lwork`.
 * @param[in]     lwork  The dimension of the array `work`.
 *                       If `n<=1`, `lwork` must be at least 1.
 *                       If `jobz='N'` and `n>1`, `lwork` must be queried.
 *                       @rst
 *                       .. code-block:: text
 *
 *                           lwork = MAX(1, dimension) where
 *                           dimension = max(stage1,stage2) + (kd+1)*n + 2*n+1
 *                                     = n*kd + n*max(kd+1,FACTOPTNB)
 *                                       + max(2*kd*kd, kd*NTHREADS)
 *                                       + (kd+1)*n + 2*n+1
 *                       @endrst
 *                       where `kd` is the blocking size of the reduction,
 *                       FACTOPTNB is the blocking used by the QR or LQ
 *                       algorithm, usually FACTOPTNB=128 is a good choice,
 *                       NTHREADS is the number of threads used when openMP
 *                       compilation is enabled, otherwise =1.
 *                       If `jobz='V'` and `n>1`, `lwork` must be at least `1 + 6*n + 2*n**2`.
 *                       If `lwork=-1`, then a workspace query is assumed; the routine
 *                       only calculates the optimal sizes of the `work` and `iwork`
 *                       arrays, returns these values as the first entries of the `work`
 *                       and `iwork` arrays.
 * @param[out]    iwork  Integer array, dimension (`max(1,liwork)`). On exit, if
 *                       `info=0`, `iwork[0]` returns the optimal `liwork`.
 * @param[in]     liwork The dimension of the array `iwork`.
 *                       If `n<=1`, `liwork` must be at least 1.
 *                       If `jobz='N'` and `n>1`, `liwork` must be at least 1.
 *                       If `jobz='V'` and `n>1`, `liwork` must be at least `3 + 5*n`.
 *                       If `liwork=-1`, then a workspace query is assumed; the routine
 *                       only calculates the optimal sizes of the `work` and `iwork`
 *                       arrays, returns these values as the first entries of the `work`
 *                       and `iwork` arrays.
 * @param[out]    info   `info=0`: successful exit
 *                       `info<0`: if `info=-i`, the i-th argument had an illegal value
 *                       `info>0`: if `info=i` and `jobz='N'`, then the algorithm failed
 *                                 to converge; i off-diagonal elements of an intermediate
 *                                 tridiagonal form did not converge to zero;
 *                                 if `info=i` and `jobz='V'`, then the algorithm failed
 *                                 to compute an eigenvalue while working on the submatrix
 *                                 lying in rows and columns `info/(n+1)` through
 *                                 `mod(info,n+1)`.
 *
 * @par Further Details:
 * @rst
 * All details about the 2stage techniques are available in:
 *
 * Azzam Haidar, Hatem Ltaief, and Jack Dongarra.
 * Parallel reduction to condensed forms for symmetric eigenvalue problems
 * using aggregated fine-grained and memory-aware kernels. In Proceedings
 * of 2011 International Conference for High Performance Computing,
 * Networking, Storage and Analysis (SC '11), New York, NY, USA,
 * Article 8 , 11 pages.
 * https://doi.org/10.1145/2063384.2063394
 *
 * A. Haidar, J. Kurzak, P. Luszczek, 2013.
 * An improved parallel singular value algorithm and its implementation
 * for multicore hardware, In Proceedings of 2013 International Conference
 * for High Performance Computing, Networking, Storage and Analysis (SC '13).
 * Denver, Colorado, USA, 2013.
 * Article 90, 12 pages.
 * https://doi.org/10.1145/2503210.2503292
 *
 * A. Haidar, R. Solca, S. Tomov, T. Schulthess and J. Dongarra.
 * A novel hybrid CPU-GPU generalized eigensolver for electronic structure
 * calculations based on fine-grained memory aware tasks.
 * International Journal of High Performance Computing Applications.
 * Volume 28 Issue 2, Pages 196-209, May 2014.
 * https://doi.org/10.1177/1094342013502097
 * @endrst
 */
void ssyevd_2stage(const char* jobz, const char* uplo, const INT n,
                   f32* A, const INT lda,
                   f32* W,
                   f32* work, const INT lwork,
                   INT* iwork, const INT liwork, INT* info)
{
    const f32 zero = 0.0f;
    const f32 one = 1.0f;

    INT lower, lquery, wantz;
    INT iinfo, inde, indtau, indwrk, iscale;
    INT liwmin, llwork, lwmin;
    /* INT indwk2, llwrk2; - used only in eigenvector path (disabled) */
    INT lhtrd = 0, lwtrd, kd, ib, indhous;
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
