/**
 * @file zheevd_2stage.c
 * @brief ZHEEVD_2STAGE computes eigenvalues and optionally eigenvectors using
 *        divide-and-conquer with 2-stage reduction.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include "semicolon_cblas.h"
#include <math.h>

/**
 * ZHEEVD_2STAGE computes all eigenvalues and, optionally, eigenvectors of a
 * complex Hermitian matrix A using the 2stage technique for
 * the reduction to tridiagonal.  If eigenvectors are desired, it uses a
 * divide and conquer algorithm.
 *
 * @param[in]     jobz    = 'N':  Compute eigenvalues only;
 *                          = 'V':  Compute eigenvalues and eigenvectors.
 *                                  Not available in this release.
 * @param[in]     uplo    = 'U':  Upper triangle of A is stored;
 *                          = 'L':  Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A.  N >= 0.
 * @param[in,out] A       Complex*16 array, dimension (lda, n).
 *                         On entry, the Hermitian matrix A.
 *                         On exit, if JOBZ = 'V', then if INFO = 0, A contains
 *                         the orthonormal eigenvectors of the matrix A.
 * @param[in]     lda     The leading dimension of the array A.  lda >= max(1,n).
 * @param[out]    W       Double precision array, dimension (n).
 *                         If INFO = 0, the eigenvalues in ascending order.
 * @param[out]    work    Complex*16 array, dimension (max(1,lwork)).
 *                         On exit, if INFO = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of the array work.
 * @param[out]    rwork   Double precision array, dimension (lrwork).
 *                         On exit, if INFO = 0, rwork[0] returns the optimal lrwork.
 * @param[in]     lrwork  The dimension of the array rwork.
 * @param[out]    iwork   Integer array, dimension (max(1,liwork)).
 *                         On exit, if INFO = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork  The dimension of the array iwork.
 * @param[out]    info    = 0:  successful exit
 *                         < 0:  if info = -i, the i-th argument had an illegal value
 *                         > 0:  if info = i and JOBZ = 'N', then the algorithm failed
 *                               to converge; i off-diagonal elements of an intermediate
 *                               tridiagonal form did not converge to zero;
 *                               if info = i and JOBZ = 'V', then the algorithm failed
 *                               to compute an eigenvalue while working on the submatrix
 *                               lying in rows and columns INFO/(N+1) through
 *                               mod(INFO,N+1).
 */
void zheevd_2stage(const char* jobz, const char* uplo, const INT n,
                   c128* A, const INT lda,
                   f64* W,
                   c128* work, const INT lwork,
                   f64* rwork, const INT lrwork,
                   INT* iwork, const INT liwork, INT* info)
{
    const f64 zero = 0.0;
    const f64 one = 1.0;
    const c128 cone = CMPLX(1.0, 0.0);

    INT lower, lquery, wantz;
    INT iinfo, imax, inde, indrwk, indtau, indwk2, indwrk, iscale;
    INT liwmin, llrwk, llwork, llwrk2, lrwmin, lwmin;
    INT lhtrd = 0, lwtrd, kd, ib, indhous;
    f64 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

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
            lwmin = 1;
            lrwmin = 1;
            liwmin = 1;
        } else {
            kd = ilaenv2stage(1, "ZHETRD_2STAGE", jobz, n, -1, -1, -1);
            ib = ilaenv2stage(2, "ZHETRD_2STAGE", jobz, n, kd, -1, -1);
            lhtrd = ilaenv2stage(3, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);
            lwtrd = ilaenv2stage(4, "ZHETRD_2STAGE", jobz, n, kd, ib, -1);
            if (wantz) {
                lwmin = 2 * n + n * n;
                lrwmin = 1 + 5 * n + 2 * n * n;
                liwmin = 3 + 5 * n;
            } else {
                lwmin = n + 1 + lhtrd + lwtrd;
                lrwmin = n;
                liwmin = 1;
            }
        }
        work[0] = CMPLX((f64)lwmin, 0.0);
        rwork[0] = (f64)lrwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -10;
        } else if (liwork < liwmin && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("ZHEEVD_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */

    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = creal(A[0]);
        if (wantz) {
            A[0] = cone;
        }
        return;
    }

    /* Get machine constants. */

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = one / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */

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

    /* Call ZHETRD_2STAGE to reduce Hermitian matrix to tridiagonal form. */

    inde = 0;
    indrwk = inde + n;
    llrwk = lrwork - indrwk;
    indtau = 0;
    indhous = indtau + n;
    indwrk = indhous + lhtrd;
    llwork = lwork - indwrk;
    indwk2 = indwrk + n * n;
    llwrk2 = lwork - indwk2;

    zhetrd_2stage(jobz, uplo, n, A, lda, W, &rwork[inde],
                  &work[indtau], &work[indhous], lhtrd,
                  &work[indwrk], llwork, &iinfo);

    /*
     * For eigenvalues only, call DSTERF.  For eigenvectors, first call
     * ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
     * tridiagonal matrix, then call ZUNMTR to multiply it to the
     * Householder transformations represented as Householder vectors in
     * A.
     */

    if (!wantz) {
        dsterf(n, W, &rwork[inde], info);
    } else {
        zstedc("I", n, W, &rwork[inde], &work[indwrk], n,
               &work[indwk2], llwrk2, &rwork[indrwk], llrwk,
               iwork, liwork, info);
        zunmtr("L", uplo, "N", n, n, A, lda, &work[indtau],
               &work[indwrk], n, &work[indwk2], llwrk2, &iinfo);
        zlacpy("A", n, n, &work[indwrk], n, A, lda);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, one / sigma, W, 1);
    }

    work[0] = CMPLX((f64)lwmin, 0.0);
    rwork[0] = (f64)lrwmin;
    iwork[0] = liwmin;
}
