/**
 * @file zheevd.c
 * @brief ZHEEVD computes all eigenvalues and, optionally, eigenvectors of a
 *        complex Hermitian matrix using divide and conquer algorithm.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"
#include "../include/lapack_tuning.h"

/**
 * ZHEEVD computes all eigenvalues and, optionally, eigenvectors of a
 * complex Hermitian matrix A.  If eigenvectors are desired, it uses a
 * divide and conquer algorithm.
 *
 * @param[in]     jobz    = 'N': Compute eigenvalues only;
 *                          = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo    = 'U': Upper triangle of A is stored;
 *                          = 'L': Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in,out] A       On entry, the Hermitian matrix A.
 *                        On exit, if JOBZ = 'V', then A contains the orthonormal
 *                        eigenvectors of the matrix A.
 *                        If JOBZ = 'N', then the triangle is destroyed.
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[out]    W       Array of dimension (n). The eigenvalues in ascending order.
 * @param[out]    work    Complex workspace array, dimension (max(1, lwork)).
 *                        On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork   The dimension of work.
 *                        If N <= 1, LWORK >= 1.
 *                        If JOBZ = 'N' and N > 1, LWORK >= N + 1.
 *                        If JOBZ = 'V' and N > 1, LWORK >= 2*N + N**2.
 *                        If lwork = -1, workspace query only.
 * @param[out]    rwork   Double precision workspace array, dimension (max(1, lrwork)).
 *                        On exit, if info = 0, rwork[0] returns the optimal lrwork.
 * @param[in]     lrwork  The dimension of rwork.
 *                        If N <= 1, LRWORK >= 1.
 *                        If JOBZ = 'N' and N > 1, LRWORK >= N.
 *                        If JOBZ = 'V' and N > 1, LRWORK >= 1 + 5*N + 2*N**2.
 *                        If lrwork = -1, workspace query only.
 * @param[out]    iwork   Integer workspace array, dimension (max(1, liwork)).
 *                        On exit, if info = 0, iwork[0] returns the optimal liwork.
 * @param[in]     liwork  The dimension of iwork.
 *                        If N <= 1, LIWORK >= 1.
 *                        If JOBZ = 'N' and N > 1, LIWORK >= 1.
 *                        If JOBZ = 'V' and N > 1, LIWORK >= 3 + 5*N.
 *                        If liwork = -1, workspace query only.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i and JOBZ = 'N', the algorithm failed to
 *                           converge; i off-diagonal elements did not converge
 *                           to zero; if info = i and JOBZ = 'V', the algorithm
 *                           failed to compute an eigenvalue while working on
 *                           the submatrix lying in rows and columns INFO/(N+1)
 *                           through mod(INFO,N+1).
 */
void zheevd(const char* jobz, const char* uplo, const INT n,
            c128* restrict A, const INT lda,
            f64* restrict W,
            c128* restrict work, const INT lwork,
            f64* restrict rwork, const INT lrwork,
            INT* restrict iwork, const INT liwork,
            INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);

    INT lower, wantz, lquery;
    INT iinfo, imax, inde, indrwk, indtau, indwk2, indwrk, iscale;
    INT liopt, liwmin, llrwk, llwork, llwrk2, lopt, lropt, lrwmin, lwmin;
    f64 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;
    INT nb;

    /* Test the input parameters */
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

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
            lwmin = 1;
            lrwmin = 1;
            liwmin = 1;
            lopt = lwmin;
            lropt = lrwmin;
            liopt = liwmin;
        } else {
            if (wantz) {
                lwmin = 2 * n + n * n;
                lrwmin = 1 + 5 * n + 2 * n * n;
                liwmin = 3 + 5 * n;
            } else {
                lwmin = n + 1;
                lrwmin = n;
                liwmin = 1;
            }
            nb = lapack_get_nb("HETRD");
            lopt = lwmin > (n + n * nb) ? lwmin : (n + n * nb);
            lropt = lrwmin;
            liopt = liwmin;
        }
        work[0] = CMPLX((f64)lopt, 0.0);
        rwork[0] = (f64)lropt;
        iwork[0] = liopt;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -10;
        } else if (liwork < liwmin && !lquery) {
            *info = -12;
        }
    }

    if (*info != 0) {
        xerbla("ZHEEVD", -(*info));
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
            A[0] = CONE;
        }
        return;
    }

    /* Get machine constants */
    safmin = dlamch("S");
    eps = dlamch("E");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

    /* Scale matrix to allowable range, if necessary */
    anrm = zlanhe("M", uplo, n, A, lda, rwork);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        zlascl(uplo, 0, 0, ONE, sigma, n, n, A, lda, info);
    }

    /* Call ZHETRD to reduce Hermitian matrix to tridiagonal form */
    inde = 0;
    indtau = 0;
    indwrk = indtau + n;
    indrwk = inde + n;
    indwk2 = indwrk + n * n;
    llwork = lwork - indwrk;
    llwrk2 = lwork - indwk2;
    llrwk = lrwork - indrwk;
    zhetrd(uplo, n, A, lda, W, &rwork[inde], &work[indtau],
           &work[indwrk], llwork, &iinfo);

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
     * ZSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
     * tridiagonal matrix, then call ZUNMTR to multiply it to the
     * Householder transformations represented as Householder vectors in
     * A. */
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

    /* If matrix was scaled, then rescale eigenvalues appropriately */
    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, W, 1);
    }

    work[0] = CMPLX((f64)lopt, 0.0);
    rwork[0] = (f64)lropt;
    iwork[0] = liopt;
}
