/**
 * @file dspevd.c
 * @brief DSPEVD computes all the eigenvalues and, optionally, eigenvectors
 *        of a real symmetric matrix A in packed storage using divide and conquer.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSPEVD computes all the eigenvalues and, optionally, eigenvectors
 * of a real symmetric matrix A in packed storage. If eigenvectors are
 * desired, it uses a divide and conquer algorithm.
 *
 * @param[in]     jobz    = 'N': Compute eigenvalues only;
 *                        = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo    = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in,out] AP      Double precision array, dimension (n*(n+1)/2).
 *                        On entry, the upper or lower triangle of the symmetric
 *                        matrix A, packed columnwise.
 *                        On exit, AP is overwritten by values generated during
 *                        the reduction to tridiagonal form.
 * @param[out]    W       Double precision array, dimension (n).
 *                        If info = 0, the eigenvalues in ascending order.
 * @param[out]    Z       Double precision array, dimension (ldz, n).
 *                        If jobz = 'V', then if info = 0, Z contains the
 *                        orthonormal eigenvectors of the matrix A.
 *                        If jobz = 'N', then Z is not referenced.
 * @param[in]     ldz     The leading dimension of the array Z. ldz >= 1, and if
 *                        jobz = 'V', ldz >= max(1, n).
 * @param[out]    work    Double precision array, dimension (max(1, lwork)).
 *                        On exit, if info = 0, work[0] returns the required lwork.
 * @param[in]     lwork   The dimension of the array work.
 *                        If n <= 1, lwork must be at least 1.
 *                        If jobz = 'N' and n > 1, lwork must be at least 2*n.
 *                        If jobz = 'V' and n > 1, lwork must be at least 1 + 6*n + n**2.
 *                        If lwork = -1, a workspace query is assumed.
 * @param[out]    iwork   Integer array, dimension (max(1, liwork)).
 *                        On exit, if info = 0, iwork[0] returns the required liwork.
 * @param[in]     liwork  The dimension of the array iwork.
 *                        If jobz = 'N' or n <= 1, liwork must be at least 1.
 *                        If jobz = 'V' and n > 1, liwork must be at least 3 + 5*n.
 *                        If liwork = -1, a workspace query is assumed.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if info = i, the algorithm failed to converge.
 */
void dspevd(const char* jobz, const char* uplo, const int n,
            f64* restrict AP, f64* restrict W,
            f64* restrict Z, const int ldz,
            f64* restrict work, const int lwork,
            int* restrict iwork, const int liwork, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int lquery, wantz;
    int iinfo, inde, indtau, indwrk, iscale, liwmin, llwork, lwmin;
    f64 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
               !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -7;
    }

    if (*info == 0) {
        if (n <= 1) {
            liwmin = 1;
            lwmin = 1;
        } else {
            if (wantz) {
                liwmin = 3 + 5 * n;
                lwmin = 1 + 6 * n + n * n;
            } else {
                liwmin = 1;
                lwmin = 2 * n;
            }
        }
        iwork[0] = liwmin;
        work[0] = (f64)lwmin;

        if (lwork < lwmin && !lquery) {
            *info = -9;
        } else if (liwork < liwmin && !lquery) {
            *info = -11;
        }
    }

    if (*info != 0) {
        xerbla("DSPEVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    if (n == 1) {
        W[0] = AP[0];
        if (wantz) {
            Z[0 + 0 * ldz] = ONE;
        }
        return;
    }

    /* Get machine constants. */

    safmin = dlamch("Safe minimum");
    eps = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */

    anrm = dlansp("M", uplo, n, AP, work);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        cblas_dscal((n * (n + 1)) / 2, sigma, AP, 1);
    }

    /* Call DSPTRD to reduce symmetric packed matrix to tridiagonal form. */

    inde = 0;
    indtau = inde + n;
    dsptrd(uplo, n, AP, W, &work[inde], &work[indtau], &iinfo);

    /* For eigenvalues only, call DSTERF.  For eigenvectors, first call
       DSTEDC to generate the eigenvector matrix, WORK(INDWRK), of the
       tridiagonal matrix, then call DOPMTR to multiply it by the
       Householder transformations represented in AP. */

    if (!wantz) {
        dsterf(n, W, &work[inde], info);
    } else {
        indwrk = indtau + n;
        llwork = lwork - indwrk;
        dstedc("I", n, W, &work[inde], Z, ldz, &work[indwrk], llwork,
               iwork, liwork, info);
        dopmtr("L", uplo, "N", n, n, AP, &work[indtau], Z, ldz,
               &work[indwrk], &iinfo);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
        cblas_dscal(n, ONE / sigma, W, 1);
    }

    work[0] = (f64)lwmin;
    iwork[0] = liwmin;
}
