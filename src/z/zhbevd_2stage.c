/**
 * @file zhbevd_2stage.c
 * @brief ZHBEVD_2STAGE computes the eigenvalues and, optionally, eigenvectors
 *        of a complex Hermitian band matrix using the 2stage technique.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZHBEVD_2STAGE computes all the eigenvalues and, optionally, eigenvectors of
 * a complex Hermitian band matrix A using the 2stage technique for
 * the reduction to tridiagonal.  If eigenvectors are desired, it
 * uses a divide and conquer algorithm.
 *
 * @param[in]     jobz    = 'N':  Compute eigenvalues only;
 *                          = 'V':  Compute eigenvalues and eigenvectors.
 *                                  Not available in this release.
 * @param[in]     uplo    = 'U':  Upper triangle of A is stored;
 *                          = 'L':  Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A.  N >= 0.
 * @param[in]     kd      The number of superdiagonals of the matrix A if
 *                         UPLO = 'U', or the number of subdiagonals if
 *                         UPLO = 'L'.  KD >= 0.
 * @param[in,out] AB      Complex*16 array, dimension (ldab, n).
 *                         On entry, the upper or lower triangle of the
 *                         Hermitian band matrix A, stored in the first KD+1
 *                         rows of the array.
 *                         On exit, AB is overwritten by values generated
 *                         during the reduction to tridiagonal form.
 * @param[in]     ldab    The leading dimension of the array AB.
 *                         LDAB >= KD + 1.
 * @param[out]    W       Double precision array, dimension (n).
 *                         If INFO = 0, the eigenvalues in ascending order.
 * @param[out]    Z       Complex*16 array, dimension (ldz, n).
 *                         If JOBZ = 'V', then if INFO = 0, Z contains the
 *                         orthonormal eigenvectors of the matrix A.
 *                         If JOBZ = 'N', then Z is not referenced.
 * @param[in]     ldz     The leading dimension of the array Z.  LDZ >= 1,
 *                         and if JOBZ = 'V', LDZ >= max(1,N).
 * @param[out]    work    Complex*16 array, dimension (max(1,lwork)).
 *                         On exit, if INFO = 0, WORK(0) returns the optimal
 *                         LWORK.
 * @param[in]     lwork   The length of the array WORK.
 * @param[out]    rwork   Double precision array, dimension (lrwork).
 *                         On exit, if INFO = 0, RWORK(0) returns the optimal
 *                         LRWORK.
 * @param[in]     lrwork  The dimension of array RWORK.
 * @param[out]    iwork   Integer array, dimension (max(1,liwork)).
 *                         On exit, if INFO = 0, IWORK(0) returns the optimal
 *                         LIWORK.
 * @param[in]     liwork  The dimension of array IWORK.
 * @param[out]    info    = 0:  successful exit.
 *                         < 0:  if INFO = -i, the i-th argument had an
 *                               illegal value.
 *                         > 0:  if INFO = i, the algorithm failed to converge;
 *                               i off-diagonal elements of an intermediate
 *                               tridiagonal form did not converge to zero.
 */
void zhbevd_2stage(const char* jobz, const char* uplo, const INT n,
                   const INT kd, c128* restrict AB,
                   const INT ldab, f64* restrict W,
                   c128* restrict Z, const INT ldz,
                   c128* restrict work, const INT lwork,
                   f64* restrict rwork, const INT lrwork,
                   INT* restrict iwork, const INT liwork,
                   INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT lower, lquery, wantz;
    INT iinfo, imax, inde, indwk2, indrwk, iscale;
    INT llwork, indwk, lhtrd = 0, lwtrd, ib, indhous;
    INT liwmin, llrwk, llwk2, lrwmin, lwmin;
    f64 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1 || liwork == -1 || lrwork == -1);

    *info = 0;
    if (n <= 1) {
        lwmin = 1;
        lrwmin = 1;
        liwmin = 1;
    } else {
        ib    = ilaenv2stage(2, "ZHETRD_HB2ST", jobz, n, kd, -1, -1);
        lhtrd = ilaenv2stage(3, "ZHETRD_HB2ST", jobz, n, kd, ib, -1);
        lwtrd = ilaenv2stage(4, "ZHETRD_HB2ST", jobz, n, kd, ib, -1);
        if (wantz) {
            lwmin = 2 * n * n;
            lrwmin = 1 + 5 * n + 2 * n * n;
            liwmin = 3 + 5 * n;
        } else {
            lwmin = (n > lhtrd + lwtrd) ? n : lhtrd + lwtrd;
            lrwmin = n;
            liwmin = 1;
        }
    }
    if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(lower || uplo[0] == 'U' || uplo[0] == 'u')) {
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
        work[0] = CMPLX((f64)lwmin, 0.0);
        rwork[0] = (f64)lrwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -11;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -13;
        } else if (liwork < liwmin && !lquery) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("ZHBEVD_2STAGE", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */

    if (n == 0)
        return;

    if (n == 1) {
        W[0] = creal(AB[0]);
        if (wantz)
            Z[0] = CONE;
        return;
    }

    /* Get machine constants. */

    safmin = dlamch("Safe minimum");
    eps    = dlamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin   = sqrt(smlnum);
    rmax   = sqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */

    anrm = zlanhb("M", uplo, n, kd, AB, ldab, rwork);
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
            zlascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            zlascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
    }

    /* Call ZHETRD_HB2ST to reduce Hermitian band matrix to tridiagonal form. */

    inde    = 0;
    indrwk  = inde + n;
    llrwk   = lrwork - indrwk;
    indhous = 0;
    indwk   = indhous + lhtrd;
    llwork  = lwork - indwk;
    indwk2  = indwk + n * n;
    llwk2   = lwork - indwk2;

    zhetrd_hb2st("N", jobz, uplo, n, kd, AB, ldab, W,
                 &rwork[inde], &work[indhous], lhtrd,
                 &work[indwk], llwork, &iinfo);

    /* For eigenvalues only, call DSTERF.  For eigenvectors, call ZSTEDC. */

    if (!wantz) {
        dsterf(n, W, &rwork[inde], info);
    } else {
        zstedc("I", n, W, &rwork[inde], work, n,
               &work[indwk2],
               llwk2, &rwork[indrwk], llrwk, iwork, liwork,
               info);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, &CONE, Z, ldz, work, n, &CZERO,
                    &work[indwk2], n);
        zlacpy("A", n, n, &work[indwk2], n, Z, ldz);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, W, 1);
    }

    work[0] = CMPLX((f64)lwmin, 0.0);
    rwork[0] = (f64)lrwmin;
    iwork[0] = liwmin;
    return;
}
