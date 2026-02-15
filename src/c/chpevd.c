/**
 * @file chpevd.c
 * @brief CHPEVD computes the eigenvalues and, optionally, eigenvectors for a
 *        complex Hermitian matrix in packed storage.
 */
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

/**
 * CHPEVD computes all the eigenvalues and, optionally, eigenvectors of
 * a complex Hermitian matrix A in packed storage.  If eigenvectors are
 * desired, it uses a divide and conquer algorithm.
 *
 * @param[in]     jobz    = 'N':  Compute eigenvalues only;
 *                          = 'V':  Compute eigenvalues and eigenvectors.
 * @param[in]     uplo    = 'U':  Upper triangle of A is stored;
 *                          = 'L':  Lower triangle of A is stored.
 * @param[in]     n       The order of the matrix A.  N >= 0.
 * @param[in,out] AP      COMPLEX*16 array, dimension (N*(N+1)/2).
 *                         On entry, the upper or lower triangle of the
 *                         Hermitian matrix A, packed columnwise in a linear
 *                         array.  The j-th column of A is stored in the array
 *                         AP as follows:
 *                         if UPLO = 'U', AP(i + (j-1)*j/2) = A(i,j) for
 *                         1<=i<=j;
 *                         if UPLO = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for
 *                         j<=i<=n.
 *                         On exit, AP is overwritten by values generated during
 *                         the reduction to tridiagonal form.
 * @param[out]    W       DOUBLE PRECISION array, dimension (N).
 *                         If INFO = 0, the eigenvalues in ascending order.
 * @param[out]    Z       COMPLEX*16 array, dimension (LDZ, N).
 *                         If JOBZ = 'V', then if INFO = 0, Z contains the
 *                         orthonormal eigenvectors of the matrix A, with the
 *                         i-th column of Z holding the eigenvector associated
 *                         with W(i).
 *                         If JOBZ = 'N', then Z is not referenced.
 * @param[in]     ldz     The leading dimension of the array Z.  LDZ >= 1, and
 *                         if JOBZ = 'V', LDZ >= max(1,N).
 * @param[out]    work    COMPLEX*16 array, dimension (MAX(1,LWORK)).
 *                         On exit, if INFO = 0, WORK(1) returns the required
 *                         LWORK.
 * @param[in]     lwork   The dimension of array WORK.
 *                         If N <= 1,               LWORK must be at least 1.
 *                         If JOBZ = 'N' and N > 1, LWORK must be at least N.
 *                         If JOBZ = 'V' and N > 1, LWORK must be at least 2*N.
 *                         If LWORK = -1, then a workspace query is assumed.
 * @param[out]    rwork   DOUBLE PRECISION array, dimension (MAX(1,LRWORK)).
 *                         On exit, if INFO = 0, RWORK(1) returns the required
 *                         LRWORK.
 * @param[in]     lrwork  The dimension of array RWORK.
 *                         If N <= 1,               LRWORK must be at least 1.
 *                         If JOBZ = 'N' and N > 1, LRWORK must be at least N.
 *                         If JOBZ = 'V' and N > 1, LRWORK must be at least
 *                         1 + 5*N + 2*N**2.
 *                         If LRWORK = -1, then a workspace query is assumed.
 * @param[out]    iwork   INTEGER array, dimension (MAX(1,LIWORK)).
 *                         On exit, if INFO = 0, IWORK(1) returns the required
 *                         LIWORK.
 * @param[in]     liwork  The dimension of array IWORK.
 *                         If JOBZ = 'N' or N <= 1, LIWORK must be at least 1.
 *                         If JOBZ = 'V' and N > 1, LIWORK must be at least
 *                         3 + 5*N.
 *                         If LIWORK = -1, then a workspace query is assumed.
 * @param[out]    info    = 0:  successful exit
 *                         < 0:  if INFO = -i, the i-th argument had an illegal
 *                         value.
 *                         > 0:  if INFO = i, the algorithm failed to converge;
 *                         i off-diagonal elements of an intermediate
 *                         tridiagonal form did not converge to zero.
 */
void chpevd(
    const char* jobz,
    const char* uplo,
    const int n,
    c64* restrict AP,
    f32* restrict W,
    c64* restrict Z,
    const int ldz,
    c64* restrict work,
    const int lwork,
    f32* restrict rwork,
    const int lrwork,
    int* restrict iwork,
    const int liwork,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    int wantz, lquery;
    int iinfo, imax, inde, indrwk, indtau, indwrk;
    int iscale, liwmin, llrwk, llwrk, lrwmin, lwmin;
    f32 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    *info = 0;
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!(uplo[0] == 'L' || uplo[0] == 'l' ||
                 uplo[0] == 'U' || uplo[0] == 'u')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -7;
    }

    if (*info == 0) {
        if (n <= 1) {
            lwmin = 1;
            liwmin = 1;
            lrwmin = 1;
        } else {
            if (wantz) {
                lwmin = 2 * n;
                lrwmin = 1 + 5 * n + 2 * n * n;
                liwmin = 3 + 5 * n;
            } else {
                lwmin = n;
                lrwmin = n;
                liwmin = 1;
            }
        }
        work[0] = CMPLXF((f32)lwmin, 0.0f);
        rwork[0] = (f32)lrwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -9;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -11;
        } else if (liwork < liwmin && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        xerbla("CHPEVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */

    if (n == 0)
        return;

    if (n == 1) {
        W[0] = crealf(AP[0]);
        if (wantz)
            Z[0] = CONE;
        return;
    }

    /* Get machine constants. */

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    /* Scale matrix to allowable range, if necessary. */

    anrm = clanhp("M", uplo, n, AP, rwork);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        cblas_csscal((n * (n + 1)) / 2, sigma, AP, 1);
    }

    /* Call CHPTRD to reduce Hermitian packed matrix to tridiagonal form. */

    inde = 0;
    indtau = 0;
    indrwk = inde + n;
    indwrk = indtau + n;
    llwrk = lwork - indwrk;
    llrwk = lrwork - indrwk;
    chptrd(uplo, n, AP, W, &rwork[inde], &work[indtau], &iinfo);

    /* For eigenvalues only, call SSTERF.  For eigenvectors, first call
       CSTEDC to generate the eigenvectors from the tridiagonal form,
       then call CUPMTR to multiply by the Householder transformations. */

    if (!wantz) {
        ssterf(n, W, &rwork[inde], info);
    } else {
        cstedc("I", n, W, &rwork[inde], Z, ldz,
               &work[indwrk], llwrk, &rwork[indrwk], llrwk, iwork, liwork,
               info);
        cupmtr("L", uplo, "N", n, n, AP, &work[indtau], Z, ldz,
               &work[indwrk], &iinfo);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
    rwork[0] = (f32)lrwmin;
    iwork[0] = liwmin;
}
