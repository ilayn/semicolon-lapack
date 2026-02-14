/**
 * @file sstevd.c
 * @brief SSTEVD computes all eigenvalues and, optionally, eigenvectors of a
 *        real symmetric tridiagonal matrix using divide and conquer.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSTEVD computes all eigenvalues and, optionally, eigenvectors of a
 * real symmetric tridiagonal matrix. If eigenvectors are desired, it
 * uses a divide and conquer algorithm.
 *
 * @param[in]     jobz   = 'N':  Compute eigenvalues only;
 *                         = 'V':  Compute eigenvalues and eigenvectors.
 * @param[in]     n      The order of the matrix.  N >= 0.
 * @param[in,out] D      Double precision array, dimension (N).
 *                        On entry, the n diagonal elements of the tridiagonal
 *                        matrix A.
 *                        On exit, if INFO = 0, the eigenvalues in ascending order.
 * @param[in,out] E      Double precision array, dimension (N-1).
 *                        On entry, the (n-1) subdiagonal elements of the
 *                        tridiagonal matrix A, stored in elements 0 to N-2.
 *                        On exit, the contents of E are destroyed.
 * @param[out]    Z      Double precision array, dimension (LDZ, N).
 *                        If JOBZ = 'V', then if INFO = 0, Z contains the
 *                        orthonormal eigenvectors of the matrix A, with the
 *                        i-th column of Z holding the eigenvector associated
 *                        with D(i).
 *                        If JOBZ = 'N', then Z is not referenced.
 * @param[in]     ldz    The leading dimension of the array Z.  LDZ >= 1, and if
 *                        JOBZ = 'V', LDZ >= max(1,N).
 * @param[out]    work   Double precision array, dimension (LWORK).
 *                        On exit, if INFO = 0, WORK(0) returns the optimal LWORK.
 * @param[in]     lwork  The dimension of the array WORK.
 *                        If JOBZ = 'N' or N <= 1 then LWORK must be at least 1.
 *                        If JOBZ = 'V' and N > 1 then LWORK must be at least
 *                        (1 + 4*N + N**2).
 *                        If LWORK = -1, then a workspace query is assumed; the
 *                        routine only calculates the optimal sizes of the WORK
 *                        and IWORK arrays, returns these values as the first
 *                        entries of the WORK and IWORK arrays, and no error
 *                        message related to LWORK or LIWORK is issued by XERBLA.
 * @param[out]    iwork  Integer array, dimension (MAX(1,LIWORK)).
 *                        On exit, if INFO = 0, IWORK(0) returns the optimal LIWORK.
 * @param[in]     liwork The dimension of the array IWORK.
 *                        If JOBZ = 'N' or N <= 1 then LIWORK must be at least 1.
 *                        If JOBZ = 'V' and N > 1 then LIWORK must be at least
 *                        3+5*N.
 *                        If LIWORK = -1, then a workspace query is assumed; the
 *                        routine only calculates the optimal sizes of the WORK
 *                        and IWORK arrays, returns these values as the first
 *                        entries of the WORK and IWORK arrays, and no error
 *                        message related to LWORK or LIWORK is issued by XERBLA.
 * @param[out]    info
 *                         - = 0:  successful exit
 *                         - < 0:  if INFO = -i, the i-th argument had an illegal value
 *                         - > 0:  if INFO = i, the algorithm failed to converge; i
 *                           off-diagonal elements of E did not converge to zero.
 */
void sstevd(const char* jobz, const int n, f32* D, f32* E,
            f32* Z, const int ldz, f32* work, const int lwork,
            int* iwork, const int liwork, int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int wantz, lquery;
    int iscale, liwmin, lwmin;
    f32 bignum, eps, rmax, rmin, safmin, sigma, smlnum, tnrm;

    /* Test the input parameters. */
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    liwmin = 1;
    lwmin = 1;
    if (n > 1 && wantz) {
        lwmin = 1 + 4 * n + n * n;
        liwmin = 3 + 5 * n;
    }

    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -6;
    }

    if (*info == 0) {
        work[0] = (f32)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -8;
        } else if (liwork < liwmin && !lquery) {
            *info = -10;
        }
    }

    if (*info != 0) {
        xerbla("SSTEVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    if (n == 1) {
        if (wantz) {
            Z[0] = ONE;
        }
        return;
    }

    /* Get machine constants. */
    safmin = slamch("S");
    eps = slamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    /* Scale matrix to allowable range, if necessary. */
    iscale = 0;
    tnrm = slanst("M", n, D, E);
    if (tnrm > ZERO && tnrm < rmin) {
        iscale = 1;
        sigma = rmin / tnrm;
    } else if (tnrm > rmax) {
        iscale = 1;
        sigma = rmax / tnrm;
    }
    if (iscale == 1) {
        cblas_sscal(n, sigma, D, 1);
        cblas_sscal(n - 1, sigma, E, 1);
    }

    /* For eigenvalues only, call SSTERF.  For eigenvalues and
       eigenvectors, call SSTEDC. */
    if (!wantz) {
        ssterf(n, D, E, info);
    } else {
        sstedc("I", n, D, E, Z, ldz, work, lwork, iwork, liwork, info);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        cblas_sscal(n, ONE / sigma, D, 1);
    }

    work[0] = (f32)lwmin;
    iwork[0] = liwmin;
}
