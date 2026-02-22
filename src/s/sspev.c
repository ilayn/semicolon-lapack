/**
 * @file sspev.c
 * @brief SSPEV computes all the eigenvalues and, optionally, eigenvectors of a
 *        real symmetric matrix A in packed storage.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SSPEV computes all the eigenvalues and, optionally, eigenvectors of a
 * real symmetric matrix A in packed storage.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only;
 *                       = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                       = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in,out] AP     Double precision array, dimension (n*(n+1)/2).
 *                       On entry, the upper or lower triangle of the symmetric
 *                       matrix A, packed columnwise.
 *                       On exit, AP is overwritten by values generated during
 *                       the reduction to tridiagonal form.
 * @param[out]    W      Double precision array, dimension (n).
 *                       If info = 0, the eigenvalues in ascending order.
 * @param[out]    Z      Double precision array, dimension (ldz, n).
 *                       If jobz = 'V', then if info = 0, Z contains the
 *                       orthonormal eigenvectors of the matrix A.
 *                       If jobz = 'N', then Z is not referenced.
 * @param[in]     ldz    The leading dimension of the array Z. ldz >= 1, and if
 *                       jobz = 'V', ldz >= max(1, n).
 * @param[out]    work   Double precision array, dimension (3*n).
 * @param[out]    info
 *                         - = 0: successful exit.
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 *                         - > 0: if info = i, the algorithm failed to converge; i
 *                           off-diagonal elements of an intermediate tridiagonal
 *                           form did not converge to zero.
 */
void sspev(const char* jobz, const char* uplo, const INT n,
           f32* restrict AP, f32* restrict W,
           f32* restrict Z, const INT ldz,
           f32* restrict work, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT wantz;
    INT iinfo, imax, inde, indtau, indwrk, iscale;
    f32 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');

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

    if (*info != 0) {
        xerbla("SSPEV ", -(*info));
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

    safmin = slamch("Safe minimum");
    eps = slamch("Precision");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    /* Scale matrix to allowable range, if necessary. */

    anrm = slansp("M", uplo, n, AP, work);
    iscale = 0;
    if (anrm > ZERO && anrm < rmin) {
        iscale = 1;
        sigma = rmin / anrm;
    } else if (anrm > rmax) {
        iscale = 1;
        sigma = rmax / anrm;
    }
    if (iscale == 1) {
        cblas_sscal((n * (n + 1)) / 2, sigma, AP, 1);
    }

    /* Call SSPTRD to reduce symmetric packed matrix to tridiagonal form. */

    inde = 0;
    indtau = inde + n;
    ssptrd(uplo, n, AP, W, &work[inde], &work[indtau], &iinfo);

    /* For eigenvalues only, call SSTERF.  For eigenvectors, first call
       SOPGTR to generate the orthogonal matrix, then call SSTEQR. */

    if (!wantz) {
        ssterf(n, W, &work[inde], info);
    } else {
        indwrk = indtau + n;
        sopgtr(uplo, n, AP, &work[indtau], Z, ldz, &work[indwrk], &iinfo);
        ssteqr(jobz, n, W, &work[inde], Z, ldz, &work[indtau], info);
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
}
