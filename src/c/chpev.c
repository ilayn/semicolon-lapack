/**
 * @file chpev.c
 * @brief CHPEV computes all the eigenvalues and, optionally, eigenvectors of a
 *        complex Hermitian matrix in packed storage.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"

/**
 * CHPEV computes all the eigenvalues and, optionally, eigenvectors of a
 * complex Hermitian matrix in packed storage.
 *
 * @param[in]     jobz  = 'N': Compute eigenvalues only;
 *                        = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo  = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in,out] AP    Complex array, dimension (n*(n+1)/2).
 *                      On entry, the upper or lower triangle of the Hermitian
 *                      matrix A, packed columnwise in a linear array.
 *                      The j-th column of A is stored in the array AP as
 *                      follows:
 *                      if uplo = 'U', AP(i + (j-1)*j/2) = A(i,j) for 0<=i<=j;
 *                      if uplo = 'L', AP(i + (j-1)*(2*n-j)/2) = A(i,j) for j<=i<n.
 *                      On exit, AP is overwritten by values generated during
 *                      the reduction to tridiagonal form.
 * @param[out]    W     Single precision array, dimension (n).
 *                      If info = 0, the eigenvalues in ascending order.
 * @param[out]    Z     Complex array, dimension (ldz, n).
 *                      If jobz = 'V', then if info = 0, Z contains the
 *                      orthonormal eigenvectors of the matrix A, with the
 *                      i-th column of Z holding the eigenvector associated
 *                      with W(i).
 *                      If jobz = 'N', then Z is not referenced.
 * @param[in]     ldz   The leading dimension of the array Z. ldz >= 1, and if
 *                      jobz = 'V', ldz >= max(1, n).
 * @param[out]    work  Complex array, dimension (max(1, 2*n-1)).
 * @param[out]    rwork Single precision array, dimension (max(1, 3*n-2)).
 * @param[out]    info  = 0: successful exit.
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 *                      > 0: if info = i, the algorithm failed to converge; i
 *                           off-diagonal elements of an intermediate tridiagonal
 *                           form did not converge to zero.
 */
void chpev(
    const char* jobz,
    const char* uplo,
    const INT n,
    c64* restrict AP,
    f32* restrict W,
    c64* restrict Z,
    const INT ldz,
    c64* restrict work,
    f32* restrict rwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT wantz = (jobz[0] == 'V' || jobz[0] == 'v');

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

    if (*info != 0) {
        xerbla("CHPEV", -(*info));
        return;
    }

    /* Quick return if possible */

    if (n == 0)
        return;

    if (n == 1) {
        W[0] = crealf(AP[0]);
        rwork[0] = 1;
        if (wantz)
            Z[0] = CMPLXF(ONE, 0.0f);
        return;
    }

    /* Get machine constants. */

    f32 safmin = slamch("Safe minimum");
    f32 eps = slamch("Precision");
    f32 smlnum = safmin / eps;
    f32 bignum = ONE / smlnum;
    f32 rmin = sqrtf(smlnum);
    f32 rmax = sqrtf(bignum);

    /* Scale matrix to allowable range, if necessary. */

    f32 anrm = clanhp("M", uplo, n, AP, rwork);
    INT iscale = 0;
    f32 sigma;
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

    INT inde = 0;
    INT indtau = 0;
    INT iinfo;
    chptrd(uplo, n, AP, W, &rwork[inde], &work[indtau], &iinfo);

    /*
     * For eigenvalues only, call SSTERF. For eigenvectors, first call
     * CUPGTR to generate the orthogonal matrix, then call CSTEQR.
     */

    if (!wantz) {
        ssterf(n, W, &rwork[inde], info);
    } else {
        INT indwrk = indtau + n;
        cupgtr(uplo, n, AP, &work[indtau], Z, ldz, &work[indwrk], &iinfo);
        INT indrwk = inde + n;
        csteqr(jobz, n, W, &rwork[inde], Z, ldz, &rwork[indrwk], info);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
        INT imax;
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }
}
