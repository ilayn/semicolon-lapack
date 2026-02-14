/**
 * @file dstev.c
 * @brief DSTEV computes all eigenvalues and, optionally, eigenvectors of a
 *        real symmetric tridiagonal matrix A.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSTEV computes all eigenvalues and, optionally, eigenvectors of a
 * real symmetric tridiagonal matrix A.
 *
 * @param[in]     jobz  = 'N': Compute eigenvalues only;
 *                        = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     n     The order of the matrix. n >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the n diagonal elements of the tridiagonal
 *                      matrix A. On exit, if info = 0, the eigenvalues in
 *                      ascending order.
 * @param[in,out] E     Double precision array, dimension (n-1).
 *                      On entry, the (n-1) subdiagonal elements of the
 *                      tridiagonal matrix A. On exit, the contents of E are
 *                      destroyed.
 * @param[out]    Z     Double precision array, dimension (ldz, n).
 *                      If jobz = 'V', then if info = 0, Z contains the
 *                      orthonormal eigenvectors of the matrix A, with the
 *                      i-th column of Z holding the eigenvector associated
 *                      with D(i). If jobz = 'N', then Z is not referenced.
 * @param[in]     ldz   The leading dimension of the array Z. ldz >= 1, and
 *                      if jobz = 'V', ldz >= max(1,n).
 * @param[out]    work  Double precision array, dimension (max(1,2*n-2)).
 *                      If jobz = 'N', work is not referenced.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal
 *                           value
 *                         - > 0: if info = i, the algorithm failed to converge;
 *                           i off-diagonal elements of E did not converge to
 *                           zero.
 */
void dstev(const char* jobz, const int n,
           f64* restrict D, f64* restrict E,
           f64* restrict Z, const int ldz,
           f64* restrict work, int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    /* Test the input parameters. */
    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');

    *info = 0;
    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -6;
    }

    if (*info != 0) {
        xerbla("DSTEV", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    if (n == 1) {
        if (wantz)
            Z[0] = ONE;
        return;
    }

    /* Get machine constants. */
    f64 safmin = dlamch("S");
    f64 eps = dlamch("P");
    f64 smlnum = safmin / eps;
    f64 bignum = ONE / smlnum;
    f64 rmin = sqrt(smlnum);
    f64 rmax = sqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */
    int iscale = 0;
    f64 tnrm = dlanst("M", n, D, E);
    f64 sigma = ZERO;
    if (tnrm > ZERO && tnrm < rmin) {
        iscale = 1;
        sigma = rmin / tnrm;
    } else if (tnrm > rmax) {
        iscale = 1;
        sigma = rmax / tnrm;
    }
    if (iscale == 1) {
        cblas_dscal(n, sigma, D, 1);
        cblas_dscal(n - 1, sigma, E, 1);
    }

    /* For eigenvalues only, call DSTERF.  For eigenvalues and
     * eigenvectors, call DSTEQR. */
    if (!wantz) {
        dsterf(n, D, E, info);
    } else {
        dsteqr("I", n, D, E, Z, ldz, work, info);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */
    if (iscale == 1) {
        int imax;
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, D, 1);
    }
}
