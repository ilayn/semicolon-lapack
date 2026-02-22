/**
 * @file chbev.c
 * @brief CHBEV computes all the eigenvalues and, optionally, eigenvectors of a
 *        complex Hermitian band matrix.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * CHBEV computes all the eigenvalues and, optionally, eigenvectors of
 * a complex Hermitian band matrix A.
 *
 * @param[in]     jobz  = 'N': Compute eigenvalues only;
 *                        = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo  = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     kd    The number of superdiagonals of the matrix A if
 *                      uplo = 'U', or the number of subdiagonals if
 *                      uplo = 'L'. kd >= 0.
 * @param[in,out] AB    Complex array, dimension (ldab, n).
 *                      On entry, the upper or lower triangle of the Hermitian
 *                      band matrix A, stored in the first kd+1 rows of the
 *                      array. The j-th column of A is stored in the j-th
 *                      column of the array AB as follows:
 *                      if uplo = 'U', AB(kd+1+i-j,j) = A(i,j) for max(1,j-kd)<=i<=j;
 *                      if uplo = 'L', AB(1+i-j,j)    = A(i,j) for j<=i<=min(n,j+kd).
 *                      On exit, AB is overwritten by values generated during
 *                      the reduction to tridiagonal form. If uplo = 'U', the
 *                      first superdiagonal and the diagonal of the tridiagonal
 *                      matrix T are returned in rows kd and kd+1 of AB, and
 *                      if uplo = 'L', the diagonal and first subdiagonal of T
 *                      are returned in the first two rows of AB.
 * @param[in]     ldab  The leading dimension of the array AB. ldab >= kd+1.
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
 * @param[out]    work  Complex array, dimension (n).
 * @param[out]    rwork Single precision array, dimension (max(1, 3*n-2)).
 * @param[out]    info  = 0: successful exit.
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 *                      > 0: if info = i, the algorithm failed to converge; i
 *                           off-diagonal elements of an intermediate tridiagonal
 *                           form did not converge to zero.
 */
void chbev(
    const char* jobz,
    const char* uplo,
    const INT n,
    const INT kd,
    c64* restrict AB,
    const INT ldab,
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
    INT lower = (uplo[0] == 'L' || uplo[0] == 'l');

    *info = 0;
    if (!(wantz || jobz[0] == 'N' || jobz[0] == 'n')) {
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

    if (*info != 0) {
        xerbla("CHBEV ", -(*info));
        return;
    }

    /* Quick return if possible */

    if (n == 0)
        return;

    if (n == 1) {
        if (lower) {
            W[0] = crealf(AB[0]);
        } else {
            W[0] = crealf(AB[kd]);
        }
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

    f32 anrm = clanhb("M", uplo, n, kd, AB, ldab, rwork);
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
        if (lower) {
            clascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            clascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
    }

    /* Call CHBTRD to reduce Hermitian band matrix to tridiagonal form. */

    INT inde = 0;
    INT iinfo;
    chbtrd(jobz, uplo, n, kd, AB, ldab, W, &rwork[inde], Z, ldz,
           work, &iinfo);

    /* For eigenvalues only, call SSTERF. For eigenvectors, call CSTEQR. */

    if (!wantz) {
        ssterf(n, W, &rwork[inde], info);
    } else {
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
