/**
 * @file zhbev.c
 * @brief ZHBEV computes all the eigenvalues and, optionally, eigenvectors of a
 *        complex Hermitian band matrix.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZHBEV computes all the eigenvalues and, optionally, eigenvectors of
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
 * @param[out]    W     Double precision array, dimension (n).
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
 * @param[out]    rwork Double precision array, dimension (max(1, 3*n-2)).
 * @param[out]    info  = 0: successful exit.
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 *                      > 0: if info = i, the algorithm failed to converge; i
 *                           off-diagonal elements of an intermediate tridiagonal
 *                           form did not converge to zero.
 */
void zhbev(
    const char* jobz,
    const char* uplo,
    const int n,
    const int kd,
    double complex* const restrict AB,
    const int ldab,
    double* const restrict W,
    double complex* const restrict Z,
    const int ldz,
    double complex* const restrict work,
    double* const restrict rwork,
    int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    int lower = (uplo[0] == 'L' || uplo[0] == 'l');

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
        xerbla("ZHBEV ", -(*info));
        return;
    }

    /* Quick return if possible */

    if (n == 0)
        return;

    if (n == 1) {
        if (lower) {
            W[0] = creal(AB[0]);
        } else {
            W[0] = creal(AB[kd]);
        }
        if (wantz)
            Z[0] = CMPLX(ONE, 0.0);
        return;
    }

    /* Get machine constants. */

    double safmin = dlamch("Safe minimum");
    double eps = dlamch("Precision");
    double smlnum = safmin / eps;
    double bignum = ONE / smlnum;
    double rmin = sqrt(smlnum);
    double rmax = sqrt(bignum);

    /* Scale matrix to allowable range, if necessary. */

    double anrm = zlanhb("M", uplo, n, kd, AB, ldab, rwork);
    int iscale = 0;
    double sigma;
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

    /* Call ZHBTRD to reduce Hermitian band matrix to tridiagonal form. */

    int inde = 0;
    int iinfo;
    zhbtrd(jobz, uplo, n, kd, AB, ldab, W, &rwork[inde], Z, ldz,
           work, &iinfo);

    /* For eigenvalues only, call DSTERF. For eigenvectors, call ZSTEQR. */

    if (!wantz) {
        dsterf(n, W, &rwork[inde], info);
    } else {
        int indrwk = inde + n;
        zsteqr(jobz, n, W, &rwork[inde], Z, ldz, &rwork[indrwk], info);
    }

    /* If matrix was scaled, then rescale eigenvalues appropriately. */

    if (iscale == 1) {
        int imax;
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_dscal(imax, ONE / sigma, W, 1);
    }
}
