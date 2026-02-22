/**
 * @file zhbevd.c
 * @brief ZHBEVD computes all eigenvalues and eigenvectors using divide-and-conquer.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZHBEVD computes all the eigenvalues and, optionally, eigenvectors of
 * a complex Hermitian band matrix A. If eigenvectors are desired, it uses
 * a divide and conquer algorithm.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only
 *                        = 'V': Compute eigenvalues and eigenvectors
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    W      The eigenvalues in ascending order. Array of dimension (n).
 * @param[out]    Z      If jobz='V', the orthonormal eigenvectors.
 *                       Array of dimension (ldz, n).
 * @param[in]     ldz    The leading dimension of Z. ldz >= 1, and >= n if jobz='V'.
 * @param[out]    work   Complex workspace array of dimension (lwork).
 * @param[in]     lwork  The dimension of work. If lwork=-1, workspace query.
 * @param[out]    rwork  Double precision workspace array of dimension (lrwork).
 * @param[in]     lrwork The dimension of rwork. If lrwork=-1, workspace query.
 * @param[out]    iwork  Integer workspace array of dimension (liwork).
 * @param[in]     liwork The dimension of iwork. If liwork=-1, workspace query.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the algorithm failed to converge
 */
void zhbevd(
    const char* jobz,
    const char* uplo,
    const INT n,
    const INT kd,
    c128* restrict AB,
    const INT ldab,
    f64* restrict W,
    c128* restrict Z,
    const INT ldz,
    c128* restrict work,
    const INT lwork,
    f64* restrict rwork,
    const INT lrwork,
    INT* restrict iwork,
    const INT liwork,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT lower, lquery, wantz;
    INT iinfo, imax, inde, indwk2, indwrk, iscale;
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
        if (wantz) {
            lwmin = 2 * n * n;
            lrwmin = 1 + 5 * n + 2 * n * n;
            liwmin = 3 + 5 * n;
        } else {
            lwmin = n;
            lrwmin = n;
            liwmin = 1;
        }
    }

    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!lower && !(uplo[0] == 'U' || uplo[0] == 'u')) {
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
        xerbla("ZHBEVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0)
        return;

    if (n == 1) {
        W[0] = creal(AB[0]);
        if (wantz)
            Z[0] = CONE;
        return;
    }

    safmin = dlamch("S");
    eps = dlamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

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

    // Call ZHBTRD to reduce Hermitian band matrix to tridiagonal form.
    inde = 0;
    indwrk = inde + n;
    indwk2 = n * n;
    llwk2 = lwork - indwk2;
    llrwk = lrwork - indwrk;
    zhbtrd(jobz, uplo, n, kd, AB, ldab, W, &rwork[inde], Z, ldz, work, &iinfo);

    if (!wantz) {
        dsterf(n, W, &rwork[inde], info);
    } else {
        zstedc("I", n, W, &rwork[inde], work, n, &work[indwk2], llwk2,
               &rwork[indwrk], llrwk, iwork, liwork, info);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    &CONE, Z, ldz, work, n, &CZERO, &work[indwk2], n);
        zlacpy("A", n, n, &work[indwk2], n, Z, ldz);
    }

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
}
