/**
 * @file dsbevd.c
 * @brief DSBEVD computes all eigenvalues and eigenvectors using divide-and-conquer.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSBEVD computes all the eigenvalues and, optionally, eigenvectors of
 * a real symmetric band matrix A. If eigenvectors are desired, it uses
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
 * @param[out]    work   Workspace array of dimension (lwork).
 * @param[in]     lwork  The dimension of work. If lwork=-1, workspace query.
 * @param[out]    iwork  Integer workspace array of dimension (liwork).
 * @param[in]     liwork The dimension of iwork. If liwork=-1, workspace query.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 *                       > 0: if info = i, the algorithm failed to converge
 */
void dsbevd(
    const char* jobz,
    const char* uplo,
    const int n,
    const int kd,
    double* const restrict AB,
    const int ldab,
    double* const restrict W,
    double* const restrict Z,
    const int ldz,
    double* const restrict work,
    const int lwork,
    int* const restrict iwork,
    const int liwork,
    int* info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int lower, lquery, wantz;
    int iinfo, inde, indwk2, indwrk, iscale, liwmin, llwrk2, lwmin;
    double anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    if (n <= 1) {
        liwmin = 1;
        lwmin = 1;
    } else {
        if (wantz) {
            liwmin = 3 + 5 * n;
            lwmin = 1 + 5 * n + 2 * n * n;
        } else {
            liwmin = 1;
            lwmin = 2 * n;
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
        work[0] = (double)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -11;
        } else if (liwork < liwmin && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        xerbla("DSBEVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0)
        return;

    if (n == 1) {
        W[0] = AB[0 + 0 * ldab];
        if (wantz)
            Z[0 + 0 * ldz] = ONE;
        return;
    }

    // Get machine constants
    safmin = dlamch("S");
    eps = dlamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrt(smlnum);
    rmax = sqrt(bignum);

    // Scale matrix to allowable range, if necessary
    anrm = dlansb("M", uplo, n, kd, AB, ldab, work);
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
            dlascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            dlascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
    }

    // Call DSBTRD to reduce symmetric band matrix to tridiagonal form
    inde = 0;
    indwrk = inde + n;
    indwk2 = indwrk + n * n;
    llwrk2 = lwork - indwk2;
    dsbtrd(jobz, uplo, n, kd, AB, ldab, W, &work[inde], Z, ldz, &work[indwrk], &iinfo);

    // For eigenvalues only, call DSTERF. For eigenvectors, call DSTEDC.
    if (!wantz) {
        dsterf(n, W, &work[inde], info);
    } else {
        dstedc("I", n, W, &work[inde], &work[indwrk], n, &work[indwk2], llwrk2, iwork, liwork, info);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n, ONE, Z, ldz,
                    &work[indwrk], n, ZERO, &work[indwk2], n);
        dlacpy("A", n, n, &work[indwk2], n, Z, ldz);
    }

    // If matrix was scaled, then rescale eigenvalues appropriately
    if (iscale == 1)
        cblas_dscal(n, ONE / sigma, W, 1);

    work[0] = (double)lwmin;
    iwork[0] = liwmin;
}
