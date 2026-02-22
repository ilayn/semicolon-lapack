/**
 * @file ssbev.c
 * @brief SSBEV computes all eigenvalues and, optionally, eigenvectors of a symmetric band matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSBEV computes all the eigenvalues and, optionally, eigenvectors of
 * a real symmetric band matrix A.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only
 *                        = 'V': Compute eigenvalues and eigenvectors
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 *                       On exit, overwritten by values generated during reduction.
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[out]    W      The eigenvalues in ascending order. Array of dimension (n).
 * @param[out]    Z      If jobz='V', the orthonormal eigenvectors.
 *                       Array of dimension (ldz, n).
 * @param[in]     ldz    The leading dimension of Z. ldz >= 1, and >= n if jobz='V'.
 * @param[out]    work   Workspace array of dimension (max(1, 3*n-2)).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the algorithm failed to converge
 */
void ssbev(
    const char* jobz,
    const char* uplo,
    const INT n,
    const INT kd,
    f32* restrict AB,
    const INT ldab,
    f32* restrict W,
    f32* restrict Z,
    const INT ldz,
    f32* restrict work,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT lower, wantz;
    INT iinfo, imax, inde, indwrk, iscale;
    f32 anrm, bignum, eps, rmax, rmin, safmin, sigma, smlnum;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    lower = (uplo[0] == 'L' || uplo[0] == 'l');

    *info = 0;
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

    if (*info != 0) {
        xerbla("SSBEV ", -(*info));
        return;
    }

    if (n == 0)
        return;

    if (n == 1) {
        if (lower) {
            W[0] = AB[0 + 0 * ldab];
        } else {
            W[0] = AB[kd + 0 * ldab];
        }
        if (wantz)
            Z[0 + 0 * ldz] = ONE;
        return;
    }

    // Get machine constants
    safmin = slamch("S");
    eps = slamch("P");
    smlnum = safmin / eps;
    bignum = ONE / smlnum;
    rmin = sqrtf(smlnum);
    rmax = sqrtf(bignum);

    // Scale matrix to allowable range, if necessary
    anrm = slansb("M", uplo, n, kd, AB, ldab, work);
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
            slascl("B", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        } else {
            slascl("Q", kd, kd, ONE, sigma, n, n, AB, ldab, info);
        }
    }

    // Call SSBTRD to reduce symmetric band matrix to tridiagonal form
    inde = 0;
    indwrk = inde + n;
    ssbtrd(jobz, uplo, n, kd, AB, ldab, W, &work[inde], Z, ldz, &work[indwrk], &iinfo);

    // For eigenvalues only, call SSTERF. For eigenvectors, call SSTEQR.
    if (!wantz) {
        ssterf(n, W, &work[inde], info);
    } else {
        ssteqr(jobz, n, W, &work[inde], Z, ldz, &work[indwrk], info);
    }

    // If matrix was scaled, then rescale eigenvalues appropriately
    if (iscale == 1) {
        if (*info == 0) {
            imax = n;
        } else {
            imax = *info - 1;
        }
        cblas_sscal(imax, ONE / sigma, W, 1);
    }
}
