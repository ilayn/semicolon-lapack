/**
 * @file ssbgv.c
 * @brief SSBGV computes all eigenvalues of a generalized symmetric-definite banded eigenproblem.
 */

#include "semicolon_lapack_single.h"

/**
 * SSBGV computes all the eigenvalues, and optionally, the eigenvectors
 * of a real generalized symmetric-definite banded eigenproblem, of
 * the form A*x = lambda*B*x. Here A and B are assumed to be symmetric
 * and banded, and B is also positive definite.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only
 *                        = 'V': Compute eigenvalues and eigenvectors
 * @param[in]     uplo   = 'U': Upper triangles of A and B are stored
 *                        = 'L': Lower triangles of A and B are stored
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in]     ka     The number of superdiagonals (if uplo='U') or
 *                       subdiagonals (if uplo='L') of A. ka >= 0.
 * @param[in]     kb     The number of superdiagonals (if uplo='U') or
 *                       subdiagonals (if uplo='L') of B. kb >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 *                       On exit, contents are destroyed.
 * @param[in]     ldab   The leading dimension of AB. ldab >= ka+1.
 * @param[in,out] BB     The banded matrix B. Array of dimension (ldbb, n).
 *                       On exit, the split Cholesky factor S from spbstf.
 * @param[in]     ldbb   The leading dimension of BB. ldbb >= kb+1.
 * @param[out]    W      The eigenvalues in ascending order. Array of dimension (n).
 * @param[out]    Z      If jobz='V', the eigenvectors. Array of dimension (ldz, n).
 * @param[in]     ldz    The leading dimension of Z.
 *                       ldz >= 1, and if jobz='V', ldz >= n.
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, for i <= n, the algorithm failed to converge
 *                           if info = n + i, spbstf returned info = i (B not positive definite)
 */
void ssbgv(
    const char* jobz,
    const char* uplo,
    const int n,
    const int ka,
    const int kb,
    f32* const restrict AB,
    const int ldab,
    f32* const restrict BB,
    const int ldbb,
    f32* const restrict W,
    f32* const restrict Z,
    const int ldz,
    f32* const restrict work,
    int* info)
{
    int upper, wantz;
    int iinfo, inde, indwrk;
    char vect;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    *info = 0;
    if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -1;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (ka < 0) {
        *info = -4;
    } else if (kb < 0 || kb > ka) {
        *info = -5;
    } else if (ldab < ka + 1) {
        *info = -7;
    } else if (ldbb < kb + 1) {
        *info = -9;
    } else if (ldz < 1 || (wantz && ldz < n)) {
        *info = -12;
    }
    if (*info != 0) {
        xerbla("SSBGV", -(*info));
        return;
    }

    if (n == 0)
        return;

    // Form a split Cholesky factorization of B
    spbstf(uplo, n, kb, BB, ldbb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    // Transform problem to standard eigenvalue problem
    inde = 0;
    indwrk = inde + n;
    ssbgst(jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, Z, ldz, &work[indwrk], &iinfo);

    // Reduce to tridiagonal form
    if (wantz) {
        vect = 'U';
    } else {
        vect = 'N';
    }
    ssbtrd(&vect, uplo, n, ka, AB, ldab, W, &work[inde], Z, ldz, &work[indwrk], &iinfo);

    // For eigenvalues only, call ssterf. For eigenvectors, call ssteqr.
    if (!wantz) {
        ssterf(n, W, &work[inde], info);
    } else {
        ssteqr(jobz, n, W, &work[inde], Z, ldz, &work[indwrk], info);
    }
}
