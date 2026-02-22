/**
 * @file zhbgv.c
 * @brief ZHBGV computes all eigenvalues and, optionally, eigenvectors of a
 *        complex generalized Hermitian-definite banded eigenproblem.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZHBGV computes all the eigenvalues, and optionally, the eigenvectors
 * of a complex generalized Hermitian-definite banded eigenproblem, of
 * the form A*x=(lambda)*B*x. Here A and B are assumed to be Hermitian
 * and banded, and B is also positive definite.
 *
 * @param[in]     jobz   = 'N': Compute eigenvalues only;
 *                        = 'V': Compute eigenvalues and eigenvectors.
 * @param[in]     uplo   = 'U': Upper triangles of A and B are stored;
 *                        = 'L': Lower triangles of A and B are stored.
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in]     ka     The number of superdiagonals of the matrix A if
 *                       uplo = 'U', or the number of subdiagonals if
 *                       uplo = 'L'. ka >= 0.
 * @param[in]     kb     The number of superdiagonals of the matrix B if
 *                       uplo = 'U', or the number of subdiagonals if
 *                       uplo = 'L'. kb >= 0.
 * @param[in,out] AB     Complex array, dimension (ldab, n).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       band matrix A, stored in the first ka+1 rows.
 *                       On exit, the contents of AB are destroyed.
 * @param[in]     ldab   The leading dimension of AB. ldab >= ka+1.
 * @param[in,out] BB     Complex array, dimension (ldbb, n).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       band matrix B, stored in the first kb+1 rows.
 *                       On exit, the factor S from the split Cholesky
 *                       factorization B = S**H*S, as returned by ZPBSTF.
 * @param[in]     ldbb   The leading dimension of BB. ldbb >= kb+1.
 * @param[out]    W      Double precision array, dimension (n).
 *                       If info = 0, the eigenvalues in ascending order.
 * @param[out]    Z      Complex array, dimension (ldz, n).
 *                       If jobz = 'V', then if info = 0, Z contains the matrix
 *                       Z of eigenvectors. If jobz = 'N', Z is not referenced.
 * @param[in]     ldz    The leading dimension of Z. ldz >= 1, and if
 *                       jobz = 'V', ldz >= n.
 * @param[out]    work   Complex array, dimension (n).
 * @param[out]    rwork  Double precision array, dimension (3*n).
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 *                       > 0: if info = i, and i is:
 *                          <= n: the algorithm failed to converge;
 *                          > n: if info = n + i, for 1 <= i <= n, then ZPBSTF
 *                               returned info = i: B is not positive definite.
 */
void zhbgv(
    const char* jobz,
    const char* uplo,
    const INT n,
    const INT ka,
    const INT kb,
    c128* restrict AB,
    const INT ldab,
    c128* restrict BB,
    const INT ldbb,
    f64* restrict W,
    c128* restrict Z,
    const INT ldz,
    c128* restrict work,
    f64* restrict rwork,
    INT* info)
{
    INT wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');

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
        xerbla("ZHBGV ", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    /* Form a split Cholesky factorization of B. */
    zpbstf(uplo, n, kb, BB, ldbb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    /* Transform problem to standard eigenvalue problem. */
    INT inde = 0;
    INT indwrk = inde + n;
    INT iinfo;
    zhbgst(jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, Z, ldz,
           work, &rwork[indwrk], &iinfo);

    /* Reduce to tridiagonal form. */
    char vect;
    if (wantz) {
        vect = 'U';
    } else {
        vect = 'N';
    }
    zhbtrd(&vect, uplo, n, ka, AB, ldab, W, &rwork[inde], Z, ldz,
           work, &iinfo);

    /* For eigenvalues only, call DSTERF. For eigenvectors, call ZSTEQR. */
    if (!wantz) {
        dsterf(n, W, &rwork[inde], info);
    } else {
        zsteqr(jobz, n, W, &rwork[inde], Z, ldz, &rwork[indwrk], info);
    }
}
