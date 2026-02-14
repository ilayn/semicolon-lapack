/**
 * @file dsbgvd.c
 * @brief DSBGVD computes all eigenvalues of a generalized symmetric-definite banded eigenproblem using D&C.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DSBGVD computes all the eigenvalues, and optionally, the eigenvectors
 * of a real generalized symmetric-definite banded eigenproblem, of the
 * form A*x = lambda*B*x. Here A and B are assumed to be symmetric and
 * banded, and B is also positive definite. If eigenvectors are
 * desired, it uses a divide and conquer algorithm.
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
 *                       On exit, the split Cholesky factor S from dpbstf.
 * @param[in]     ldbb   The leading dimension of BB. ldbb >= kb+1.
 * @param[out]    W      The eigenvalues in ascending order. Array of dimension (n).
 * @param[out]    Z      If jobz='V', the eigenvectors. Array of dimension (ldz, n).
 * @param[in]     ldz    The leading dimension of Z.
 *                       ldz >= 1, and if jobz='V', ldz >= max(1,n).
 * @param[out]    work   Workspace array of dimension (max(1,lwork)).
 * @param[in]     lwork  The dimension of work.
 *                       If n <= 1, lwork >= 1.
 *                       If jobz='N' and n > 1, lwork >= 2*n.
 *                       If jobz='V' and n > 1, lwork >= 1 + 5*n + 2*n**2.
 *                       If lwork = -1, workspace query mode.
 * @param[out]    iwork  Integer workspace array of dimension (max(1,liwork)).
 * @param[in]     liwork The dimension of iwork.
 *                       If jobz='N' or n <= 1, liwork >= 1.
 *                       If jobz='V' and n > 1, liwork >= 3 + 5*n.
 *                       If liwork = -1, workspace query mode.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, for i <= n, the algorithm failed to converge
 *                           if info = n + i, dpbstf returned info = i (B not positive definite)
 */
void dsbgvd(
    const char* jobz,
    const char* uplo,
    const int n,
    const int ka,
    const int kb,
    f64* restrict AB,
    const int ldab,
    f64* restrict BB,
    const int ldbb,
    f64* restrict W,
    f64* restrict Z,
    const int ldz,
    f64* restrict work,
    const int lwork,
    int* restrict iwork,
    const int liwork,
    int* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    int lquery, upper, wantz;
    int iinfo, inde, indwk2, indwrk, liwmin, llwrk2, lwmin;
    char vect;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    if (n <= 1) {
        liwmin = 1;
        lwmin = 1;
    } else if (wantz) {
        liwmin = 3 + 5 * n;
        lwmin = 1 + 5 * n + 2 * n * n;
    } else {
        liwmin = 1;
        lwmin = 2 * n;
    }

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

    if (*info == 0) {
        work[0] = (f64)lwmin;
        iwork[0] = liwmin;

        if (lwork < lwmin && !lquery) {
            *info = -14;
        } else if (liwork < liwmin && !lquery) {
            *info = -16;
        }
    }

    if (*info != 0) {
        xerbla("DSBGVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0)
        return;

    // Form a split Cholesky factorization of B
    dpbstf(uplo, n, kb, BB, ldbb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    // Transform problem to standard eigenvalue problem
    inde = 0;
    indwrk = inde + n;
    indwk2 = indwrk + n * n;
    llwrk2 = lwork - indwk2;
    dsbgst(jobz, uplo, n, ka, kb, AB, ldab, BB, ldbb, Z, ldz, work, &iinfo);

    // Reduce to tridiagonal form
    if (wantz) {
        vect = 'U';
    } else {
        vect = 'N';
    }
    dsbtrd(&vect, uplo, n, ka, AB, ldab, W, &work[inde], Z, ldz, &work[indwrk], &iinfo);

    // For eigenvalues only, call dsterf. For eigenvectors, call dstedc.
    if (!wantz) {
        dsterf(n, W, &work[inde], info);
    } else {
        dstedc("I", n, W, &work[inde], &work[indwrk], n, &work[indwk2], llwrk2, iwork, liwork, info);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n, n, n,
                    ONE, Z, ldz, &work[indwrk], n, ZERO, &work[indwk2], n);
        dlacpy("A", n, n, &work[indwk2], n, Z, ldz);
    }

    work[0] = (f64)lwmin;
    iwork[0] = liwmin;
}
