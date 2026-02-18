/**
 * @file ssygvx.c
 * @brief SSYGVX computes selected eigenvalues and optionally eigenvectors of a
 *        real generalized symmetric-definite eigenproblem.
 */

#include "semicolon_lapack_single.h"
#include "lapack_tuning.h"
#include <cblas.h>

/**
 * SSYGVX computes selected eigenvalues, and optionally, eigenvectors
 * of a real generalized symmetric-definite eigenproblem, of the form
 * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x.
 * Eigenvalues and eigenvectors can be selected by specifying either a
 * range of values or a range of indices for the desired eigenvalues.
 *
 * @param[in]     itype   = 1: A*x = lambda*B*x; = 2: A*B*x = lambda*x; = 3: B*A*x = lambda*x
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': eigenvalues and eigenvectors
 * @param[in]     range   = 'A': all; = 'V': in (vl,vu]; = 'I': il-th through iu-th
 * @param[in]     uplo    = 'U': upper triangles stored; = 'L': lower triangles stored
 * @param[in]     n       The order of matrices A and B. n >= 0.
 * @param[in,out] A       On entry, symmetric matrix A. On exit, destroyed.
 * @param[in]     lda     Leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       On entry, symmetric positive definite B. On exit, Cholesky factor.
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1, n).
 * @param[in]     vl      Lower bound of interval (if range='V').
 * @param[in]     vu      Upper bound of interval (if range='V').
 * @param[in]     il      Index of smallest eigenvalue (if range='I', 0-based).
 * @param[in]     iu      Index of largest eigenvalue (if range='I', 0-based).
 * @param[in]     abstol  Absolute error tolerance for eigenvalues.
 * @param[out]    m       Total number of eigenvalues found.
 * @param[out]    W       Selected eigenvalues in ascending order.
 * @param[out]    Z       Eigenvectors if jobz='V'.
 * @param[in]     ldz     Leading dimension of Z. ldz >= 1, and if jobz='V', ldz >= n.
 * @param[out]    work    Workspace array.
 * @param[in]     lwork   Length of work. lwork >= max(1, 8*n). If -1, workspace query.
 * @param[out]    iwork   Integer workspace array, dimension (5*n).
 * @param[out]    ifail   Indices of eigenvectors that failed to converge.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: SPOTRF/SSYEVX error.
 */
void ssygvx(
    const int itype,
    const char* jobz,
    const char* range,
    const char* uplo,
    const int n,
    f32* restrict A,
    const int lda,
    f32* restrict B,
    const int ldb,
    const f32 vl,
    const f32 vu,
    const int il,
    const int iu,
    const f32 abstol,
    int* m,
    f32* restrict W,
    f32* restrict Z,
    const int ldz,
    f32* restrict work,
    const int lwork,
    int* restrict iwork,
    int* restrict ifail,
    int* info)
{
    const f32 ONE = 1.0f;
    int wantz, upper, alleig, valeig, indeig, lquery;
    int lwkmin, lwkopt, nb;

    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    alleig = (range[0] == 'A' || range[0] == 'a');
    valeig = (range[0] == 'V' || range[0] == 'v');
    indeig = (range[0] == 'I' || range[0] == 'i');
    lquery = (lwork == -1);

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!(alleig || valeig || indeig)) {
        *info = -3;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    } else {
        if (valeig) {
            if (n > 0 && vu <= vl) {
                *info = -11;
            }
        } else if (indeig) {
            if (il < 0 || il > (0 > n - 1 ? 0 : n - 1)) {
                *info = -12;
            } else if (iu < ((n - 1) < il ? (n - 1) : il) || iu > n - 1) {
                *info = -13;
            }
        }
    }
    if (*info == 0) {
        if (ldz < 1 || (wantz && ldz < n)) {
            *info = -18;
        }
    }

    if (*info == 0) {
        lwkmin = (8 * n > 1) ? 8 * n : 1;
        nb = lapack_get_nb("SYTRD");
        lwkopt = ((nb + 3) * n > lwkmin) ? (nb + 3) * n : lwkmin;
        work[0] = (f32)lwkopt;

        if (lwork < lwkmin && !lquery) {
            *info = -20;
        }
    }

    if (*info != 0) {
        xerbla("SSYGVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    *m = 0;
    if (n == 0) {
        return;
    }

    /* Form a Cholesky factorization of B */
    spotrf(uplo, n, B, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    /* Transform problem to standard eigenvalue problem and solve */
    ssygst(itype, uplo, n, A, lda, B, ldb, info);
    ssyevx(jobz, range, uplo, n, A, lda, vl, vu, il, iu, abstol,
           m, W, Z, ldz, work, lwork, iwork, ifail, info);

    if (wantz) {
        /* Backtransform eigenvectors to the original problem */
        if (*info > 0) {
            *m = *info - 1;
        }
        if (itype == 1 || itype == 2) {
            /* x = inv(L)**T*y or inv(U)*y */
            if (upper) {
                cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                            CblasNonUnit, n, *m, ONE, B, ldb, Z, ldz);
            } else {
                cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                            CblasNonUnit, n, *m, ONE, B, ldb, Z, ldz);
            }
        } else if (itype == 3) {
            /* x = L*y or U**T*y */
            if (upper) {
                cblas_strmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                            CblasNonUnit, n, *m, ONE, B, ldb, Z, ldz);
            } else {
                cblas_strmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, n, *m, ONE, B, ldb, Z, ldz);
            }
        }
    }

    work[0] = (f32)lwkopt;
}
