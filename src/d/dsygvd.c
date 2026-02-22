/**
 * @file dsygvd.c
 * @brief DSYGVD computes all eigenvalues and optionally eigenvectors of a
 *        real generalized symmetric-definite eigenproblem using divide and conquer.
 */

#include "semicolon_lapack_double.h"
#include "semicolon_cblas.h"

/**
 * DSYGVD computes all the eigenvalues, and optionally, the eigenvectors
 * of a real generalized symmetric-definite eigenproblem, of the form
 * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x.
 * If eigenvectors are desired, it uses a divide and conquer algorithm.
 *
 * @param[in]     itype   = 1: A*x = lambda*B*x; = 2: A*B*x = lambda*x; = 3: B*A*x = lambda*x
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': eigenvalues and eigenvectors
 * @param[in]     uplo    = 'U': upper triangles stored; = 'L': lower triangles stored
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       On entry, symmetric matrix A. On exit, eigenvectors if jobz='V'.
 * @param[in]     lda     Leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       On entry, symmetric positive definite B. On exit, Cholesky factor.
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1, n).
 * @param[out]    W       Eigenvalues in ascending order.
 * @param[out]    work    Workspace array.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    iwork   Integer workspace array.
 * @param[in]     liwork  Length of iwork. If -1, workspace query.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: DPOTRF/DSYEVD error.
 */
void dsygvd(
    const INT itype,
    const char* jobz,
    const char* uplo,
    const INT n,
    f64* restrict A,
    const INT lda,
    f64* restrict B,
    const INT ldb,
    f64* restrict W,
    f64* restrict work,
    const INT lwork,
    INT* restrict iwork,
    const INT liwork,
    INT* info)
{
    const f64 ONE = 1.0;
    INT wantz, upper, lquery;
    INT liwmin, lwmin, lopt, liopt;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1 || liwork == -1);

    *info = 0;
    if (n <= 1) {
        liwmin = 1;
        lwmin = 1;
    } else if (wantz) {
        liwmin = 3 + 5 * n;
        lwmin = 1 + 6 * n + 2 * n * n;
    } else {
        liwmin = 1;
        lwmin = 2 * n + 1;
    }
    lopt = lwmin;
    liopt = liwmin;

    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!wantz && !(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    }

    if (*info == 0) {
        work[0] = (f64)lopt;
        iwork[0] = liopt;

        if (lwork < lwmin && !lquery) {
            *info = -11;
        } else if (liwork < liwmin && !lquery) {
            *info = -13;
        }
    }

    if (*info != 0) {
        xerbla("DSYGVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    /* Form a Cholesky factorization of B */
    dpotrf(uplo, n, B, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    /* Transform problem to standard eigenvalue problem and solve */
    dsygst(itype, uplo, n, A, lda, B, ldb, info);
    dsyevd(jobz, uplo, n, A, lda, W, work, lwork, iwork, liwork, info);

    /* Update optimal workspace sizes */
    lopt = (lopt > (INT)work[0]) ? lopt : (INT)work[0];
    liopt = (liopt > iwork[0]) ? liopt : iwork[0];

    if (wantz && *info == 0) {
        /* Backtransform eigenvectors to the original problem */
        if (itype == 1 || itype == 2) {
            /* x = inv(L)**T*y or inv(U)*y */
            if (upper) {
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                            CblasNonUnit, n, n, ONE, B, ldb, A, lda);
            } else {
                cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans,
                            CblasNonUnit, n, n, ONE, B, ldb, A, lda);
            }
        } else if (itype == 3) {
            /* x = L*y or U**T*y */
            if (upper) {
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans,
                            CblasNonUnit, n, n, ONE, B, ldb, A, lda);
            } else {
                cblas_dtrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, n, n, ONE, B, ldb, A, lda);
            }
        }
    }

    work[0] = (f64)lopt;
    iwork[0] = liopt;
}
