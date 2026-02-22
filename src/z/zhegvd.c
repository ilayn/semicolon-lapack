/**
 * @file zhegvd.c
 * @brief ZHEGVD computes all eigenvalues and optionally eigenvectors of a
 *        complex generalized Hermitian-definite eigenproblem using divide and conquer.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZHEGVD computes all the eigenvalues, and optionally, the eigenvectors
 * of a complex generalized Hermitian-definite eigenproblem, of the form
 * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x.
 * Here A and B are assumed to be Hermitian and B is also positive definite.
 * If eigenvectors are desired, it uses a divide and conquer algorithm.
 *
 * @param[in]     itype   = 1: A*x = lambda*B*x; = 2: A*B*x = lambda*x; = 3: B*A*x = lambda*x
 * @param[in]     jobz    = 'N': eigenvalues only; = 'V': eigenvalues and eigenvectors
 * @param[in]     uplo    = 'U': upper triangles stored; = 'L': lower triangles stored
 * @param[in]     n       The order of the matrices A and B. n >= 0.
 * @param[in,out] A       On entry, Hermitian matrix A. On exit, eigenvectors if jobz='V'.
 * @param[in]     lda     Leading dimension of A. lda >= max(1, n).
 * @param[in,out] B       On entry, Hermitian positive definite B. On exit, Cholesky factor.
 * @param[in]     ldb     Leading dimension of B. ldb >= max(1, n).
 * @param[out]    W       Eigenvalues in ascending order.
 * @param[out]    work    Complex workspace array.
 * @param[in]     lwork   Length of work. If -1, workspace query.
 * @param[out]    rwork   Real workspace array.
 * @param[in]     lrwork  Length of rwork. If -1, workspace query.
 * @param[out]    iwork   Integer workspace array.
 * @param[in]     liwork  Length of iwork. If -1, workspace query.
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: ZPOTRF/ZHEEVD error.
 */
void zhegvd(
    const INT itype,
    const char* jobz,
    const char* uplo,
    const INT n,
    c128* restrict A,
    const INT lda,
    c128* restrict B,
    const INT ldb,
    f64* restrict W,
    c128* restrict work,
    const INT lwork,
    f64* restrict rwork,
    const INT lrwork,
    INT* restrict iwork,
    const INT liwork,
    INT* info)
{
    const c128 CONE = CMPLX(1.0, 0.0);
    INT wantz, upper, lquery;
    INT liwmin, lrwmin, lwmin, lopt, lropt, liopt;
    char trans;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1 || lrwork == -1 || liwork == -1);

    *info = 0;
    if (n <= 1) {
        lwmin = 1;
        lrwmin = 1;
        liwmin = 1;
    } else if (wantz) {
        lwmin = 2 * n + n * n;
        lrwmin = 1 + 5 * n + 2 * n * n;
        liwmin = 3 + 5 * n;
    } else {
        lwmin = n + 1;
        lrwmin = n;
        liwmin = 1;
    }
    lopt = lwmin;
    lropt = lrwmin;
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
        work[0] = CMPLX((f64)lopt, 0.0);
        rwork[0] = (f64)lropt;
        iwork[0] = liopt;

        if (lwork < lwmin && !lquery) {
            *info = -11;
        } else if (lrwork < lrwmin && !lquery) {
            *info = -13;
        } else if (liwork < liwmin && !lquery) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("ZHEGVD", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    /* Form a Cholesky factorization of B */
    zpotrf(uplo, n, B, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    /* Transform problem to standard eigenvalue problem and solve */
    zhegst(itype, uplo, n, A, lda, B, ldb, info);
    zheevd(jobz, uplo, n, A, lda, W, work, lwork, rwork, lrwork,
           iwork, liwork, info);

    /* Update optimal workspace sizes */
    lopt = (lopt > (INT)creal(work[0])) ? lopt : (INT)creal(work[0]);
    lropt = (lropt > (INT)rwork[0]) ? lropt : (INT)rwork[0];
    liopt = (liopt > iwork[0]) ? liopt : iwork[0];

    if (wantz && *info == 0) {
        /* Backtransform eigenvectors to the original problem */
        if (itype == 1 || itype == 2) {
            /* x = inv(L)**H*y or inv(U)*y */
            if (upper) {
                trans = 'N';
            } else {
                trans = 'C';
            }

            cblas_ztrsm(CblasColMajor, CblasLeft,
                        upper ? CblasUpper : CblasLower,
                        trans == 'N' ? CblasNoTrans : CblasConjTrans,
                        CblasNonUnit, n, n, &CONE, B, ldb, A, lda);
        } else if (itype == 3) {
            /* x = L*y or U**H*y */
            if (upper) {
                trans = 'C';
            } else {
                trans = 'N';
            }

            cblas_ztrmm(CblasColMajor, CblasLeft,
                        upper ? CblasUpper : CblasLower,
                        trans == 'C' ? CblasConjTrans : CblasNoTrans,
                        CblasNonUnit, n, n, &CONE, B, ldb, A, lda);
        }
    }

    work[0] = CMPLX((f64)lopt, 0.0);
    rwork[0] = (f64)lropt;
    iwork[0] = liopt;
}
