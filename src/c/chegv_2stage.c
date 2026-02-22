/**
 * @file chegv_2stage.c
 * @brief CHEGV_2STAGE computes all eigenvalues and optionally eigenvectors of a
 *        complex generalized Hermitian-definite eigenproblem using 2-stage reduction.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>

/**
 * CHEGV_2STAGE computes all the eigenvalues, and optionally, the eigenvectors
 * of a complex generalized Hermitian-definite eigenproblem, of the form
 * A*x=(lambda)*B*x, A*B*x=(lambda)*x, or B*A*x=(lambda)*x.
 * Here A and B are assumed to be Hermitian and B is also positive definite.
 * This routine uses the 2-stage technique for the reduction to tridiagonal
 * which showed higher performance on recent architecture and for large sizes N>2000.
 *
 * @param[in]     itype  = 1: A*x = lambda*B*x; = 2: A*B*x = lambda*x; = 3: B*A*x = lambda*x
 * @param[in]     jobz   = 'N': eigenvalues only; = 'V': not available in this release.
 * @param[in]     uplo   = 'U': upper triangles stored; = 'L': lower triangles stored
 * @param[in]     n      The order of the matrices A and B. n >= 0.
 * @param[in,out] A      Hermitian matrix A. On exit, eigenvectors if jobz='V'.
 * @param[in]     lda    Leading dimension of A. lda >= max(1, n).
 * @param[in,out] B      Hermitian positive definite B. On exit, Cholesky factor.
 * @param[in]     ldb    Leading dimension of B. ldb >= max(1, n).
 * @param[out]    W      Eigenvalues in ascending order.
 * @param[out]    work   Workspace. On exit, work[0] = optimal LWORK.
 * @param[in]     lwork  Length of work. If -1, workspace query.
 * @param[out]    rwork  Real workspace, dimension max(1, 3*n-2).
 * @param[out]    info
 *                         - = 0: success; < 0: illegal argument; > 0: CPOTRF/CHEEV error.
 */
void chegv_2stage(
    const INT itype,
    const char* jobz,
    const char* uplo,
    const INT n,
    c64* restrict A,
    const INT lda,
    c64* restrict B,
    const INT ldb,
    f32* restrict W,
    c64* restrict work,
    const INT lwork,
    f32* restrict rwork,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    INT lquery, upper, wantz;
    INT neig, lwmin, lhtrd, lwtrd, kd, ib;

    wantz = (jobz[0] == 'V' || jobz[0] == 'v');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    lquery = (lwork == -1);

    *info = 0;
    if (itype < 1 || itype > 3) {
        *info = -1;
    } else if (!(jobz[0] == 'N' || jobz[0] == 'n')) {
        *info = -2;
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    }

    if (*info == 0) {
        kd = ilaenv2stage(1, "CHETRD_2STAGE", jobz, n, -1, -1, -1);
        ib = ilaenv2stage(2, "CHETRD_2STAGE", jobz, n, kd, -1, -1);
        lhtrd = ilaenv2stage(3, "CHETRD_2STAGE", jobz, n, kd, ib, -1);
        lwtrd = ilaenv2stage(4, "CHETRD_2STAGE", jobz, n, kd, ib, -1);
        lwmin = n + lhtrd + lwtrd;
        work[0] = CMPLXF((f32)lwmin, 0.0f);

        if (lwork < lwmin && !lquery) {
            *info = -11;
        }
    }

    if (*info != 0) {
        xerbla("CHEGV_2STAGE ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (n == 0) {
        return;
    }

    cpotrf(uplo, n, B, ldb, info);
    if (*info != 0) {
        *info = n + *info;
        return;
    }

    chegst(itype, uplo, n, A, lda, B, ldb, info);
    cheev_2stage(jobz, uplo, n, A, lda, W, work, lwork, rwork, info);

    if (wantz) {
        neig = n;
        if (*info > 0) {
            neig = *info - 1;
        }
        if (itype == 1 || itype == 2) {
            if (upper) {
                cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                            CblasNonUnit, n, neig, &ONE, B, ldb, A, lda);
            } else {
                cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                            CblasNonUnit, n, neig, &ONE, B, ldb, A, lda);
            }
        } else if (itype == 3) {
            if (upper) {
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                            CblasNonUnit, n, neig, &ONE, B, ldb, A, lda);
            } else {
                cblas_ctrmm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                            CblasNonUnit, n, neig, &ONE, B, ldb, A, lda);
            }
        }
    }

    work[0] = CMPLXF((f32)lwmin, 0.0f);
}
