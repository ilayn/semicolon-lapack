/**
 * @file strt02.c
 * @brief STRT02 computes the residual for triangular solve.
 *
 * Port of LAPACK TESTING/LIN/strt02.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f32 slamch(const char* cmach);
extern f32 slantr(const char* norm, const char* uplo, const char* diag,
                     const int m, const int n, const f32* A, const int lda,
                     f32* work);

/**
 * STRT02 computes the residual for the computed solution to a
 * triangular system of linear equations op(A)*X = B, where A is a
 * triangular matrix. The test ratio is the maximum over
 *    norm(b - op(A)*x) / (||op(A)||_1 * norm(x) * EPS),
 * where op(A) = A or A**T.
 *
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     trans   = 'N': A*X = B (No transpose); = 'T'/'C': A**T*X = B.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of right hand sides.
 * @param[in]     A       Array (lda, n). The triangular matrix A.
 * @param[in]     lda     The leading dimension of A.
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X.
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B.
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    resid   max over NRHS of norm(op(A)*X - B) / (norm(op(A)) * norm(X) * EPS).
 */
void strt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs,
            const f32* A, const int lda,
            const f32* X, const int ldx,
            const f32* B, const int ldb,
            f32* work, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    int j;
    f32 anorm, bnorm, eps, xnorm;

    /* Quick exit if N = 0 or NRHS = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    /* Compute the 1-norm of op(A) */
    if (trans[0] == 'N' || trans[0] == 'n') {
        anorm = slantr("1", uplo, diag, n, n, A, lda, work);
    } else {
        anorm = slantr("I", uplo, diag, n, n, A, lda, work);
    }

    /* Exit with RESID = 1/EPS if ANORM = 0 */
    eps = slamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute the maximum over the number of right hand sides of
       norm(op(A)*X - B) / (norm(op(A)) * norm(X) * EPS) */
    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        /* Copy X(:,j) to work */
        cblas_scopy(n, &X[j * ldx], 1, work, 1);

        /* work = op(A) * X(:,j) */
        CBLAS_UPLO uplo_cblas = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE trans_cblas = (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasTrans;
        CBLAS_DIAG diag_cblas = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;
        cblas_strmv(CblasColMajor, uplo_cblas, trans_cblas, diag_cblas, n, A, lda, work, 1);

        /* work = work - B(:,j) = op(A)*X(:,j) - B(:,j) */
        cblas_saxpy(n, -ONE, &B[j * ldb], 1, work, 1);

        /* Compute norms */
        bnorm = cblas_sasum(n, work, 1);
        xnorm = cblas_sasum(n, &X[j * ldx], 1);

        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f32 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }
}
