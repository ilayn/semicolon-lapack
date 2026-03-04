/**
 * @file ztrt02.c
 * @brief ZTRT02 computes the residual for triangular solve.
 *
 * Port of LAPACK TESTING/LIN/ztrt02.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * ZTRT02 computes the residual for the computed solution to a
 * triangular system of linear equations op(A)*X = B, where A is a
 * triangular matrix. The test ratio is the maximum over
 *    norm(b - op(A)*x) / (||op(A)||_1 * norm(x) * EPS),
 * where op(A) = A, A**T, or A**H.
 *
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     trans   = 'N': A*X = B; = 'T': A**T*X = B; = 'C': A**H*X = B.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of right hand sides.
 * @param[in]     A       Array (lda, n). The triangular matrix A.
 * @param[in]     lda     The leading dimension of A.
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X.
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B.
 * @param[out]    work    Array (n). Complex workspace.
 * @param[out]    rwork   Array (n). Real workspace.
 * @param[out]    resid   max over NRHS of norm(op(A)*X - B) / (norm(op(A)) * norm(X) * EPS).
 */
void ztrt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const c128* A, const INT lda,
            const c128* X, const INT ldx,
            const c128* B, const INT ldb,
            c128* work, f64* rwork, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CNEGONE = CMPLX(-1.0, 0.0);
    INT j;
    f64 anorm, bnorm, eps, xnorm;

    /* Quick exit if N = 0 or NRHS = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    /* Compute the 1-norm of op(A) */
    if (trans[0] == 'N' || trans[0] == 'n') {
        anorm = zlantr("1", uplo, diag, n, n, A, lda, rwork);
    } else {
        anorm = zlantr("I", uplo, diag, n, n, A, lda, rwork);
    }

    /* Exit with RESID = 1/EPS if ANORM = 0 */
    eps = dlamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute the maximum over the number of right hand sides of
       norm(op(A)*X - B) / (norm(op(A)) * norm(X) * EPS) */
    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        /* Copy X(:,j) to work */
        cblas_zcopy(n, &X[j * ldx], 1, work, 1);

        /* work = op(A) * X(:,j) */
        CBLAS_UPLO uplo_cblas = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE trans_cblas;
        if (trans[0] == 'N' || trans[0] == 'n')
            trans_cblas = CblasNoTrans;
        else if (trans[0] == 'T' || trans[0] == 't')
            trans_cblas = CblasTrans;
        else
            trans_cblas = CblasConjTrans;
        CBLAS_DIAG diag_cblas = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;
        cblas_ztrmv(CblasColMajor, uplo_cblas, trans_cblas, diag_cblas, n, A, lda, work, 1);

        /* work = work - B(:,j) = op(A)*X(:,j) - B(:,j) */
        cblas_zaxpy(n, &CNEGONE, &B[j * ldb], 1, work, 1);

        /* Compute norms */
        bnorm = cblas_dzasum(n, work, 1);
        xnorm = cblas_dzasum(n, &X[j * ldx], 1);

        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f64 ratio = ((bnorm / anorm) / xnorm) / eps;
            if (ratio > *resid) {
                *resid = ratio;
            }
        }
    }
}
