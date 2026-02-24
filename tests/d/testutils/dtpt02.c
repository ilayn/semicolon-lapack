/**
 * @file dtpt02.c
 * @brief DTPT02 computes the residual for the computed solution to a
 *        triangular system of linear equations when A is in packed format.
 *
 * Port of LAPACK TESTING/LIN/dtpt02.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * DTPT02 computes the residual for the computed solution to a
 * triangular system of linear equations op(A)*X = B, when the
 * triangular matrix A is stored in packed format. The test ratio is
 * the maximum over
 *    norm(b - op(A)*x) / (||op(A)||_1 * norm(x) * EPS),
 * where op(A) = A or A**T.
 *
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     trans   = 'N': A * X = B; = 'T' or 'C': A**T * X = B.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     AP      Array (n*(n+1)/2). The triangular matrix A in packed storage.
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1, n).
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    resid   The maximum over NRHS of norm(op(A)*X - B) / (norm(op(A)) * norm(X) * EPS).
 */
void dtpt02(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f64* AP, const f64* X, const INT ldx,
            const f64* B, const INT ldb,
            f64* work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    INT j;
    f64 anorm, bnorm, eps, xnorm;

    /* Quick exit if N = 0 or NRHS = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    /* Compute the 1-norm of op(A) */
    if (trans[0] == 'N' || trans[0] == 'n') {
        anorm = dlantp("1", uplo, diag, n, AP, work);
    } else {
        anorm = dlantp("I", uplo, diag, n, AP, work);
    }

    /* Exit with RESID = 1/EPS if ANORM = 0 */
    eps = dlamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute the maximum over the number of right hand sides of
     * norm(op(A)*X - B) / (norm(op(A)) * norm(X) * EPS). */
    *resid = ZERO;

    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'N' || trans[0] == 'n') {
        cblas_trans = CblasNoTrans;
    } else {
        cblas_trans = CblasTrans;
    }
    CBLAS_DIAG cblas_diag = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;

    for (j = 0; j < nrhs; j++) {
        cblas_dcopy(n, &X[j * ldx], 1, work, 1);
        cblas_dtpmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                   n, AP, work, 1);
        cblas_daxpy(n, -ONE, &B[j * ldb], 1, work, 1);
        bnorm = cblas_dasum(n, work, 1);
        xnorm = cblas_dasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f64 r = ((bnorm / anorm) / xnorm) / eps;
            if (r > *resid) {
                *resid = r;
            }
        }
    }
}
