/**
 * @file dtpt03.c
 * @brief DTPT03 computes the residual for the solution to a scaled triangular
 *        system of equations when A is in packed format.
 *
 * Port of LAPACK TESTING/LIN/dtpt03.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * DTPT03 computes the residual for the solution to a scaled triangular
 * system of equations A*x = s*b or A'*x = s*b when the triangular
 * matrix A is stored in packed format. The test ratio is the maximum
 * over the number of right hand sides of
 *    norm(s*b - op(A)*x) / (norm(op(A)) * norm(x) * EPS),
 * where op(A) denotes A or A'.
 *
 * @param[in]     uplo    = 'U': Upper triangular; = 'L': Lower triangular.
 * @param[in]     trans   = 'N': A*x = s*b; = 'T' or 'C': A'*x = s*b.
 * @param[in]     diag    = 'N': Non-unit triangular; = 'U': Unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     AP      Array (n*(n+1)/2). The triangular matrix A in packed storage.
 * @param[in]     scale   The scaling factor s used in solving the triangular system.
 * @param[in]     cnorm   Array (n). The 1-norms of the columns of A, not counting diagonal.
 * @param[in]     tscal   The scaling factor used in computing the 1-norms in cnorm.
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1, n).
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    resid   The maximum over NRHS of norm(op(A)*x - s*b) / (norm(op(A)) * norm(x) * EPS).
 */
void dtpt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs,
            const f64* AP, const f64 scale, const f64* cnorm,
            const f64 tscal, const f64* X, const INT ldx,
            const f64* B, const INT ldb,
            f64* work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    INT ix, j, jj;
    f64 eps, err, smlnum, tnorm, xnorm, xscal;

    /* Quick exit if N = 0 or NRHS = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    smlnum = dlamch("S");

    /* Compute the norm of the triangular matrix A using the column
     * norms already computed by DLATPS */
    tnorm = ZERO;
    if (diag[0] == 'N' || diag[0] == 'n') {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            jj = 0;
            for (j = 0; j < n; j++) {
                f64 t = tscal * fabs(AP[jj]) + cnorm[j];
                if (t > tnorm) tnorm = t;
                jj += j + 2;
            }
        } else {
            jj = 0;
            for (j = 0; j < n; j++) {
                f64 t = tscal * fabs(AP[jj]) + cnorm[j];
                if (t > tnorm) tnorm = t;
                jj += n - j;
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            f64 t = tscal + cnorm[j];
            if (t > tnorm) tnorm = t;
        }
    }

    /* Compute the maximum over the number of right hand sides of
     * norm(op(A)*x - s*b) / (norm(op(A)) * norm(x) * EPS) */
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
        ix = cblas_idamax(n, work, 1);
        xnorm = fmax(ONE, fabs(X[ix + j * ldx]));
        xscal = (ONE / xnorm) / (f64)n;
        cblas_dscal(n, xscal, work, 1);
        cblas_dtpmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                   n, AP, work, 1);
        cblas_daxpy(n, -scale * xscal, &B[j * ldb], 1, work, 1);
        ix = cblas_idamax(n, work, 1);
        err = tscal * fabs(work[ix]);
        ix = cblas_idamax(n, &X[j * ldx], 1);
        xnorm = fabs(X[ix + j * ldx]);
        if (err * smlnum <= xnorm) {
            if (xnorm > ZERO) {
                err = err / xnorm;
            }
        } else {
            if (err > ZERO) {
                err = ONE / eps;
            }
        }
        if (err * smlnum <= tnorm) {
            if (tnorm > ZERO) {
                err = err / tnorm;
            }
        } else {
            if (err > ZERO) {
                err = ONE / eps;
            }
        }
        if (err > *resid) {
            *resid = err;
        }
    }
}
