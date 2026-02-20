/**
 * @file dtrt03.c
 * @brief DTRT03 computes the residual for a scaled triangular system.
 *
 * Port of LAPACK TESTING/LIN/dtrt03.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f64 dlamch(const char* cmach);

/**
 * DTRT03 computes the residual for the solution to a scaled triangular
 * system of equations A*x = s*b or A'*x = s*b.
 *
 * The test ratio is the maximum over the number of right hand sides of
 *    norm(s*b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
 * where op(A) denotes A or A'.
 *
 * @param[in]     uplo    'U' for upper triangular, 'L' for lower triangular.
 * @param[in]     trans   'N' for A*x = s*b, 'T' or 'C' for A'*x = s*b.
 * @param[in]     diag    'N' for non-unit triangular, 'U' for unit triangular.
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     A       Array (lda, n). The triangular matrix A.
 * @param[in]     lda     The leading dimension of A. lda >= max(1,n).
 * @param[in]     scale   The scaling factor s used in the triangular solve.
 * @param[in]     cnorm   Array (n). The 1-norms of the columns of A (not
 *                        counting the diagonal).
 * @param[in]     tscal   The scaling factor used in computing cnorm.
 *                        cnorm contains the column norms of tscal*A.
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1,n).
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    resid   The maximum residual over all right hand sides.
 */
void dtrt03(const char* uplo, const char* trans, const char* diag,
            const int n, const int nrhs, const f64* A, const int lda,
            const f64 scale, const f64* cnorm, const f64 tscal,
            const f64* X, const int ldx, const f64* B, const int ldb,
            f64* work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int j, ix;
    f64 eps, smlnum, tnorm, xnorm, xscal, err;

    /* Quick exit if n = 0 or nrhs = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    smlnum = dlamch("S");
    /* Note: BIGNUM = ONE / smlnum is computed in Fortran but unused */

    /* Compute the norm of the triangular matrix A using the column
       norms already computed by DLATRS. */
    tnorm = ZERO;
    if (diag[0] == 'N' || diag[0] == 'n') {
        for (j = 0; j < n; j++) {
            f64 val = tscal * fabs(A[j * lda + j]) + cnorm[j];
            if (val > tnorm) tnorm = val;
        }
    } else {
        for (j = 0; j < n; j++) {
            f64 val = tscal + cnorm[j];
            if (val > tnorm) tnorm = val;
        }
    }

    /* Compute the maximum over the number of right hand sides of
       norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ). */
    *resid = ZERO;

    for (j = 0; j < nrhs; j++) {
        /* Copy X(:,j) to work */
        cblas_dcopy(n, &X[j * ldx], 1, work, 1);

        /* Find max element of X(:,j) */
        ix = cblas_idamax(n, work, 1);
        xnorm = fabs(X[j * ldx + ix]);
        if (xnorm < ONE) xnorm = ONE;

        xscal = (ONE / xnorm) / (f64)n;
        cblas_dscal(n, xscal, work, 1);

        /* Compute work = op(A) * work */
        CBLAS_UPLO cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasTrans;
        CBLAS_DIAG cblas_diag = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;

        cblas_dtrmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                    n, A, lda, work, 1);

        /* Compute work = work - scale * xscal * B(:,j) */
        cblas_daxpy(n, -scale * xscal, &B[j * ldb], 1, work, 1);

        /* Find max element of residual */
        ix = cblas_idamax(n, work, 1);
        err = tscal * fabs(work[ix]);

        /* Find max element of X(:,j) for this column */
        ix = cblas_idamax(n, &X[j * ldx], 1);
        xnorm = fabs(X[j * ldx + ix]);

        if (err * smlnum <= xnorm) {
            if (xnorm > ZERO)
                err = err / xnorm;
        } else {
            if (err > ZERO)
                err = ONE / eps;
        }

        if (err * smlnum <= tnorm) {
            if (tnorm > ZERO)
                err = err / tnorm;
        } else {
            if (err > ZERO)
                err = ONE / eps;
        }

        if (err > *resid)
            *resid = err;
    }
}
