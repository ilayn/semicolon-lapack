/**
 * @file strt03.c
 * @brief STRT03 computes the residual for a scaled triangular system.
 *
 * Port of LAPACK TESTING/LIN/strt03.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * STRT03 computes the residual for the solution to a scaled triangular
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
void strt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT nrhs, const f32* A, const INT lda,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const f32* X, const INT ldx, const f32* B, const INT ldb,
            f32* work, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT j, ix;
    f32 eps, smlnum, tnorm, xnorm, xscal, err;

    /* Quick exit if n = 0 or nrhs = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    smlnum = slamch("S");
    /* Note: BIGNUM = ONE / smlnum is computed in Fortran but unused */

    /* Compute the norm of the triangular matrix A using the column
       norms already computed by SLATRS. */
    tnorm = ZERO;
    if (diag[0] == 'N' || diag[0] == 'n') {
        for (j = 0; j < n; j++) {
            f32 val = tscal * fabsf(A[j * lda + j]) + cnorm[j];
            if (val > tnorm) tnorm = val;
        }
    } else {
        for (j = 0; j < n; j++) {
            f32 val = tscal + cnorm[j];
            if (val > tnorm) tnorm = val;
        }
    }

    /* Compute the maximum over the number of right hand sides of
       norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ). */
    *resid = ZERO;

    for (j = 0; j < nrhs; j++) {
        /* Copy X(:,j) to work */
        cblas_scopy(n, &X[j * ldx], 1, work, 1);

        /* Find max element of X(:,j) */
        ix = cblas_isamax(n, work, 1);
        xnorm = fabsf(X[j * ldx + ix]);
        if (xnorm < ONE) xnorm = ONE;

        xscal = (ONE / xnorm) / (f32)n;
        cblas_sscal(n, xscal, work, 1);

        /* Compute work = op(A) * work */
        CBLAS_UPLO cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
        CBLAS_TRANSPOSE cblas_trans = (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasTrans;
        CBLAS_DIAG cblas_siag = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;

        cblas_strmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_siag,
                    n, A, lda, work, 1);

        /* Compute work = work - scale * xscal * B(:,j) */
        cblas_saxpy(n, -scale * xscal, &B[j * ldb], 1, work, 1);

        /* Find max element of residual */
        ix = cblas_isamax(n, work, 1);
        err = tscal * fabsf(work[ix]);

        /* Find max element of X(:,j) for this column */
        ix = cblas_isamax(n, &X[j * ldx], 1);
        xnorm = fabsf(X[j * ldx + ix]);

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
