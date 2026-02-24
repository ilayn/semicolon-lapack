/**
 * @file stbt03.c
 * @brief STBT03 computes the residual for the solution to a scaled triangular
 *        system of equations when A is a triangular band matrix.
 *
 * Port of LAPACK TESTING/LIN/stbt03.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * STBT03 computes the residual for the solution to a scaled triangular
 * system of equations A*x = s*b or A'*x = s*b when A is a
 * triangular band matrix. Here A' is the transpose of A, s is a scalar,
 * and x and b are N by NRHS matrices. The test ratio is the maximum
 * over the number of right hand sides of
 *    norm(s*b - op(A)*x) / ( norm(op(A)) * norm(x) * EPS ),
 * where op(A) denotes A or A' and EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the matrix A is upper or lower triangular.
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     trans   Specifies the operation applied to A.
 *                        = 'N': A *x = b  (No transpose)
 *                        = 'T': A'*x = b  (Transpose)
 *                        = 'C': A'*x = b  (Conjugate transpose = Transpose)
 * @param[in]     diag    Specifies whether or not the matrix A is unit triangular.
 *                        = 'N': Non-unit triangular
 *                        = 'U': Unit triangular
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kd      The number of superdiagonals or subdiagonals of the
 *                        triangular band matrix A. kd >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     AB      Array (ldab, n). The triangular band matrix A.
 * @param[in]     ldab    The leading dimension of AB. ldab >= kd+1.
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
void stbt03(const char* uplo, const char* trans, const char* diag,
            const INT n, const INT kd, const INT nrhs,
            const f32* AB, const INT ldab,
            const f32 scale, const f32* cnorm, const f32 tscal,
            const f32* X, const INT ldx,
            const f32* B, const INT ldb,
            f32* work, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    INT ix, j;
    f32 eps, err, smlnum, tnorm, xnorm, xscal;

    /* Quick exit if N = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    smlnum = slamch("S");

    /* Compute the norm of the triangular matrix A using the column
     * norms already computed by SLATBS. */
    tnorm = ZERO;
    if (diag[0] == 'N' || diag[0] == 'n') {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            for (j = 0; j < n; j++) {
                f32 t = tscal * fabsf(AB[kd + j * ldab]) + cnorm[j];
                if (t > tnorm) tnorm = t;
            }
        } else {
            for (j = 0; j < n; j++) {
                f32 t = tscal * fabsf(AB[j * ldab]) + cnorm[j];
                if (t > tnorm) tnorm = t;
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            f32 t = tscal + cnorm[j];
            if (t > tnorm) tnorm = t;
        }
    }

    /* Compute the maximum over the number of right hand sides of
     * norm(op(A)*x - s*b) / ( norm(op(A)) * norm(x) * EPS ). */
    *resid = ZERO;

    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'N' || trans[0] == 'n') {
        cblas_trans = CblasNoTrans;
    } else {
        cblas_trans = CblasTrans;
    }
    CBLAS_DIAG cblas_siag = (diag[0] == 'U' || diag[0] == 'u') ? CblasUnit : CblasNonUnit;

    for (j = 0; j < nrhs; j++) {
        cblas_scopy(n, &X[j * ldx], 1, work, 1);
        ix = cblas_isamax(n, work, 1);
        xnorm = fmaxf(ONE, fabsf(X[ix + j * ldx]));
        xscal = (ONE / xnorm) / (f32)(kd + 1);
        cblas_sscal(n, xscal, work, 1);
        cblas_stbmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_siag,
                    n, kd, AB, ldab, work, 1);
        cblas_saxpy(n, -scale * xscal, &B[j * ldb], 1, work, 1);
        ix = cblas_isamax(n, work, 1);
        err = tscal * fabsf(work[ix]);
        ix = cblas_isamax(n, &X[j * ldx], 1);
        xnorm = fabsf(X[ix + j * ldx]);
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
