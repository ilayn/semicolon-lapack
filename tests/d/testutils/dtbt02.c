/**
 * @file dtbt02.c
 * @brief DTBT02 computes the residual for the computed solution to a
 *        triangular system of linear equations when A is a triangular band matrix.
 *
 * Port of LAPACK TESTING/LIN/dtbt02.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f64 dlamch(const char* cmach);
extern f64 dlantb(const char* norm, const char* uplo, const char* diag,
                     const int n, const int kd, const f64* AB, const int ldab,
                     f64* work);

/**
 * DTBT02 computes the residual for the computed solution to a
 * triangular system of linear equations op(A)*X = B, when A is a
 * triangular band matrix. The test ratio is the maximum over
 *    norm(b - op(A)*x) / ( ||op(A)||_1 * norm(x) * EPS ),
 * where op(A) = A or A**T, b is the column of B, x is the solution
 * vector, and EPS is the machine epsilon.
 * The norm used is the 1-norm.
 *
 * @param[in]     uplo    Specifies whether the matrix A is upper or lower triangular.
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     trans   Specifies the operation applied to A.
 *                        = 'N': A    * X = B  (No transpose)
 *                        = 'T': A**T * X = B  (Transpose)
 *                        = 'C': A**H * X = B  (Conjugate transpose = Transpose)
 * @param[in]     diag    Specifies whether or not the matrix A is unit triangular.
 *                        = 'N': Non-unit triangular
 *                        = 'U': Unit triangular
 * @param[in]     n       The order of the matrix A. n >= 0.
 * @param[in]     kd      The number of superdiagonals or subdiagonals of the
 *                        triangular band matrix A. kd >= 0.
 * @param[in]     nrhs    The number of right hand sides, i.e., the number of columns
 *                        of the matrices X and B. nrhs >= 0.
 * @param[in]     AB      Array (ldab, n). The upper or lower triangular band matrix A,
 *                        stored in the first kd+1 rows of the array.
 * @param[in]     ldab    The leading dimension of the array AB. ldab >= kd+1.
 * @param[in]     X       Array (ldx, nrhs). The computed solution vectors.
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1, n).
 * @param[in]     B       Array (ldb, nrhs). The right hand side vectors.
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[out]    work    Array (n). Workspace.
 * @param[out]    resid   The maximum over the number of right hand sides of
 *                        norm(op(A)*x - b) / ( norm(op(A)) * norm(x) * EPS ).
 */
void dtbt02(const char* uplo, const char* trans, const char* diag,
            const int n, const int kd, const int nrhs,
            const f64* AB, const int ldab,
            const f64* X, const int ldx,
            const f64* B, const int ldb,
            f64* work, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    int j;
    f64 anorm, bnorm, eps, xnorm;

    /* Quick exit if N = 0 or NRHS = 0 */
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    /* Compute the 1-norm of op(A). */
    if (trans[0] == 'N' || trans[0] == 'n') {
        anorm = dlantb("1", uplo, diag, n, kd, AB, ldab, work);
    } else {
        anorm = dlantb("I", uplo, diag, n, kd, AB, ldab, work);
    }

    /* Exit with RESID = 1/EPS if ANORM = 0. */
    eps = dlamch("E");
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute the maximum over the number of right hand sides of
     * norm(op(A)*x - b) / ( norm(op(A)) * norm(x) * EPS ). */
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
        cblas_dtbmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                    n, kd, AB, ldab, work, 1);
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
