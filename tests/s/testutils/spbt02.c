/**
 * @file spbt02.c
 * @brief SPBT02 computes the residual for a solution of a symmetric banded
 *        system of equations A*x = b.
 *
 * Port of LAPACK TESTING/LIN/spbt02.f
 */

#include <math.h>
#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SPBT02 computes the residual for a solution of a symmetric banded
 * system of equations A*x = b:
 *    RESID = norm( B - A*X ) / ( norm(A) * norm(X) * EPS )
 * where EPS is the machine precision.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     kd      The number of super-diagonals of the matrix A if
 *                        uplo = 'U', or the number of sub-diagonals if
 *                        uplo = 'L'. kd >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in]     A       The original symmetric band matrix A in band storage.
 *                        Dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= kd+1.
 * @param[in]     X       The computed solution vectors. Dimension (ldx, nrhs).
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1,n).
 * @param[in,out] B       On entry, the right hand side vectors.
 *                        On exit, B is overwritten with the difference B - A*X.
 *                        Dimension (ldb, nrhs).
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1,n).
 * @param[out]    rwork   Workspace array, dimension (n).
 * @param[out]    resid   The maximum over the number of right hand sides of
 *                        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void spbt02(const char* uplo, const INT n, const INT kd, const INT nrhs,
            const f32* A, const INT lda,
            const f32* X, const INT ldx,
            f32* B, const INT ldb,
            f32* rwork, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    *resid = ZERO;
    if (n <= 0 || nrhs <= 0) {
        return;
    }

    f32 eps = slamch("Epsilon");
    f32 anorm = slansb("1", uplo, n, kd, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;

    for (INT j = 0; j < nrhs; j++) {
        cblas_ssbmv(CblasColMajor, cblas_uplo, n, kd, -ONE, A, lda,
                    &X[j * ldx], 1, ONE, &B[j * ldb], 1);
    }

    *resid = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        f32 bnorm = cblas_sasum(n, &B[j * ldb], 1);
        f32 xnorm = cblas_sasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f32 tmp = ((bnorm / anorm) / xnorm) / eps;
            if (tmp > *resid) {
                *resid = tmp;
            }
        }
    }
}
