/**
 * @file dpot02.c
 * @brief DPOT02 computes the residual for the solution of a symmetric system.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DPOT02 computes the residual for the solution of a symmetric system
 * of linear equations A*x = b:
 *
 *    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS ),
 *
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     nrhs    The number of columns of B. nrhs >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original symmetric matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     X       Double precision array, dimension (ldx, nrhs).
 *                        The computed solution vectors.
 * @param[in]     ldx     The leading dimension of the array X. ldx >= max(1,n).
 * @param[in,out] B       Double precision array, dimension (ldb, nrhs).
 *                        On entry, the right hand side vectors.
 *                        On exit, B is overwritten with the difference B - A*X.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    resid   The maximum over the number of right hand sides of
 *                        norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void dpot02(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* const restrict A,
    const INT lda,
    const f64* const restrict X,
    const INT ldx,
    f64* const restrict B,
    const INT ldb,
    f64* const restrict rwork,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    // Quick exit if n = 0 or nrhs = 0
    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    // Exit with RESID = 1/EPS if ANORM = 0
    f64 eps = dlamch("E");
    f64 anorm = dlansy("1", uplo, n, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    // Compute B - A*X
    // DSYMM('Left', UPLO, N, NRHS, -ONE, A, LDA, X, LDX, ONE, B, LDB)
    cblas_dsymm(CblasColMajor, CblasLeft,
                (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower,
                n, nrhs, -ONE, A, lda, X, ldx, ONE, B, ldb);

    // Compute the maximum over the number of right hand sides of
    //   norm(B - A*X) / (norm(A) * norm(X) * EPS)
    *resid = ZERO;
    for (INT j = 0; j < nrhs; j++) {
        f64 bnorm = cblas_dasum(n, &B[j * ldb], 1);
        f64 xnorm = cblas_dasum(n, &X[j * ldx], 1);
        if (xnorm <= ZERO) {
            *resid = ONE / eps;
        } else {
            f64 tmp = ((bnorm / anorm) / xnorm) / eps;
            if (tmp > *resid) *resid = tmp;
        }
    }
}
