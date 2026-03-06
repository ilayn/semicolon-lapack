/**
 * @file csyt02.c
 * @brief CSYT02 computes the residual for a solution to a complex symmetric
 *        system of linear equations A*x = b.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CSYT02 computes the residual for a solution to a complex symmetric
 * system of linear equations  A*x = b:
 *
 *    RESID = norm(B - A*X) / ( norm(A) * norm(X) * EPS ),
 *
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo   Specifies whether the upper or lower triangular part
 *                       of the symmetric matrix A is stored:
 *                       = 'U': Upper triangular
 *                       = 'L': Lower triangular
 * @param[in]     n      The number of rows and columns of the matrix A.  n >= 0.
 * @param[in]     nrhs   The number of columns of B, the matrix of right hand
 *                       sides.  nrhs >= 0.
 * @param[in]     A      Complex*16 array, dimension (lda, n).
 *                       The original complex symmetric matrix A.
 * @param[in]     lda    The leading dimension of the array A.  lda >= max(1,n).
 * @param[in]     X      Complex*16 array, dimension (ldx, nrhs).
 *                       The computed solution vectors for the system of linear
 *                       equations.
 * @param[in]     ldx    The leading dimension of the array X.  ldx >= max(1,n).
 * @param[in,out] B      Complex*16 array, dimension (ldb, nrhs).
 *                       On entry, the right hand side vectors for the system of
 *                       linear equations.
 *                       On exit, B is overwritten with the difference B - A*X.
 * @param[in]     ldb    The leading dimension of the array B.  ldb >= max(1,n).
 * @param[out]    rwork  Double precision array, dimension (n).
 * @param[out]    resid  The maximum over the number of right hand sides of
 *                       norm(B - A*X) / ( norm(A) * norm(X) * EPS ).
 */
void csyt02(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c64* const restrict A,
    const INT lda,
    const c64* const restrict X,
    const INT ldx,
    c64* const restrict B,
    const INT ldb,
    f32* const restrict rwork,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT j;
    f32 anorm, bnorm, eps, xnorm;

    if (n <= 0 || nrhs <= 0) {
        *resid = ZERO;
        return;
    }

    eps = slamch("E");
    anorm = clansy("1", uplo, n, A, lda, rwork);
    if (anorm <= ZERO) {
        *resid = ONE / eps;
        return;
    }

    /* Compute  B - A*X  and store in B. */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
    cblas_csymm(CblasColMajor, CblasLeft, cblas_uplo, n, nrhs,
                &CNEGONE, A, lda, X, ldx, &CONE, B, ldb);

    *resid = ZERO;
    for (j = 0; j < nrhs; j++) {
        bnorm = cblas_scasum(n, &B[j * ldb], 1);
        xnorm = cblas_scasum(n, &X[j * ldx], 1);
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
