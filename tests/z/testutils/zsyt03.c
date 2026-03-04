/**
 * @file zsyt03.c
 * @brief ZSYT03 computes the residual for a complex symmetric matrix times
 *        its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZSYT03 computes the residual for a complex symmetric matrix times
 * its inverse:
 *    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS )
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the complex symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A.  n >= 0.
 * @param[in]     A       Complex*16 array, dimension (lda, n).
 *                        The original complex symmetric matrix A.
 * @param[in]     lda     The leading dimension of the array A.  lda >= max(1,n).
 * @param[in,out] AINV    Complex*16 array, dimension (ldainv, n).
 *                        On entry, the inverse of the matrix A, stored as a
 *                        symmetric matrix in the same format as A.
 *                        In this version, AINV is expanded into a full matrix and
 *                        multiplied by A, so the opposing triangle of AINV will be
 *                        changed; i.e., if the upper triangular part of AINV is
 *                        stored, the lower triangular part will be used as work space.
 * @param[in]     ldainv  The leading dimension of the array AINV.  ldainv >= max(1,n).
 * @param[out]    work    Complex*16 array, dimension (ldwork, n).
 * @param[in]     ldwork  The leading dimension of the array work.  ldwork >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    rcond   The reciprocal of the condition number of A, computed as
 *                        RCOND = 1/ (norm(A) * norm(AINV)).
 * @param[out]    resid   norm(I - A*AINV) / ( N * norm(A) * norm(AINV) * EPS )
 */
void zsyt03(
    const char* uplo,
    const INT n,
    const c128* const restrict A,
    const INT lda,
    c128* const restrict AINV,
    const INT ldainv,
    c128* const restrict work,
    const INT ldwork,
    f64* const restrict rwork,
    f64* rcond,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT i, j;
    f64 ainvnm, anorm, eps;

    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    anorm = zlansy("1", uplo, n, A, lda, rwork);
    ainvnm = zlansy("1", uplo, n, AINV, ldainv, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    /*
     * Expand AINV into a full matrix and call ZSYMM to multiply
     * AINV on the left by A (store the result in WORK).
     */
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                AINV[j + i * ldainv] = AINV[i + j * ldainv];
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            for (i = j + 1; i < n; i++) {
                AINV[j + i * ldainv] = AINV[i + j * ldainv];
            }
        }
    }

    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
    cblas_zsymm(CblasColMajor, CblasLeft, cblas_uplo, n, n,
                &CNEGONE, A, lda, AINV, ldainv, &CZERO, work, ldwork);

    /* Add the identity matrix to WORK. */
    for (i = 0; i < n; i++) {
        work[i + i * ldwork] += CONE;
    }

    /* Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS) */
    *resid = zlange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f64)n;
}
