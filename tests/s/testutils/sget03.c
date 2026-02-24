/**
 * @file sget03.c
 * @brief SGET03 computes the residual for a general matrix times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * SGET03 computes the residual for a general matrix times its inverse:
 *    norm( I - AINV*A ) / ( N * norm(A) * norm(AINV) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     n       The number of rows and columns of the matrix A.
 *                        n >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original n x n matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in]     AINV    Double precision array, dimension (ldainv, n).
 *                        The inverse of the matrix A.
 * @param[in]     ldainv  The leading dimension of the array AINV.
 *                        ldainv >= max(1,n).
 * @param[out]    work    Double precision array, dimension (ldwork, n).
 * @param[in]     ldwork  The leading dimension of the array work.
 *                        ldwork >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    rcond   The reciprocal of the condition number of A, computed
 *                        as ( 1/norm(A) ) / norm(AINV).
 * @param[out]    resid   norm(I - AINV*A) / ( N * norm(A) * norm(AINV) * EPS )
 */
void sget03(
    const INT n,
    const f32 * const restrict A,
    const INT lda,
    const f32 * const restrict AINV,
    const INT ldainv,
    f32 * const restrict work,
    const INT ldwork,
    f32 * const restrict rwork,
    f32 *rcond,
    f32 *resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT i;
    f32 ainvnm, anorm, eps;

    // Quick exit if n = 0
    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    // Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0
    eps = slamch("E");
    anorm = slange("1", n, n, A, lda, rwork);
    ainvnm = slange("1", n, n, AINV, ldainv, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    // Compute I - AINV * A
    // First compute: work = -AINV * A
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, -ONE, AINV, ldainv, A, lda, ZERO, work, ldwork);

    // Add identity: work = I + work = I - AINV*A
    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = ONE + work[i + i * ldwork];
    }

    // Compute norm(I - AINV*A) / (N * norm(A) * norm(AINV) * EPS)
    *resid = slange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f32)n;
}
