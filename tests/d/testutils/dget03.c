/**
 * @file dget03.c
 * @brief DGET03 computes the residual for a general matrix times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DGET03 computes the residual for a general matrix times its inverse:
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
void dget03(
    const INT n,
    const f64 * const restrict A,
    const INT lda,
    const f64 * const restrict AINV,
    const INT ldainv,
    f64 * const restrict work,
    const INT ldwork,
    f64 * const restrict rwork,
    f64 *rcond,
    f64 *resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i;
    f64 ainvnm, anorm, eps;

    // Quick exit if n = 0
    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    // Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0
    eps = dlamch("E");
    anorm = dlange("1", n, n, A, lda, rwork);
    ainvnm = dlange("1", n, n, AINV, ldainv, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    // Compute I - AINV * A
    // First compute: work = -AINV * A
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, -ONE, AINV, ldainv, A, lda, ZERO, work, ldwork);

    // Add identity: work = I + work = I - AINV*A
    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = ONE + work[i + i * ldwork];
    }

    // Compute norm(I - AINV*A) / (N * norm(A) * norm(AINV) * EPS)
    *resid = dlange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f64)n;
}
