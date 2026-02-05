/**
 * @file dget03.c
 * @brief DGET03 computes the residual for a general matrix times its inverse.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double * const restrict A, const int lda,
                     double * const restrict work);

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
    const int n,
    const double * const restrict A,
    const int lda,
    const double * const restrict AINV,
    const int ldainv,
    double * const restrict work,
    const int ldwork,
    double * const restrict rwork,
    double *rcond,
    double *resid)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i;
    double ainvnm, anorm, eps;

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

    *resid = ((*resid * (*rcond)) / eps) / (double)n;
}
