/**
 * @file dpot03.c
 * @brief DPOT03 computes the residual for a symmetric matrix times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DPOT03 computes the residual for a symmetric matrix times its inverse:
 *    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in]     uplo    Specifies whether the upper or lower triangular part
 *                        of the symmetric matrix A is stored:
 *                        = 'U': Upper triangular
 *                        = 'L': Lower triangular
 * @param[in]     n       The number of rows and columns of the matrix A. n >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The original symmetric matrix A.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,n).
 * @param[in,out] AINV    Double precision array, dimension (ldainv, n).
 *                        The inverse of the matrix A, stored as a symmetric
 *                        matrix. On exit, the opposing triangle is filled.
 * @param[in]     ldainv  The leading dimension of the array AINV.
 *                        ldainv >= max(1,n).
 * @param[out]    work    Double precision array, dimension (ldwork, n).
 * @param[in]     ldwork  The leading dimension of the array work.
 *                        ldwork >= max(1,n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    rcond   The reciprocal of the condition number of A, computed
 *                        as (1/norm(A)) / norm(AINV).
 * @param[out]    resid   norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
 */
void dpot03(
    const char* uplo,
    const INT n,
    const f64* const restrict A,
    const INT lda,
    f64* const restrict AINV,
    const INT ldainv,
    f64* const restrict work,
    const INT ldwork,
    f64* const restrict rwork,
    f64* rcond,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    // Quick exit if n = 0
    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    // Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0
    f64 eps = dlamch("E");
    f64 anorm = dlansy("1", uplo, n, A, lda, rwork);
    f64 ainvnm = dlansy("1", uplo, n, AINV, ldainv, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    // Expand AINV into a full matrix
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < j; i++) {
                AINV[j + i * ldainv] = AINV[i + j * ldainv];
            }
        }
    } else {
        for (INT j = 0; j < n; j++) {
            for (INT i = j + 1; i < n; i++) {
                AINV[j + i * ldainv] = AINV[i + j * ldainv];
            }
        }
    }

    // Compute work = -A * AINV using DSYMM
    cblas_dsymm(CblasColMajor, CblasLeft,
                (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower,
                n, n, -ONE, A, lda, AINV, ldainv, ZERO, work, ldwork);

    // Add the identity matrix to work
    for (INT i = 0; i < n; i++) {
        work[i + i * ldwork] += ONE;
    }

    // Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
    *resid = dlange("1", n, n, work, ldwork, rwork);
    *resid = ((*resid * (*rcond)) / eps) / (f64)n;
}
