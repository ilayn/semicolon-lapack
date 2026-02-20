/**
 * @file spot03.c
 * @brief SPOT03 computes the residual for a symmetric matrix times its inverse.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

// Forward declarations
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);

/**
 * SPOT03 computes the residual for a symmetric matrix times its inverse:
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
void spot03(
    const char* uplo,
    const int n,
    const f32* const restrict A,
    const int lda,
    f32* const restrict AINV,
    const int ldainv,
    f32* const restrict work,
    const int ldwork,
    f32* const restrict rwork,
    f32* rcond,
    f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    // Quick exit if n = 0
    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    // Exit with RESID = 1/EPS if ANORM = 0 or AINVNM = 0
    f32 eps = slamch("E");
    f32 anorm = slansy("1", uplo, n, A, lda, rwork);
    f32 ainvnm = slansy("1", uplo, n, AINV, ldainv, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    // Expand AINV into a full matrix
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < j; i++) {
                AINV[j + i * ldainv] = AINV[i + j * ldainv];
            }
        }
    } else {
        for (int j = 0; j < n; j++) {
            for (int i = j + 1; i < n; i++) {
                AINV[j + i * ldainv] = AINV[i + j * ldainv];
            }
        }
    }

    // Compute work = -A * AINV using DSYMM
    cblas_ssymm(CblasColMajor, CblasLeft,
                (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower,
                n, n, -ONE, A, lda, AINV, ldainv, ZERO, work, ldwork);

    // Add the identity matrix to work
    for (int i = 0; i < n; i++) {
        work[i + i * ldwork] += ONE;
    }

    // Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS)
    *resid = slange("1", n, n, work, ldwork, rwork);
    *resid = ((*resid * (*rcond)) / eps) / (f32)n;
}
