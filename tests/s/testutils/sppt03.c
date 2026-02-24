/**
 * @file sppt03.c
 * @brief SPPT03 computes the residual for a symmetric packed matrix times its inverse.
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * SPPT03 computes the residual for a symmetric packed matrix times its
 * inverse:
 *    norm( I - A*AINV ) / ( N * norm(A) * norm(AINV) * EPS ),
 * where EPS is the machine epsilon.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangular
 *          = 'L':  Lower triangular
 *
 * @param[in] n
 *          The number of rows and columns of the matrix A. n >= 0.
 *
 * @param[in] A
 *          The original symmetric matrix A, stored as a packed
 *          triangular matrix. Dimension (n*(n+1)/2).
 *
 * @param[in] AINV
 *          The (symmetric) inverse of the matrix A, stored as a packed
 *          triangular matrix. Dimension (n*(n+1)/2).
 *
 * @param[out] work
 *          Workspace of dimension (ldwork, n).
 *
 * @param[in] ldwork
 *          The leading dimension of the array work. ldwork >= max(1, n).
 *
 * @param[out] rwork
 *          Workspace of dimension (n).
 *
 * @param[out] rcond
 *          The reciprocal of the condition number of A, computed as
 *          ( 1/norm(A) ) / norm(AINV).
 *
 * @param[out] resid
 *          norm(I - A*AINV) / ( N * norm(A) * norm(AINV) * EPS )
 */
void sppt03(const char* uplo, const INT n,
            const f32* const restrict A,
            const f32* const restrict AINV,
            f32* const restrict work, const INT ldwork,
            f32* const restrict rwork,
            f32* rcond, f32* resid)
{
    INT i, j, jj;
    f32 ainvnm, anorm, eps;

    if (n <= 0) {
        *rcond = 1.0f;
        *resid = 0.0f;
        return;
    }

    eps = slamch("E");
    anorm = slansp("1", uplo, n, A, rwork);
    ainvnm = slansp("1", uplo, n, AINV, rwork);
    if (anorm <= 0.0f || ainvnm == 0.0f) {
        *rcond = 0.0f;
        *resid = 1.0f / eps;
        return;
    }
    *rcond = (1.0f / anorm) / ainvnm;

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        jj = 0;
        for (j = 0; j < n - 1; j++) {
            cblas_scopy(j + 1, &AINV[jj], 1, &work[(j + 1) * ldwork], 1);
            if (j > 0) {
                cblas_scopy(j, &AINV[jj], 1, &work[j + ldwork], ldwork);
            }
            jj = jj + j + 1;
        }
        jj = ((n - 1) * n) / 2;
        cblas_scopy(n - 1, &AINV[jj], 1, &work[(n - 1) + ldwork], ldwork);

        for (j = 0; j < n - 1; j++) {
            cblas_sspmv(CblasColMajor, CblasUpper, n, -1.0f, A,
                        &work[(j + 1) * ldwork], 1, 0.0f, &work[j * ldwork], 1);
        }
        cblas_sspmv(CblasColMajor, CblasUpper, n, -1.0f, A,
                    &AINV[jj], 1, 0.0f, &work[(n - 1) * ldwork], 1);

    } else {

        cblas_scopy(n - 1, &AINV[1], 1, work, ldwork);
        jj = n;
        for (j = 1; j < n; j++) {
            cblas_scopy(n - j, &AINV[jj], 1, &work[j + (j - 1) * ldwork], 1);
            if (j < n - 1) {
                cblas_scopy(n - j - 1, &AINV[jj + 1], 1, &work[j + j * ldwork], ldwork);
            }
            jj = jj + n - j;
        }

        for (j = n - 1; j >= 1; j--) {
            cblas_sspmv(CblasColMajor, CblasLower, n, -1.0f, A,
                        &work[(j - 1) * ldwork], 1, 0.0f, &work[j * ldwork], 1);
        }
        cblas_sspmv(CblasColMajor, CblasLower, n, -1.0f, A,
                    &AINV[0], 1, 0.0f, &work[0], 1);

    }

    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = work[i + i * ldwork] + 1.0f;
    }

    *resid = slange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f32)n;
}
