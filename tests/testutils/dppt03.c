/**
 * @file dppt03.c
 * @brief DPPT03 computes the residual for a symmetric packed matrix times its inverse.
 */

#include <stdlib.h>
#include <math.h>
#include "verify.h"
#include <cblas.h>

extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);
extern double dlansp(const char* norm, const char* uplo, const int n,
                     const double* const restrict AP,
                     double* const restrict work);

/**
 * DPPT03 computes the residual for a symmetric packed matrix times its
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
void dppt03(const char* uplo, const int n,
            const double* const restrict A,
            const double* const restrict AINV,
            double* const restrict work, const int ldwork,
            double* const restrict rwork,
            double* rcond, double* resid)
{
    int i, j, jj;
    double ainvnm, anorm, eps;

    if (n <= 0) {
        *rcond = 1.0;
        *resid = 0.0;
        return;
    }

    eps = dlamch("E");
    anorm = dlansp("1", uplo, n, A, rwork);
    ainvnm = dlansp("1", uplo, n, AINV, rwork);
    if (anorm <= 0.0 || ainvnm == 0.0) {
        *rcond = 0.0;
        *resid = 1.0 / eps;
        return;
    }
    *rcond = (1.0 / anorm) / ainvnm;

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        jj = 0;
        for (j = 0; j < n - 1; j++) {
            cblas_dcopy(j + 1, &AINV[jj], 1, &work[(j + 1) * ldwork], 1);
            if (j > 0) {
                cblas_dcopy(j, &AINV[jj], 1, &work[j + ldwork], ldwork);
            }
            jj = jj + j + 1;
        }
        jj = ((n - 1) * n) / 2;
        cblas_dcopy(n - 1, &AINV[jj], 1, &work[(n - 1) + ldwork], ldwork);

        for (j = 0; j < n - 1; j++) {
            cblas_dspmv(CblasColMajor, CblasUpper, n, -1.0, A,
                        &work[(j + 1) * ldwork], 1, 0.0, &work[j * ldwork], 1);
        }
        cblas_dspmv(CblasColMajor, CblasUpper, n, -1.0, A,
                    &AINV[jj], 1, 0.0, &work[(n - 1) * ldwork], 1);

    } else {

        cblas_dcopy(n - 1, &AINV[1], 1, work, ldwork);
        jj = n;
        for (j = 1; j < n; j++) {
            cblas_dcopy(n - j, &AINV[jj], 1, &work[j + (j - 1) * ldwork], 1);
            if (j < n - 1) {
                cblas_dcopy(n - j - 1, &AINV[jj + 1], 1, &work[j + j * ldwork], ldwork);
            }
            jj = jj + n - j;
        }

        for (j = n - 1; j >= 1; j--) {
            cblas_dspmv(CblasColMajor, CblasLower, n, -1.0, A,
                        &work[(j - 1) * ldwork], 1, 0.0, &work[j * ldwork], 1);
        }
        cblas_dspmv(CblasColMajor, CblasLower, n, -1.0, A,
                    &AINV[0], 1, 0.0, &work[0], 1);

    }

    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = work[i + i * ldwork] + 1.0;
    }

    *resid = dlange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (double)n;
}
