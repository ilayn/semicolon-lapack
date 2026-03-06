/**
 * @file cppt03.c
 * @brief CPPT03 computes the residual for a Hermitian packed matrix times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * CPPT03 computes the residual for a Hermitian packed matrix times its
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
 *          The original Hermitian matrix A, stored as a packed
 *          triangular matrix. Dimension (n*(n+1)/2).
 *
 * @param[in] AINV
 *          The (Hermitian) inverse of the matrix A, stored as a packed
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
void cppt03(const char* uplo, const INT n,
            const c64* const restrict A,
            const c64* const restrict AINV,
            c64* const restrict work, const INT ldwork,
            f32* const restrict rwork,
            f32* rcond, f32* resid)
{
    INT i, j, jj;
    f32 ainvnm, anorm, eps;

    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    if (n <= 0) {
        *rcond = 1.0f;
        *resid = 0.0f;
        return;
    }

    eps = slamch("E");
    anorm = clanhp("1", uplo, n, A, rwork);
    ainvnm = clanhp("1", uplo, n, AINV, rwork);
    if (anorm <= 0.0f || ainvnm <= 0.0f) {
        *rcond = 0.0f;
        *resid = 1.0f / eps;
        return;
    }
    *rcond = (1.0f / anorm) / ainvnm;

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        /* Copy AINV: expand packed upper-triangular AINV into full work matrix.
         * Column j+1 of AINV (packed) goes into column j+1 of work.
         * Row entries (below the diagonal) are filled with conjugates. */
        jj = 0;
        for (j = 0; j < n - 1; j++) {
            /* Copy column j of packed AINV (elements 0..j) into work column j+1 */
            cblas_ccopy(j + 1, &AINV[jj], 1, &work[(j + 1) * ldwork], 1);
            /* Fill row j: conjugate column entries into row positions */
            for (i = 0; i < j; i++) {
                work[j + (i + 1) * ldwork] = conjf(AINV[jj + i]);
            }
            jj = jj + j + 1;
        }
        jj = ((n - 1) * n) / 2;
        /* Fill row n-1 from last column of packed AINV */
        for (i = 0; i < n - 1; i++) {
            work[(n - 1) + (i + 1) * ldwork] = conjf(AINV[jj + i]);
        }

        /* Multiply by A */
        for (j = 0; j < n - 1; j++) {
            cblas_chpmv(CblasColMajor, CblasUpper, n, &CNEGONE, A,
                        &work[(j + 1) * ldwork], 1, &CZERO, &work[j * ldwork], 1);
        }
        cblas_chpmv(CblasColMajor, CblasUpper, n, &CNEGONE, A,
                    &AINV[jj], 1, &CZERO, &work[(n - 1) * ldwork], 1);

    } else {

        /* Copy AINV: expand packed lower-triangular AINV into full work matrix.
         * Row entries (above the diagonal) are filled with conjugates. */
        for (i = 0; i < n - 1; i++) {
            work[0 + i * ldwork] = conjf(AINV[i + 1]);
        }
        jj = n;
        for (j = 1; j < n; j++) {
            /* Copy column j of packed AINV (elements j..n-1) into work column j-1 */
            cblas_ccopy(n - j, &AINV[jj], 1, &work[j + (j - 1) * ldwork], 1);
            /* Fill row entries with conjugates */
            for (i = 0; i < n - j - 1; i++) {
                work[j + (j + i) * ldwork] = conjf(AINV[jj + i + 1]);
            }
            jj = jj + n - j;
        }

        /* Multiply by A */
        for (j = n - 1; j >= 1; j--) {
            cblas_chpmv(CblasColMajor, CblasLower, n, &CNEGONE, A,
                        &work[(j - 1) * ldwork], 1, &CZERO, &work[j * ldwork], 1);
        }
        cblas_chpmv(CblasColMajor, CblasLower, n, &CNEGONE, A,
                    &AINV[0], 1, &CZERO, &work[0], 1);

    }

    /* Add the identity matrix to WORK */
    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = work[i + i * ldwork] + CONE;
    }

    *resid = clange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f32)n;
}
