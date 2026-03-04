/**
 * @file zppt03.c
 * @brief ZPPT03 computes the residual for a Hermitian packed matrix times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZPPT03 computes the residual for a Hermitian packed matrix times its
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
void zppt03(const char* uplo, const INT n,
            const c128* const restrict A,
            const c128* const restrict AINV,
            c128* const restrict work, const INT ldwork,
            f64* const restrict rwork,
            f64* rcond, f64* resid)
{
    INT i, j, jj;
    f64 ainvnm, anorm, eps;

    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    if (n <= 0) {
        *rcond = 1.0;
        *resid = 0.0;
        return;
    }

    eps = dlamch("E");
    anorm = zlanhp("1", uplo, n, A, rwork);
    ainvnm = zlanhp("1", uplo, n, AINV, rwork);
    if (anorm <= 0.0 || ainvnm <= 0.0) {
        *rcond = 0.0;
        *resid = 1.0 / eps;
        return;
    }
    *rcond = (1.0 / anorm) / ainvnm;

    if (uplo[0] == 'U' || uplo[0] == 'u') {

        /* Copy AINV: expand packed upper-triangular AINV into full work matrix.
         * Column j+1 of AINV (packed) goes into column j+1 of work.
         * Row entries (below the diagonal) are filled with conjugates. */
        jj = 0;
        for (j = 0; j < n - 1; j++) {
            /* Copy column j of packed AINV (elements 0..j) into work column j+1 */
            cblas_zcopy(j + 1, &AINV[jj], 1, &work[(j + 1) * ldwork], 1);
            /* Fill row j: conjugate column entries into row positions */
            for (i = 0; i < j; i++) {
                work[j + (i + 1) * ldwork] = conj(AINV[jj + i]);
            }
            jj = jj + j + 1;
        }
        jj = ((n - 1) * n) / 2;
        /* Fill row n-1 from last column of packed AINV */
        for (i = 0; i < n - 1; i++) {
            work[(n - 1) + (i + 1) * ldwork] = conj(AINV[jj + i]);
        }

        /* Multiply by A */
        for (j = 0; j < n - 1; j++) {
            cblas_zhpmv(CblasColMajor, CblasUpper, n, &CNEGONE, A,
                        &work[(j + 1) * ldwork], 1, &CZERO, &work[j * ldwork], 1);
        }
        cblas_zhpmv(CblasColMajor, CblasUpper, n, &CNEGONE, A,
                    &AINV[jj], 1, &CZERO, &work[(n - 1) * ldwork], 1);

    } else {

        /* Copy AINV: expand packed lower-triangular AINV into full work matrix.
         * Row entries (above the diagonal) are filled with conjugates. */
        for (i = 0; i < n - 1; i++) {
            work[0 + i * ldwork] = conj(AINV[i + 1]);
        }
        jj = n;
        for (j = 1; j < n; j++) {
            /* Copy column j of packed AINV (elements j..n-1) into work column j-1 */
            cblas_zcopy(n - j, &AINV[jj], 1, &work[j + (j - 1) * ldwork], 1);
            /* Fill row entries with conjugates */
            for (i = 0; i < n - j - 1; i++) {
                work[j + (j + i) * ldwork] = conj(AINV[jj + i + 1]);
            }
            jj = jj + n - j;
        }

        /* Multiply by A */
        for (j = n - 1; j >= 1; j--) {
            cblas_zhpmv(CblasColMajor, CblasLower, n, &CNEGONE, A,
                        &work[(j - 1) * ldwork], 1, &CZERO, &work[j * ldwork], 1);
        }
        cblas_zhpmv(CblasColMajor, CblasLower, n, &CNEGONE, A,
                    &AINV[0], 1, &CZERO, &work[0], 1);

    }

    /* Add the identity matrix to WORK */
    for (i = 0; i < n; i++) {
        work[i + i * ldwork] = work[i + i * ldwork] + CONE;
    }

    *resid = zlange("1", n, n, work, ldwork, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f64)n;
}
