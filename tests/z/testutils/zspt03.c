/**
 * @file zspt03.c
 * @brief ZSPT03 computes the residual for a complex symmetric packed matrix
 *        times its inverse.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

void zspt03(
    const char* uplo,
    const INT n,
    const c128* const restrict A,
    const c128* const restrict AINV,
    c128* const restrict work,
    const INT ldw,
    f64* const restrict rwork,
    f64* rcond,
    f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT i, icol, j, jcol, k, kcol;
    f64 ainvnm, anorm, eps;
    c128 t;

    if (n <= 0) {
        *rcond = ONE;
        *resid = ZERO;
        return;
    }

    eps = dlamch("E");
    anorm = zlansp("1", uplo, n, A, rwork);
    ainvnm = zlansp("1", uplo, n, AINV, rwork);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        *rcond = ZERO;
        *resid = ONE / eps;
        return;
    }
    *rcond = (ONE / anorm) / ainvnm;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /*
         * Case where both A and AINV are upper triangular:
         * Each element of - A * AINV is computed by taking the dot product
         * of a row of A with a column of AINV.
         */
        for (i = 0; i < n; i++) {
            icol = (i * (i + 1)) / 2;

            /* Code when j <= i */

            for (j = 0; j <= i; j++) {
                jcol = (j * (j + 1)) / 2;
                cblas_zdotu_sub(j + 1, &A[icol], 1, &AINV[jcol], 1, &t);
                jcol = j + ((j + 1) * (j + 2)) / 2;
                kcol = icol - 1;
                for (k = j + 1; k <= i; k++) {
                    t += A[kcol + k + 1] * AINV[jcol];
                    jcol += k + 1;
                }
                kcol = i + ((i + 1) * (i + 2)) / 2;
                for (k = i + 1; k < n; k++) {
                    t += A[kcol] * AINV[jcol];
                    kcol += k + 1;
                    jcol += k + 1;
                }
                work[i + j * ldw] = -t;
            }

            /* Code when j > i */

            for (j = i + 1; j < n; j++) {
                jcol = (j * (j + 1)) / 2;
                cblas_zdotu_sub(i + 1, &A[icol], 1, &AINV[jcol], 1, &t);
                jcol = jcol - 1;
                kcol = i + ((i + 1) * (i + 2)) / 2;
                for (k = i + 1; k <= j; k++) {
                    t += A[kcol] * AINV[jcol + k + 1];
                    kcol += k + 1;
                }
                jcol = j + ((j + 1) * (j + 2)) / 2;
                for (k = j + 1; k < n; k++) {
                    t += A[kcol] * AINV[jcol];
                    kcol += k + 1;
                    jcol += k + 1;
                }
                work[i + j * ldw] = -t;
            }
        }
    } else {
        /*
         * Case where both A and AINV are lower triangular
         */
        INT nall = (n * (n + 1)) / 2;
        for (i = 0; i < n; i++) {

            /* Code when j <= i */

            icol = nall - ((n - i) * (n - i + 1)) / 2;
            for (j = 0; j <= i; j++) {
                jcol = nall - ((n - j) * (n - j + 1)) / 2 + (i - j);
                cblas_zdotu_sub(n - i, &A[icol], 1, &AINV[jcol], 1, &t);
                kcol = i;
                jcol = j;
                for (k = 0; k < j; k++) {
                    t += A[kcol] * AINV[jcol];
                    jcol += n - k - 1;
                    kcol += n - k - 1;
                }
                jcol -= j;
                for (k = j; k < i; k++) {
                    t += A[kcol] * AINV[jcol + k];
                    kcol += n - k - 1;
                }
                work[i + j * ldw] = -t;
            }

            /* Code when j > i */

            icol = nall - ((n - i - 1) * (n - i)) / 2 - 1;
            for (j = i + 1; j < n; j++) {
                jcol = nall - ((n - j) * (n - j + 1)) / 2;
                INT aoff = j + ((2 * n - i - 1) * i) / 2;
                cblas_zdotu_sub(n - j, &A[aoff], 1, &AINV[jcol], 1, &t);
                kcol = i;
                jcol = j;
                for (k = 0; k < i; k++) {
                    t += A[kcol] * AINV[jcol];
                    jcol += n - k - 1;
                    kcol += n - k - 1;
                }
                kcol -= i;
                for (k = i; k < j; k++) {
                    t += A[kcol + k] * AINV[jcol];
                    jcol += n - k - 1;
                }
                work[i + j * ldw] = -t;
            }
        }
    }

    /* Add the identity matrix to WORK. */
    for (i = 0; i < n; i++) {
        work[i + i * ldw] += CMPLX(1.0, 0.0);
    }

    /* Compute norm(I - A*AINV) / (N * norm(A) * norm(AINV) * EPS) */
    *resid = zlange("1", n, n, work, ldw, rwork);

    *resid = ((*resid * (*rcond)) / eps) / (f64)n;
}
