/**
 * @file sort01.c
 * @brief SORT01 checks that a matrix U is orthogonal.
 */

#include <cblas.h>
#include <math.h>
#include "verify.h"

/* Forward declarations */
extern f32 slamch(const char* cmach);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* A, const int lda, f32* work);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);

/**
 * SORT01 checks that the matrix U is orthogonal by computing the ratio
 *
 *    RESID = norm( I - U*U' ) / ( n * EPS ), if ROWCOL = 'R',
 * or
 *    RESID = norm( I - U'*U ) / ( m * EPS ), if ROWCOL = 'C'.
 *
 * Alternatively, if there isn't sufficient workspace to form
 * I - U*U' or I - U'*U, the ratio is computed as
 *
 *    RESID = abs( I - U*U' ) / ( n * EPS ), if ROWCOL = 'R',
 * or
 *    RESID = abs( I - U'*U ) / ( m * EPS ), if ROWCOL = 'C'.
 *
 * where EPS is the machine precision. ROWCOL is used only if m = n;
 * if m > n, ROWCOL is assumed to be 'C', and if m < n, ROWCOL is
 * assumed to be 'R'.
 *
 * @param[in] rowcol  Specifies whether to check for orthogonal rows or columns.
 *                    Used only if M = N.
 *                    = 'R': Check for orthogonal rows of U
 *                    = 'C': Check for orthogonal columns of U
 * @param[in] m       The number of rows of the matrix U.
 * @param[in] n       The number of columns of the matrix U.
 * @param[in] U       The orthogonal matrix U, dimension (ldu, n).
 * @param[in] ldu     The leading dimension of U. ldu >= max(1, m).
 * @param[out] work   Workspace array, dimension (lwork).
 * @param[in] lwork   The length of the array work. For best performance,
 *                    lwork should be at least n*(n+1) if rowcol = 'C'
 *                    or m*(m+1) if rowcol = 'R', but the test will be
 *                    done even if lwork is 0.
 * @param[out] resid  The computed residual.
 */
void sort01(const char* rowcol, const int m, const int n,
            const f32* U, const int ldu,
            f32* work, const int lwork, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    char transu;
    int i, j, k, ldwork, mnmin;
    f32 eps, tmp;

    *resid = ZERO;

    /* Quick return if possible */
    if (m <= 0 || n <= 0)
        return;

    eps = slamch("P");

    /* Determine whether to check U*U' or U'*U */
    if (m < n || (m == n && (rowcol[0] == 'R' || rowcol[0] == 'r'))) {
        transu = 'N';
        k = n;
    } else {
        transu = 'T';
        k = m;
    }
    mnmin = (m < n) ? m : n;

    /* Determine if we have enough workspace for the efficient algorithm */
    if ((mnmin + 1) * mnmin <= lwork) {
        ldwork = mnmin;
    } else {
        ldwork = 0;
    }

    if (ldwork > 0) {
        /*
         * Compute I - U*U' or I - U'*U using DSYRK.
         * First, set WORK to identity matrix (upper triangular part).
         */
        slaset("U", mnmin, mnmin, ZERO, ONE, work, ldwork);

        /* Compute WORK = I - U*U' or WORK = I - U'*U */
        if (transu == 'N') {
            /* WORK = I - U*U' (row orthogonality) */
            cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                        mnmin, k, -ONE, U, ldu, ONE, work, ldwork);
        } else {
            /* WORK = I - U'*U (column orthogonality) */
            cblas_ssyrk(CblasColMajor, CblasUpper, CblasTrans,
                        mnmin, k, -ONE, U, ldu, ONE, work, ldwork);
        }

        /* Compute norm( I - U*U' ) / ( K * EPS ) */
        *resid = slansy("1", "U", mnmin, work, ldwork,
                        &work[ldwork * mnmin]);
        *resid = (*resid / (f32)k) / eps;

    } else if (transu == 'T') {
        /*
         * Find the maximum element in abs( I - U'*U ) / ( m * EPS )
         * Check column orthogonality without forming the full product.
         */
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                if (i != j) {
                    tmp = ZERO;
                } else {
                    tmp = ONE;
                }
                /* tmp = tmp - dot(U(:,i), U(:,j)) */
                tmp = tmp - cblas_sdot(m, &U[i * ldu], 1, &U[j * ldu], 1);
                if (fabsf(tmp) > *resid) {
                    *resid = fabsf(tmp);
                }
            }
        }
        *resid = (*resid / (f32)m) / eps;

    } else {
        /*
         * Find the maximum element in abs( I - U*U' ) / ( n * EPS )
         * Check row orthogonality without forming the full product.
         */
        for (j = 0; j < m; j++) {
            for (i = 0; i <= j; i++) {
                if (i != j) {
                    tmp = ZERO;
                } else {
                    tmp = ONE;
                }
                /* tmp = tmp - dot(U(j,:), U(i,:)) */
                tmp = tmp - cblas_sdot(n, &U[j], ldu, &U[i], ldu);
                if (fabsf(tmp) > *resid) {
                    *resid = fabsf(tmp);
                }
            }
        }
        *resid = (*resid / (f32)n) / eps;
    }
}
