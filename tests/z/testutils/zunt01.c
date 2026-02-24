/**
 * @file zunt01.c
 * @brief ZUNT01 checks that a matrix U is unitary.
 *
 * Port of LAPACK's TESTING/EIG/zunt01.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZUNT01 checks that the matrix U is unitary by computing the ratio
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
 * @param[in] rowcol  Specifies whether to check for unitary rows or columns.
 *                    Used only if M = N.
 *                    = 'R': Check for unitary rows of U
 *                    = 'C': Check for unitary columns of U
 * @param[in] m       The number of rows of the matrix U.
 * @param[in] n       The number of columns of the matrix U.
 * @param[in] U       The unitary matrix U, dimension (ldu, n).
 * @param[in] ldu     The leading dimension of U. ldu >= max(1, m).
 * @param[out] work   Workspace array, dimension (lwork).
 * @param[in] lwork   The length of the array work. For best performance,
 *                    lwork should be at least n*(n+1) if rowcol = 'C'
 *                    or m*(m+1) if rowcol = 'R', but the test will be
 *                    done even if lwork is 0.
 * @param[out] rwork  Real workspace array, dimension (min(m,n)).
 * @param[out] resid  The computed residual.
 */
void zunt01(const char* rowcol, const INT m, const INT n,
            const c128* U, const INT ldu,
            c128* work, const INT lwork, f64* rwork, f64* resid)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    char transu;
    INT i, j, k, ldwork, mnmin;
    f64 eps;
    c128 tmp;

    *resid = ZERO;

    /* Quick return if possible */
    if (m <= 0 || n <= 0)
        return;

    eps = dlamch("P");

    if (m < n || (m == n && (rowcol[0] == 'R' || rowcol[0] == 'r'))) {
        transu = 'N';
        k = n;
    } else {
        transu = 'C';
        k = m;
    }
    mnmin = (m < n) ? m : n;

    if ((mnmin + 1) * mnmin <= lwork) {
        ldwork = mnmin;
    } else {
        ldwork = 0;
    }

    if (ldwork > 0) {
        /* Compute I - U*U' or I - U'*U */
        zlaset("U", mnmin, mnmin, CMPLX(ZERO, 0.0), CMPLX(ONE, 0.0),
               work, ldwork);

        if (transu == 'N') {
            cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                        mnmin, k, -ONE, U, ldu, ONE, work, ldwork);
        } else {
            cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                        mnmin, k, -ONE, U, ldu, ONE, work, ldwork);
        }

        /* Compute norm( I - U*U' ) / ( K * EPS ) */
        *resid = zlansy("1", "U", mnmin, work, ldwork, rwork);
        *resid = (*resid / (f64)k) / eps;

    } else if (transu == 'C') {
        /* Find the maximum element in abs( I - U'*U ) / ( m * EPS ) */
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                if (i != j) {
                    tmp = CMPLX(ZERO, 0.0);
                } else {
                    tmp = CMPLX(ONE, 0.0);
                }
                c128 dot;
                cblas_zdotc_sub(m, &U[i * ldu], 1, &U[j * ldu], 1, &dot);
                tmp = tmp - dot;
                if (cabs1(tmp) > *resid) {
                    *resid = cabs1(tmp);
                }
            }
        }
        *resid = (*resid / (f64)m) / eps;

    } else {
        /* Find the maximum element in abs( I - U*U' ) / ( n * EPS ) */
        for (j = 0; j < m; j++) {
            for (i = 0; i <= j; i++) {
                if (i != j) {
                    tmp = CMPLX(ZERO, 0.0);
                } else {
                    tmp = CMPLX(ONE, 0.0);
                }
                c128 dot;
                cblas_zdotc_sub(n, &U[j], ldu, &U[i], ldu, &dot);
                tmp = tmp - dot;
                if (cabs1(tmp) > *resid) {
                    *resid = cabs1(tmp);
                }
            }
        }
        *resid = (*resid / (f64)n) / eps;
    }
}
