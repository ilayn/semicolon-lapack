/**
 * @file cunt01.c
 * @brief CUNT01 checks that a matrix U is unitary.
 *
 * Port of LAPACK's TESTING/EIG/cunt01.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CUNT01 checks that the matrix U is unitary by computing the ratio
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
void cunt01(const char* rowcol, const INT m, const INT n,
            const c64* U, const INT ldu,
            c64* work, const INT lwork, f32* rwork, f32* resid)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    char transu;
    INT i, j, k, ldwork, mnmin;
    f32 eps;
    c64 tmp;

    *resid = ZERO;

    /* Quick return if possible */
    if (m <= 0 || n <= 0)
        return;

    eps = slamch("P");

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
        claset("U", mnmin, mnmin, CMPLXF(ZERO, 0.0f), CMPLXF(ONE, 0.0f),
               work, ldwork);

        if (transu == 'N') {
            cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                        mnmin, k, -ONE, U, ldu, ONE, work, ldwork);
        } else {
            cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                        mnmin, k, -ONE, U, ldu, ONE, work, ldwork);
        }

        /* Compute norm( I - U*U' ) / ( K * EPS ) */
        *resid = clansy("1", "U", mnmin, work, ldwork, rwork);
        *resid = (*resid / (f32)k) / eps;

    } else if (transu == 'C') {
        /* Find the maximum element in abs( I - U'*U ) / ( m * EPS ) */
        for (j = 0; j < n; j++) {
            for (i = 0; i <= j; i++) {
                if (i != j) {
                    tmp = CMPLXF(ZERO, 0.0f);
                } else {
                    tmp = CMPLXF(ONE, 0.0f);
                }
                c64 dot;
                cblas_cdotc_sub(m, &U[i * ldu], 1, &U[j * ldu], 1, &dot);
                tmp = tmp - dot;
                if (cabs1f(tmp) > *resid) {
                    *resid = cabs1f(tmp);
                }
            }
        }
        *resid = (*resid / (f32)m) / eps;

    } else {
        /* Find the maximum element in abs( I - U*U' ) / ( n * EPS ) */
        for (j = 0; j < m; j++) {
            for (i = 0; i <= j; i++) {
                if (i != j) {
                    tmp = CMPLXF(ZERO, 0.0f);
                } else {
                    tmp = CMPLXF(ONE, 0.0f);
                }
                c64 dot;
                cblas_cdotc_sub(n, &U[j], ldu, &U[i], ldu, &dot);
                tmp = tmp - dot;
                if (cabs1f(tmp) > *resid) {
                    *resid = cabs1f(tmp);
                }
            }
        }
        *resid = (*resid / (f32)n) / eps;
    }
}
