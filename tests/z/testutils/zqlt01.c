/**
 * @file zqlt01.c
 * @brief ZQLT01 tests ZGEQLF and partially tests ZUNGQL.
 *
 * Compares L with Q'*A, and checks that Q is unitary.
 *
 * RESULT(0) = norm( L - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZQLT01 tests ZGEQLF, which computes the QL factorization of an m-by-n
 * matrix A, and partially tests ZUNGQL which forms the m-by-m
 * unitary matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, QL factorization.
 * @param[out]    Q       The m-by-m unitary matrix Q.
 * @param[out]    L       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void zqlt01(const INT m, const INT n,
            const c128* const restrict A,
            c128* const restrict AF,
            c128* const restrict Q,
            c128* const restrict L,
            const INT lda,
            c128* const restrict tau,
            c128* const restrict work, const INT lwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    const c128 ROGUE = CMPLX(-1.0e+10, -1.0e+10);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT minmn = m < n ? m : n;
    f64 eps = dlamch("E");
    INT info;

    /* Copy the matrix A to AF */
    zlacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    zgeqlf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q */
    zlaset("F", m, m, ROGUE, ROGUE, Q, lda);
    if (m >= n) {
        if (n < m && n > 0) {
            zlacpy("F", m - n, n, AF, lda, &Q[0 + (m - n) * lda], lda);
        }
        if (n > 1) {
            zlacpy("U", n - 1, n - 1, &AF[(m - n) + 1 * lda], lda,
                   &Q[(m - n) + (m - n + 1) * lda], lda);
        }
    } else {
        if (m > 1) {
            zlacpy("U", m - 1, m - 1, &AF[0 + (n - m + 1) * lda], lda,
                   &Q[0 + 1 * lda], lda);
        }
    }

    /* Generate the m-by-m matrix Q */
    zungql(m, m, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy L */
    zlaset("F", m, n, CZERO, CZERO, L, lda);
    if (m >= n) {
        if (n > 0) {
            zlacpy("L", n, n, &AF[(m - n) + 0 * lda], lda,
                   &L[(m - n) + 0 * lda], lda);
        }
    } else {
        if (n > m && m > 0) {
            zlacpy("F", m, n - m, AF, lda, L, lda);
        }
        if (m > 0) {
            zlacpy("L", m, m, &AF[0 + (n - m) * lda], lda,
                   &L[0 + (n - m) * lda], lda);
        }
    }

    /* Compute L - Q'*A */
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, lda, A, lda, &CONE, L, lda);

    /* Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = zlange("1", m, n, A, lda, rwork);
    f64 resid = zlange("1", m, n, L, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    zlaset("F", m, m, CZERO, CONE, L, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = zlansy("1", "U", m, L, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
