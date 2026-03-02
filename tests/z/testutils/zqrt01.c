/**
 * @file zqrt01.c
 * @brief ZQRT01 tests ZGEQRF and partially tests ZUNGQR.
 *
 * Compares R with Q'*A, and checks that Q is unitary.
 *
 * RESULT(0) = norm( R - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZQRT01 tests ZGEQRF, which computes the QR factorization of an m-by-n
 * matrix A, and partially tests ZUNGQR which forms the m-by-m
 * unitary matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, QR factorization of A.
 * @param[out]    Q       The m-by-m unitary matrix Q.
 * @param[out]    R       Workspace for R factor, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of arrays. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace, dimension lwork.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2 with test ratios.
 */
void zqrt01(const INT m, const INT n,
            const c128* const restrict A,
            c128* const restrict AF,
            c128* const restrict Q,
            c128* const restrict R,
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
    zgeqrf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q */
    zlaset("F", m, m, ROGUE, ROGUE, Q, lda);
    if (m > 1) {
        zlacpy("L", m - 1, n, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the m-by-m matrix Q */
    zungqr(m, m, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy R */
    zlaset("F", m, n, CZERO, CZERO, R, lda);
    zlacpy("U", m, n, AF, lda, R, lda);

    /* Compute R - Q'*A */
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, lda, A, lda, &CONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = zlange("1", m, n, A, lda, rwork);
    f64 resid = zlange("1", m, n, R, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    zlaset("F", m, m, CZERO, CONE, R, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -1.0, Q, lda, 1.0, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = zlansy("1", "U", m, R, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
