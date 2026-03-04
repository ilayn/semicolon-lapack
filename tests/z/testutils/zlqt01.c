/**
 * @file zlqt01.c
 * @brief ZLQT01 tests ZGELQF and partially tests ZUNGLQ.
 *
 * Compares L with A*Q', and checks that Q is unitary.
 *
 * RESULT(0) = norm( L - A*Q' ) / ( N * norm(A) * EPS )
 * RESULT(1) = norm( I - Q*Q' ) / ( N * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZLQT01 tests ZGELQF, which computes the LQ factorization of an m-by-n
 * matrix A, and partially tests ZUNGLQ which forms the n-by-n
 * unitary matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, LQ factorization.
 * @param[out]    Q       The n-by-n unitary matrix Q.
 * @param[out]    L       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension max(m,n).
 * @param[out]    result  Array of dimension 2.
 */
void zlqt01(const INT m, const INT n,
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

    /* Copy A to AF */
    zlacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    zgelqf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q */
    zlaset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (n > 1) {
        zlacpy("U", m, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the n-by-n matrix Q */
    zunglq(n, n, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy L */
    zlaset("F", m, n, CZERO, CZERO, L, lda);
    zlacpy("L", m, n, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, A, lda, Q, lda, &CONE, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f64 anorm = zlange("1", m, n, A, lda, rwork);
    f64 resid = zlange("1", m, n, L, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q*Q' */
    zlaset("F", n, n, CZERO, CONE, L, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = zlansy("1", "U", n, L, lda, rwork);
    result[1] = (resid / (f64)(n > 1 ? n : 1)) / eps;
}
