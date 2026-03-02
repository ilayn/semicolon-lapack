/**
 * @file zrqt01.c
 * @brief ZRQT01 tests ZGERQF and partially tests ZUNGRQ.
 *
 * Compares R with A*Q', and checks that Q is orthogonal.
 *
 * RESULT(0) = norm( R - A*Q' ) / ( N * norm(A) * EPS )
 * RESULT(1) = norm( I - Q*Q' ) / ( N * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * ZRQT01 tests ZGERQF, which computes the RQ factorization of an m-by-n
 * matrix A, and partially tests ZUNGRQ which forms the n-by-n
 * orthogonal matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, RQ factorization.
 * @param[out]    Q       The n-by-n orthogonal matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension max(m,n).
 * @param[out]    result  Array of dimension 2.
 */
void zrqt01(const INT m, const INT n,
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

    zlacpy("F", m, n, A, lda, AF, lda);

    zgerqf(m, n, AF, lda, tau, work, lwork, &info);

    zlaset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (m <= n) {
        if (m > 0 && m < n) {
            zlacpy("F", m, n - m, AF, lda, &Q[(n - m) + 0 * lda], lda);
        }
        if (m > 1) {
            zlacpy("L", m - 1, m - 1, &AF[1 + (n - m) * lda], lda,
                   &Q[(n - m + 1) + (n - m) * lda], lda);
        }
    } else {
        if (n > 1) {
            zlacpy("L", n - 1, n - 1, &AF[(m - n + 1) + 0 * lda], lda,
                   &Q[1 + 0 * lda], lda);
        }
    }

    zungrq(n, n, minmn, Q, lda, tau, work, lwork, &info);

    zlaset("F", m, n, CZERO, CZERO, R, lda);
    if (m <= n) {
        if (m > 0) {
            zlacpy("U", m, m, &AF[0 + (n - m) * lda], lda,
                   &R[0 + (n - m) * lda], lda);
        }
    } else {
        if (m > n && n > 0) {
            zlacpy("F", m - n, n, AF, lda, R, lda);
        }
        if (n > 0) {
            zlacpy("U", n, n, &AF[(m - n) + 0 * lda], lda,
                   &R[(m - n) + 0 * lda], lda);
        }
    }

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, A, lda, Q, lda, &CONE, R, lda);

    f64 anorm = zlange("1", m, n, A, lda, rwork);
    f64 resid = zlange("1", m, n, R, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    zlaset("F", n, n, CZERO, CONE, R, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -1.0, Q, lda, 1.0, R, lda);

    resid = zlansy("1", "U", n, R, lda, rwork);
    result[1] = (resid / (f64)(n > 1 ? n : 1)) / eps;
}
