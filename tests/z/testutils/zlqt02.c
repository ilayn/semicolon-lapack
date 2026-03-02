/**
 * @file zlqt02.c
 * @brief ZLQT02 tests ZUNGLQ for generating a partial Q matrix from LQ.
 *
 * Given the LQ factorization of an m-by-n matrix A, ZLQT02 generates
 * the orthogonal matrix Q defined by the factorization of the first k
 * rows of A; it compares L(0:k-1,0:m-1) with A(0:k-1,0:n-1)*Q(0:m-1,0:n-1)',
 * and checks that the rows of Q are orthonormal.
 *
 * RESULT(0) = norm( L - A*Q' ) / ( N * norm(A) * EPS )
 * RESULT(1) = norm( I - Q*Q' ) / ( N * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * @param[in]     m       Number of rows of Q to generate. m >= 0.
 * @param[in]     n       Number of columns of Q. n >= m >= 0.
 * @param[in]     k       Number of reflectors. m >= k >= 0.
 * @param[in]     A       The k-by-n original matrix (first k rows).
 * @param[in]     AF      LQ factorization from ZGELQF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    L       Workspace, dimension (lda, m).
 * @param[in]     lda     Leading dimension >= n.
 * @param[in]     tau     Scalar factors from ZGELQF, dimension m.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void zlqt02(const INT m, const INT n, const INT k,
            const c128* const restrict A,
            const c128* const restrict AF,
            c128* const restrict Q,
            c128* const restrict L,
            const INT lda,
            const c128* const restrict tau,
            c128* const restrict work, const INT lwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    const c128 ROGUE = CMPLX(-1.0e+10, -1.0e+10);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    f64 eps = dlamch("E");
    INT info;

    /* Copy the first k rows of the factorization to Q */
    zlaset("F", m, n, ROGUE, ROGUE, Q, lda);
    if (n > 1) {
        zlacpy("U", k, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the first m rows of Q */
    zunglq(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy L(0:k-1, 0:m-1) */
    zlaset("F", k, m, CZERO, CZERO, L, lda);
    zlacpy("L", k, m, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                k, m, n, &CNEGONE, A, lda, Q, lda, &CONE, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f64 anorm = zlange("1", k, n, A, lda, rwork);
    f64 resid = zlange("1", k, m, L, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q*Q' */
    zlaset("F", m, m, CZERO, CONE, L, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                m, n, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = zlansy("1", "U", m, L, lda, rwork);
    result[1] = (resid / (f64)(n > 1 ? n : 1)) / eps;
}
