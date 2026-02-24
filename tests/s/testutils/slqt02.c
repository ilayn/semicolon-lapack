/**
 * @file slqt02.c
 * @brief SLQT02 tests SORGLQ for generating a partial Q matrix from LQ.
 *
 * Given the LQ factorization of an m-by-n matrix A, SLQT02 generates
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
 * @param[in]     AF      LQ factorization from SGELQF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    L       Workspace, dimension (lda, m).
 * @param[in]     lda     Leading dimension >= n.
 * @param[in]     tau     Scalar factors from SGELQF, dimension m.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void slqt02(const INT m, const INT n, const INT k,
            const f32 * const restrict A,
            const f32 * const restrict AF,
            f32 * const restrict Q,
            f32 * const restrict L,
            const INT lda,
            const f32 * const restrict tau,
            f32 * const restrict work, const INT lwork,
            f32 * const restrict rwork,
            f32 * restrict result)
{
    f32 eps = slamch("E");
    INT info;

    /* Copy the first k rows of the factorization to Q */
    slaset("F", m, n, -1.0e+10f, -1.0e+10f, Q, lda);
    if (n > 1) {
        slacpy("U", k, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the first m rows of Q */
    sorglq(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy L(0:k-1, 0:m-1) */
    slaset("F", k, m, 0.0f, 0.0f, L, lda);
    slacpy("L", k, m, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                k, m, n, -1.0f, A, lda, Q, lda, 1.0f, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f32 anorm = slange("1", k, n, A, lda, rwork);
    f32 resid = slange("1", k, m, L, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q*Q' */
    slaset("F", m, m, 0.0f, 1.0f, L, lda);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                m, n, -1.0f, Q, lda, 1.0f, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = slansy("1", "U", m, L, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
