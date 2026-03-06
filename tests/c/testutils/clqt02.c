/**
 * @file clqt02.c
 * @brief CLQT02 tests CUNGLQ for generating a partial Q matrix from LQ.
 *
 * Given the LQ factorization of an m-by-n matrix A, CLQT02 generates
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
 * @param[in]     AF      LQ factorization from CGELQF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    L       Workspace, dimension (lda, m).
 * @param[in]     lda     Leading dimension >= n.
 * @param[in]     tau     Scalar factors from CGELQF, dimension m.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void clqt02(const INT m, const INT n, const INT k,
            const c64* const restrict A,
            const c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict L,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    const c64 ROGUE = CMPLXF(-1.0e+10f, -1.0e+10f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    f32 eps = slamch("E");
    INT info;

    /* Copy the first k rows of the factorization to Q */
    claset("F", m, n, ROGUE, ROGUE, Q, lda);
    if (n > 1) {
        clacpy("U", k, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the first m rows of Q */
    cunglq(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy L(0:k-1, 0:m-1) */
    claset("F", k, m, CZERO, CZERO, L, lda);
    clacpy("L", k, m, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                k, m, n, &CNEGONE, A, lda, Q, lda, &CONE, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f32 anorm = clange("1", k, n, A, lda, rwork);
    f32 resid = clange("1", k, m, L, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q*Q' */
    claset("F", m, m, CZERO, CONE, L, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                m, n, -1.0f, Q, lda, 1.0f, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = clansy("1", "U", m, L, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
