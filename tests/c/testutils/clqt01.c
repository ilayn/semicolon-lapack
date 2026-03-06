/**
 * @file clqt01.c
 * @brief CLQT01 tests CGELQF and partially tests CUNGLQ.
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
 * CLQT01 tests CGELQF, which computes the LQ factorization of an m-by-n
 * matrix A, and partially tests CUNGLQ which forms the n-by-n
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
void clqt01(const INT m, const INT n,
            const c64* const restrict A,
            c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict L,
            const INT lda,
            c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    const c64 ROGUE = CMPLXF(-1.0e+10f, -1.0e+10f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT minmn = m < n ? m : n;
    f32 eps = slamch("E");
    INT info;

    /* Copy A to AF */
    clacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    cgelqf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q */
    claset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (n > 1) {
        clacpy("U", m, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the n-by-n matrix Q */
    cunglq(n, n, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy L */
    claset("F", m, n, CZERO, CZERO, L, lda);
    clacpy("L", m, n, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, A, lda, Q, lda, &CONE, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f32 anorm = clange("1", m, n, A, lda, rwork);
    f32 resid = clange("1", m, n, L, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q*Q' */
    claset("F", n, n, CZERO, CONE, L, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -1.0f, Q, lda, 1.0f, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = clansy("1", "U", n, L, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
