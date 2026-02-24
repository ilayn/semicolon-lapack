/**
 * @file slqt01.c
 * @brief SLQT01 tests SGELQF and partially tests SORGLQ.
 *
 * Compares L with A*Q', and checks that Q is orthogonal.
 *
 * RESULT(0) = norm( L - A*Q' ) / ( N * norm(A) * EPS )
 * RESULT(1) = norm( I - Q*Q' ) / ( N * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * SLQT01 tests SGELQF, which computes the LQ factorization of an m-by-n
 * matrix A, and partially tests SORGLQ which forms the n-by-n
 * orthogonal matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, LQ factorization.
 * @param[out]    Q       The n-by-n orthogonal matrix Q.
 * @param[out]    L       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension max(m,n).
 * @param[out]    result  Array of dimension 2.
 */
void slqt01(const INT m, const INT n,
            const f32 * const restrict A,
            f32 * const restrict AF,
            f32 * const restrict Q,
            f32 * const restrict L,
            const INT lda,
            f32 * const restrict tau,
            f32 * const restrict work, const INT lwork,
            f32 * const restrict rwork,
            f32 * restrict result)
{
    INT minmn = m < n ? m : n;
    f32 eps = slamch("E");
    INT info;

    /* Copy A to AF */
    slacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    sgelqf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q: the upper triangle of AF (above diagonal) contains
     * the reflectors for LQ. Copy to Q. */
    slaset("F", n, n, -1.0e+10f, -1.0e+10f, Q, lda);
    if (n > 1) {
        /* Copy upper trapezoid of AF (rows 0:m-1, cols 1:n-1) to Q */
        slacpy("U", m, n - 1, &AF[0 + 1 * lda], lda, &Q[0 + 1 * lda], lda);
    }

    /* Generate the n-by-n matrix Q */
    sorglq(n, n, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy L: lower triangle of AF */
    slaset("F", m, n, 0.0f, 0.0f, L, lda);
    slacpy("L", m, n, AF, lda, L, lda);

    /* Compute L - A*Q' */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                m, n, n, -1.0f, A, lda, Q, lda, 1.0f, L, lda);

    /* Compute norm( L - A*Q' ) / ( N * norm(A) * EPS ) */
    f32 anorm = slange("1", m, n, A, lda, rwork);
    f32 resid = slange("1", m, n, L, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q*Q' */
    slaset("F", n, n, 0.0f, 1.0f, L, lda);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -1.0f, Q, lda, 1.0f, L, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = slansy("1", "U", n, L, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
