/**
 * @file srqt02.c
 * @brief SRQT02 tests SORGRQ for generating a partial Q matrix from RQ.
 *
 * Given the RQ factorization of an m-by-n matrix A, SRQT02 generates
 * the orthogonal matrix Q defined by the factorization of the last k
 * rows of A; it compares R(m-k+1:m, n-m+1:n) with
 * A(m-k+1:m, 1:n)*Q(n-m+1:n, 1:n)', and checks that the rows of Q
 * are orthonormal.
 *
 * RESULT(0) = norm( R - A*Q' ) / ( N * norm(A) * EPS )
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
 * @param[in]     A       The m-by-n original matrix (last k rows used).
 * @param[in]     AF      RQ factorization from SGERQF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    R       Workspace, dimension (lda, m).
 * @param[in]     lda     Leading dimension >= n.
 * @param[in]     tau     Scalar factors from SGERQF, dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void srqt02(const INT m, const INT n, const INT k,
            const f32* const restrict A,
            const f32* const restrict AF,
            f32* const restrict Q,
            f32* const restrict R,
            const INT lda,
            const f32* const restrict tau,
            f32* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        result[0] = 0.0f;
        result[1] = 0.0f;
        return;
    }

    f32 eps = slamch("E");
    INT info;
    INT minmn = m < n ? m : n;

    /* Copy the last k rows of the factorization to the array Q.
     * For RQ, reflectors for the last k rows are stored in AF(m-k:m-1, 0:n-1).
     * We need to copy:
     *   - The non-triangular part: AF(m-k:m-1, 0:n-k-1) -> Q(m-k:m-1, 0:n-k-1)
     *   - The lower triangular part: AF(m-k+1:m-1, n-k:n-2) -> Q(m-k+1:m-1, n-k:n-2)
     * In 0-indexed C:
     *   - AF(m-k:m-1, 0:n-k-1) -> Q(m-k:m-1, 0:n-k-1)
     *   - Lower of AF(m-k+1:m-1, n-k:n-2) -> Q(m-k+1:m-1, n-k:n-2)
     */
    slaset("F", m, n, -1.0e+10f, -1.0e+10f, Q, lda);
    if (k < n) {
        /* Copy AF(m-k:m-1, 0:n-k-1) -> Q(m-k:m-1, 0:n-k-1) */
        slacpy("F", k, n - k, &AF[(m - k) + 0 * lda], lda,
               &Q[(m - k) + 0 * lda], lda);
    }
    if (k > 1) {
        /* Copy lower triangle of AF(m-k+1:m-1, n-k:n-2) -> Q(m-k+1:m-1, n-k:n-2) */
        slacpy("L", k - 1, k - 1, &AF[(m - k + 1) + (n - k) * lda], lda,
               &Q[(m - k + 1) + (n - k) * lda], lda);
    }

    /* Generate the last m rows of the matrix Q.
     * tau for the last k reflectors starts at tau[minmn-k] (0-indexed). */
    sorgrq(m, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

    /* Copy R(m-k:m-1, n-m:n-1) - this is a k-by-m block.
     * The upper triangular part is in AF(m-k:m-1, n-k:n-1). */
    slaset("F", k, m, 0.0f, 0.0f, &R[(m - k) + (n - m) * lda], lda);
    slacpy("U", k, k, &AF[(m - k) + (n - k) * lda], lda,
           &R[(m - k) + (n - k) * lda], lda);

    /* Compute R(m-k:m-1, n-m:n-1) - A(m-k:m-1, 0:n-1) * Q(n-m:n-1, 0:n-1)'
     * This is: R_block -= A_block * Q'
     * where R_block is k x m, A_block is k x n, Q is m x n */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                k, m, n, -1.0f, &A[(m - k) + 0 * lda], lda, Q, lda,
                1.0f, &R[(m - k) + (n - m) * lda], lda);

    /* Compute norm( R - A*Q' ) / ( N * norm(A) * EPS ) */
    f32 anorm = slange("1", k, n, &A[(m - k) + 0 * lda], lda, rwork);
    f32 resid = slange("1", k, m, &R[(m - k) + (n - m) * lda], lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q*Q' */
    slaset("F", m, m, 0.0f, 1.0f, R, lda);
    cblas_ssyrk(CblasColMajor, CblasUpper, CblasNoTrans,
                m, n, -1.0f, Q, lda, 1.0f, R, lda);

    /* Compute norm( I - Q*Q' ) / ( N * EPS ) */
    resid = slansy("1", "U", m, R, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
