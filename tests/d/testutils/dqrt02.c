/**
 * @file dqrt02.c
 * @brief DQRT02 tests DORGQR for generating a partial Q matrix.
 *
 * Given the QR factorization of an m-by-n matrix A, DQRT02 generates
 * the orthogonal matrix Q defined by the factorization of the first k
 * columns of A; it compares R(0:n-1,0:k-1) with Q(0:m-1,0:n-1)'*A(0:m-1,0:k-1),
 * and checks that the columns of Q are orthonormal.
 *
 * RESULT(0) = norm( R - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * @param[in]     m       Number of rows of Q to be generated. m >= 0.
 * @param[in]     n       Number of columns of Q. m >= n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     A       The m-by-n original matrix.
 * @param[in]     AF      The QR factorization from DGEQRF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    R       Workspace, dimension (lda, n).
 * @param[in]     lda     Leading dimension >= m.
 * @param[in]     tau     Array of dimension n. Scalar factors from DGEQRF.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void dqrt02(const INT m, const INT n, const INT k,
            const f64 * const restrict A,
            const f64 * const restrict AF,
            f64 * const restrict Q,
            f64 * const restrict R,
            const INT lda,
            const f64 * const restrict tau,
            f64 * const restrict work, const INT lwork,
            f64 * const restrict rwork,
            f64 * restrict result)
{
    f64 eps = dlamch("E");
    INT info;

    /* Copy the first k columns of the factorization to Q */
    dlaset("F", m, n, -1.0e+10, -1.0e+10, Q, lda);
    if (m > 1) {
        dlacpy("L", m - 1, k, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the first n columns of Q */
    dorgqr(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy R(0:n-1, 0:k-1) */
    dlaset("F", n, k, 0.0, 0.0, R, lda);
    dlacpy("U", n, k, AF, lda, R, lda);

    /* Compute R - Q'*A(0:m-1, 0:k-1) */
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                n, k, m, -1.0, Q, lda, A, lda, 1.0, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = dlange("1", m, k, A, lda, rwork);
    f64 resid = dlange("1", n, k, R, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    dlaset("F", n, n, 0.0, 1.0, R, lda);
    cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans,
                n, m, -1.0, Q, lda, 1.0, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = dlansy("1", "U", n, R, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
