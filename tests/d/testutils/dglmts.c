/**
 * @file dglmts.c
 * @brief DGLMTS tests DGGGLM - a subroutine for solving the generalized
 *        linear model problem.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * DGLMTS tests DGGGLM - a subroutine for solving the generalized
 * linear model problem.
 *
 * @param[in]     n       The number of rows of the matrices A and B. n >= 0.
 * @param[in]     m       The number of columns of the matrix A. m >= 0.
 * @param[in]     p       The number of columns of the matrix B. p >= 0.
 * @param[in]     A       The N-by-M matrix A.
 * @param[out]    AF      Copy of A, destroyed by DGGGLM.
 * @param[in]     lda     The leading dimension of A and AF.
 * @param[in]     B       The N-by-P matrix B.
 * @param[out]    BF      Copy of B, destroyed by DGGGLM.
 * @param[in]     ldb     The leading dimension of B and BF.
 * @param[in]     D       The N-vector D (right-hand side).
 * @param[out]    DF      Copy of D, used as workspace.
 * @param[out]    X       The M-vector solution X.
 * @param[out]    U       The P-vector solution U.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array work.
 * @param[out]    rwork   Workspace for dlange.
 * @param[out]    result  The test ratio:
 *                        norm( d - A*x - B*u ) / ((norm(A)+norm(B))*(norm(x)+norm(u))*EPS)
 */
void dglmts(
    const INT n,
    const INT m,
    const INT p,
    const f64* A,
    f64* AF,
    const INT lda,
    const f64* B,
    f64* BF,
    const INT ldb,
    const f64* D,
    f64* DF,
    f64* X,
    f64* U,
    f64* work,
    const INT lwork,
    f64* rwork,
    f64* result)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    INT info;
    f64 anorm, bnorm, dnorm, eps, unfl, xnorm, ynorm;

    eps = dlamch("E");
    unfl = dlamch("S");
    anorm = dlange("1", n, m, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = dlange("1", n, p, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Copy the matrices A and B to the arrays AF and BF,
       and the vector D the array DF. */

    dlacpy("F", n, m, A, lda, AF, lda);
    dlacpy("F", n, p, B, ldb, BF, ldb);
    cblas_dcopy(n, D, 1, DF, 1);

    /* Solve GLM problem */

    dggglm(n, m, p, AF, lda, BF, ldb, DF, X, U, work, lwork, &info);

    /*                       norm( d - A*x - B*u )
     *       RESULT = -----------------------------------------
     *                (norm(A)+norm(B))*(norm(x)+norm(u))*EPS
     */

    cblas_dcopy(n, D, 1, DF, 1);
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, m, -ONE, A, lda, X, 1, ONE, DF, 1);

    cblas_dgemv(CblasColMajor, CblasNoTrans, n, p, -ONE, B, ldb, U, 1, ONE, DF, 1);

    dnorm = cblas_dasum(n, DF, 1);
    xnorm = cblas_dasum(m, X, 1) + cblas_dasum(p, U, 1);
    ynorm = anorm + bnorm;

    if (xnorm <= ZERO) {
        *result = ZERO;
    } else {
        *result = ((dnorm / ynorm) / xnorm) / eps;
    }
}
