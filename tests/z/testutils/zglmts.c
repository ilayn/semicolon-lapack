/**
 * @file zglmts.c
 * @brief ZGLMTS tests ZGGGLM - a subroutine for solving the generalized
 *        linear model problem.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZGLMTS tests ZGGGLM - a subroutine for solving the generalized
 * linear model problem.
 *
 * @param[in]     n       The number of rows of the matrices A and B. n >= 0.
 * @param[in]     m       The number of columns of the matrix A. m >= 0.
 * @param[in]     p       The number of columns of the matrix B. p >= 0.
 * @param[in]     A       The N-by-M matrix A.
 * @param[out]    AF      Copy of A, destroyed by ZGGGLM.
 * @param[in]     lda     The leading dimension of A and AF.
 * @param[in]     B       The N-by-P matrix B.
 * @param[out]    BF      Copy of B, destroyed by ZGGGLM.
 * @param[in]     ldb     The leading dimension of B and BF.
 * @param[in]     D       The N-vector D (right-hand side).
 * @param[out]    DF      Copy of D, used as workspace.
 * @param[out]    X       The M-vector solution X.
 * @param[out]    U       The P-vector solution U.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array work.
 * @param[out]    rwork   Workspace for zlange.
 * @param[out]    result  The test ratio:
 *                        norm( d - A*x - B*u ) / ((norm(A)+norm(B))*(norm(x)+norm(u))*EPS)
 */
void zglmts(
    const INT n,
    const INT m,
    const INT p,
    const c128* A,
    c128* AF,
    const INT lda,
    const c128* B,
    c128* BF,
    const INT ldb,
    const c128* D,
    c128* DF,
    c128* X,
    c128* U,
    c128* work,
    const INT lwork,
    f64* rwork,
    f64* result)
{
    const f64 ZERO = 0.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);
    INT info;
    f64 anorm, bnorm, dnorm, eps, unfl, xnorm, ynorm;

    eps = dlamch("E");
    unfl = dlamch("S");
    anorm = zlange("1", n, m, A, lda, rwork);
    if (anorm < unfl) anorm = unfl;
    bnorm = zlange("1", n, p, B, ldb, rwork);
    if (bnorm < unfl) bnorm = unfl;

    /* Copy the matrices A and B to the arrays AF and BF,
       and the vector D the array DF. */

    zlacpy("F", n, m, A, lda, AF, lda);
    zlacpy("F", n, p, B, ldb, BF, ldb);
    cblas_zcopy(n, D, 1, DF, 1);

    /* Solve GLM problem */

    zggglm(n, m, p, AF, lda, BF, ldb, DF, X, U, work, lwork, &info);

    /*                       norm( d - A*x - B*u )
     *       RESULT = -----------------------------------------
     *                (norm(A)+norm(B))*(norm(x)+norm(u))*EPS
     */

    cblas_zcopy(n, D, 1, DF, 1);
    cblas_zgemv(CblasColMajor, CblasNoTrans, n, m, &CNEGONE, A, lda, X, 1, &CONE, DF, 1);

    cblas_zgemv(CblasColMajor, CblasNoTrans, n, p, &CNEGONE, B, ldb, U, 1, &CONE, DF, 1);

    dnorm = cblas_dzasum(n, DF, 1);
    xnorm = cblas_dzasum(m, X, 1) + cblas_dzasum(p, U, 1);
    ynorm = anorm + bnorm;

    if (xnorm <= ZERO) {
        *result = ZERO;
    } else {
        *result = ((dnorm / ynorm) / xnorm) / eps;
    }
}
