/**
 * @file clsets.c
 * @brief CLSETS tests CGGLSE - a subroutine for solving linear equality
 *        constrained least square problem (LSE).
 */

#include "semicolon_cblas.h"
#include "verify.h"

/**
 * CLSETS tests CGGLSE - a subroutine for solving linear equality
 * constrained least square problem (LSE).
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     p       The number of rows of the matrix B. p >= 0.
 * @param[in]     n       The number of columns of the matrices A and B. n >= 0.
 * @param[in]     A       The M-by-N matrix A.
 * @param[out]    AF      Copy of A, destroyed by CGGLSE.
 * @param[in]     lda     The leading dimension of A and AF.
 * @param[in]     B       The P-by-N matrix B.
 * @param[out]    BF      Copy of B, destroyed by CGGLSE.
 * @param[in]     ldb     The leading dimension of B and BF.
 * @param[in]     C       The M-vector C.
 * @param[out]    CF      Copy of C, used as workspace.
 * @param[in]     D       The P-vector D.
 * @param[out]    DF      Copy of D, used as workspace.
 * @param[out]    X       The N-vector solution X.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array work.
 * @param[out]    rwork   Workspace for cget02.
 * @param[out]    result  The test ratios:
 *                        result[0] = norm( A*x - c )/ norm(A)*norm(X)*EPS
 *                        result[1] = norm( B*x - d )/ norm(B)*norm(X)*EPS
 */
void clsets(
    const INT m,
    const INT p,
    const INT n,
    const c64* A,
    c64* AF,
    const INT lda,
    const c64* B,
    c64* BF,
    const INT ldb,
    const c64* C,
    c64* CF,
    const c64* D,
    c64* DF,
    c64* X,
    c64* work,
    const INT lwork,
    f32* rwork,
    f32* result)
{
    INT info;

    /* Copy the matrices A and B to the arrays AF and BF,
       and the vectors C and D to the arrays CF and DF, */

    clacpy("F", m, n, A, lda, AF, lda);
    clacpy("F", p, n, B, ldb, BF, ldb);
    cblas_ccopy(m, C, 1, CF, 1);
    cblas_ccopy(p, D, 1, DF, 1);

    /* Solve LSE problem */

    cgglse(m, n, p, AF, lda, BF, ldb, CF, DF, X, work, lwork, &info);

    /* Test the residual for the solution of LSE

       Compute RESULT(1) = norm( A*x - c ) / norm(A)*norm(X)*EPS */

    cblas_ccopy(m, C, 1, CF, 1);
    cblas_ccopy(p, D, 1, DF, 1);
    cget02("N", m, n, 1, A, lda, X, n, CF, m, rwork, &result[0]);

    /* Compute result(2) = norm( B*x - d ) / norm(B)*norm(X)*EPS */

    cget02("N", p, n, 1, B, ldb, X, n, DF, p, rwork, &result[1]);
}
