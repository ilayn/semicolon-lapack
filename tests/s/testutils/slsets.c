/**
 * @file slsets.c
 * @brief SLSETS tests SGGLSE - a subroutine for solving linear equality
 *        constrained least square problem (LSE).
 */

#include <cblas.h>
#include "verify.h"

extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void sgglse(const int m, const int n, const int p,
                   f32* A, const int lda, f32* B, const int ldb,
                   f32* C, f32* D, f32* X,
                   f32* work, const int lwork, int* info);

/**
 * SLSETS tests SGGLSE - a subroutine for solving linear equality
 * constrained least square problem (LSE).
 *
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     p       The number of rows of the matrix B. p >= 0.
 * @param[in]     n       The number of columns of the matrices A and B. n >= 0.
 * @param[in]     A       The M-by-N matrix A.
 * @param[out]    AF      Copy of A, destroyed by SGGLSE.
 * @param[in]     lda     The leading dimension of A and AF.
 * @param[in]     B       The P-by-N matrix B.
 * @param[out]    BF      Copy of B, destroyed by SGGLSE.
 * @param[in]     ldb     The leading dimension of B and BF.
 * @param[in]     C       The M-vector C.
 * @param[out]    CF      Copy of C, used as workspace.
 * @param[in]     D       The P-vector D.
 * @param[out]    DF      Copy of D, used as workspace.
 * @param[out]    X       The N-vector solution X.
 * @param[out]    work    Workspace array, dimension (lwork).
 * @param[in]     lwork   The dimension of the array work.
 * @param[out]    rwork   Workspace for sget02.
 * @param[out]    result  The test ratios:
 *                        result[0] = norm( A*x - c )/ norm(A)*norm(X)*EPS
 *                        result[1] = norm( B*x - d )/ norm(B)*norm(X)*EPS
 */
void slsets(
    const int m,
    const int p,
    const int n,
    const f32* A,
    f32* AF,
    const int lda,
    const f32* B,
    f32* BF,
    const int ldb,
    const f32* C,
    f32* CF,
    const f32* D,
    f32* DF,
    f32* X,
    f32* work,
    const int lwork,
    f32* rwork,
    f32* result)
{
    int info;

    /* Copy the matrices A and B to the arrays AF and BF,
       and the vectors C and D to the arrays CF and DF, */

    slacpy("F", m, n, A, lda, AF, lda);
    slacpy("F", p, n, B, ldb, BF, ldb);
    cblas_scopy(m, C, 1, CF, 1);
    cblas_scopy(p, D, 1, DF, 1);

    /* Solve LSE problem */

    sgglse(m, n, p, AF, lda, BF, ldb, CF, DF, X, work, lwork, &info);

    /* Test the residual for the solution of LSE

       Compute RESULT(1) = norm( A*x - c ) / norm(A)*norm(X)*EPS */

    cblas_scopy(m, C, 1, CF, 1);
    cblas_scopy(p, D, 1, DF, 1);
    sget02("N", m, n, 1, A, lda, X, n, CF, m, rwork, &result[0]);

    /* Compute result(2) = norm( B*x - d ) / norm(B)*norm(X)*EPS */

    sget02("N", p, n, 1, B, ldb, X, n, DF, p, rwork, &result[1]);
}
