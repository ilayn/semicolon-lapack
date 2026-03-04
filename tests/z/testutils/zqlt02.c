/**
 * @file zqlt02.c
 * @brief ZQLT02 tests ZUNGQL for generating a partial Q matrix from QL.
 *
 * Given the QL factorization of an m-by-n matrix A, ZQLT02 generates
 * the orthogonal matrix Q defined by the factorization of the last k
 * columns of A; it compares L(m-n+1:m, n-k+1:n) with
 * Q(1:m, m-n+1:m)' * A(1:m, n-k+1:n), and checks that the columns of Q
 * are orthonormal.
 *
 * RESULT(0) = norm( L - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * @param[in]     m       Number of rows of Q to generate. m >= 0.
 * @param[in]     n       Number of columns of Q. m >= n >= 0.
 * @param[in]     k       Number of reflectors. n >= k >= 0.
 * @param[in]     A       The m-by-n original matrix (last k columns used).
 * @param[in]     AF      QL factorization from ZGEQLF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    L       Workspace, dimension (lda, n).
 * @param[in]     lda     Leading dimension >= m.
 * @param[in]     tau     Scalar factors from ZGEQLF, dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void zqlt02(const INT m, const INT n, const INT k,
            const c128* const restrict A,
            const c128* const restrict AF,
            c128* const restrict Q,
            c128* const restrict L,
            const INT lda,
            const c128* const restrict tau,
            c128* const restrict work, const INT lwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    const c128 ROGUE = CMPLX(-1.0e+10, -1.0e+10);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0) {
        result[0] = 0.0;
        result[1] = 0.0;
        return;
    }

    f64 eps = dlamch("E");
    INT info;
    INT minmn = m < n ? m : n;

    /* Copy the last k columns of the factorization to the array Q */
    zlaset("F", m, n, ROGUE, ROGUE, Q, lda);
    if (k < m) {
        zlacpy("F", m - k, k, &AF[0 + (n - k) * lda], lda,
               &Q[0 + (n - k) * lda], lda);
    }
    if (k > 1) {
        zlacpy("U", k - 1, k - 1, &AF[(m - k) + (n - k + 1) * lda], lda,
               &Q[(m - k) + (n - k + 1) * lda], lda);
    }

    /* Generate the last n columns of the matrix Q */
    zungql(m, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

    /* Copy L(m-n:m-1, n-k:n-1) */
    zlaset("F", n, k, CZERO, CZERO, &L[(m - n) + (n - k) * lda], lda);
    zlacpy("L", k, k, &AF[(m - k) + (n - k) * lda], lda,
           &L[(m - k) + (n - k) * lda], lda);

    /* Compute L(m-n:m-1, n-k:n-1) - Q(0:m-1, m-n:m-1)' * A(0:m-1, n-k:n-1) */
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, k, m, &CNEGONE, Q, lda, &A[0 + (n - k) * lda], lda,
                &CONE, &L[(m - n) + (n - k) * lda], lda);

    /* Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = zlange("1", m, k, &A[0 + (n - k) * lda], lda, rwork);
    f64 resid = zlange("1", n, k, &L[(m - n) + (n - k) * lda], lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    zlaset("F", n, n, CZERO, CONE, L, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, m, -1.0, Q, lda, 1.0, L, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = zlansy("1", "U", n, L, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
