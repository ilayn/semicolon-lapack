/**
 * @file zrqt02.c
 * @brief ZRQT02 tests ZUNGRQ for generating a partial Q matrix from RQ.
 *
 * Given the RQ factorization of an m-by-n matrix A, ZRQT02 generates
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
 * @param[in]     AF      RQ factorization from ZGERQF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    R       Workspace, dimension (lda, m).
 * @param[in]     lda     Leading dimension >= n.
 * @param[in]     tau     Scalar factors from ZGERQF, dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void zrqt02(const INT m, const INT n, const INT k,
            const c128* const restrict A,
            const c128* const restrict AF,
            c128* const restrict Q,
            c128* const restrict R,
            const INT lda,
            const c128* const restrict tau,
            c128* const restrict work, const INT lwork,
            f64* const restrict rwork,
            f64* restrict result)
{
    if (m == 0 || n == 0 || k == 0) {
        result[0] = 0.0;
        result[1] = 0.0;
        return;
    }

    const c128 ROGUE = CMPLX(-1.0e+10, -1.0e+10);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    f64 eps = dlamch("E");
    INT info;
    INT minmn = m < n ? m : n;

    zlaset("F", m, n, ROGUE, ROGUE, Q, lda);
    if (k < n) {
        zlacpy("F", k, n - k, &AF[(m - k) + 0 * lda], lda,
               &Q[(m - k) + 0 * lda], lda);
    }
    if (k > 1) {
        zlacpy("L", k - 1, k - 1, &AF[(m - k + 1) + (n - k) * lda], lda,
               &Q[(m - k + 1) + (n - k) * lda], lda);
    }

    zungrq(m, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

    zlaset("F", k, m, CZERO, CZERO, &R[(m - k) + (n - m) * lda], lda);
    zlacpy("U", k, k, &AF[(m - k) + (n - k) * lda], lda,
           &R[(m - k) + (n - k) * lda], lda);

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                k, m, n, &CNEGONE, &A[(m - k) + 0 * lda], lda, Q, lda,
                &CONE, &R[(m - k) + (n - m) * lda], lda);

    f64 anorm = zlange("1", k, n, &A[(m - k) + 0 * lda], lda, rwork);
    f64 resid = zlange("1", k, m, &R[(m - k) + (n - m) * lda], lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    zlaset("F", m, m, CZERO, CONE, R, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasNoTrans,
                m, n, -1.0, Q, lda, 1.0, R, lda);

    resid = zlansy("1", "U", m, R, lda, rwork);
    result[1] = (resid / (f64)(n > 1 ? n : 1)) / eps;
}
