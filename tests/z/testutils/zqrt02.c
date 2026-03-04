/**
 * @file zqrt02.c
 * @brief ZQRT02 tests ZUNGQR for generating a partial Q matrix.
 *
 * Given the QR factorization of an m-by-n matrix A, ZQRT02 generates
 * the unitary matrix Q defined by the factorization of the first k
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
 * @param[in]     AF      The QR factorization from ZGEQRF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    R       Workspace, dimension (lda, n).
 * @param[in]     lda     Leading dimension >= m.
 * @param[in]     tau     Array of dimension n. Scalar factors from ZGEQRF.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void zqrt02(const INT m, const INT n, const INT k,
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
    const c128 ROGUE = CMPLX(-1.0e+10, -1.0e+10);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    f64 eps = dlamch("E");
    INT info;

    /* Copy the first k columns of the factorization to Q */
    zlaset("F", m, n, ROGUE, ROGUE, Q, lda);
    if (m > 1) {
        zlacpy("L", m - 1, k, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the first n columns of Q */
    zungqr(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy R(0:n-1, 0:k-1) */
    zlaset("F", n, k, CZERO, CZERO, R, lda);
    zlacpy("U", n, k, AF, lda, R, lda);

    /* Compute R - Q'*A(0:m-1, 0:k-1) */
    cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, k, m, &CNEGONE, Q, lda, A, lda, &CONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    f64 anorm = zlange("1", m, k, A, lda, rwork);
    f64 resid = zlange("1", n, k, R, lda, rwork);
    if (anorm > 0.0) {
        result[0] = ((resid / (f64)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0;
    }

    /* Compute I - Q'*Q */
    zlaset("F", n, n, CZERO, CONE, R, lda);
    cblas_zherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, m, -1.0, Q, lda, 1.0, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = zlansy("1", "U", n, R, lda, rwork);
    result[1] = (resid / (f64)(m > 1 ? m : 1)) / eps;
}
