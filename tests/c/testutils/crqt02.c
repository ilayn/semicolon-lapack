/**
 * @file crqt02.c
 * @brief CRQT02 tests CUNGRQ for generating a partial Q matrix from RQ.
 *
 * Given the RQ factorization of an m-by-n matrix A, CRQT02 generates
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
 * @param[in]     AF      RQ factorization from CGERQF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    R       Workspace, dimension (lda, m).
 * @param[in]     lda     Leading dimension >= n.
 * @param[in]     tau     Scalar factors from CGERQF, dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void crqt02(const INT m, const INT n, const INT k,
            const c64* const restrict A,
            const c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict R,
            const INT lda,
            const c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    if (m == 0 || n == 0 || k == 0) {
        result[0] = 0.0f;
        result[1] = 0.0f;
        return;
    }

    const c64 ROGUE = CMPLXF(-1.0e+10f, -1.0e+10f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    f32 eps = slamch("E");
    INT info;
    INT minmn = m < n ? m : n;

    claset("F", m, n, ROGUE, ROGUE, Q, lda);
    if (k < n) {
        clacpy("F", k, n - k, &AF[(m - k) + 0 * lda], lda,
               &Q[(m - k) + 0 * lda], lda);
    }
    if (k > 1) {
        clacpy("L", k - 1, k - 1, &AF[(m - k + 1) + (n - k) * lda], lda,
               &Q[(m - k + 1) + (n - k) * lda], lda);
    }

    cungrq(m, n, k, Q, lda, &tau[minmn - k], work, lwork, &info);

    claset("F", k, m, CZERO, CZERO, &R[(m - k) + (n - m) * lda], lda);
    clacpy("U", k, k, &AF[(m - k) + (n - k) * lda], lda,
           &R[(m - k) + (n - k) * lda], lda);

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                k, m, n, &CNEGONE, &A[(m - k) + 0 * lda], lda, Q, lda,
                &CONE, &R[(m - k) + (n - m) * lda], lda);

    f32 anorm = clange("1", k, n, &A[(m - k) + 0 * lda], lda, rwork);
    f32 resid = clange("1", k, m, &R[(m - k) + (n - m) * lda], lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    claset("F", m, m, CZERO, CONE, R, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                m, n, -1.0f, Q, lda, 1.0f, R, lda);

    resid = clansy("1", "U", m, R, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
