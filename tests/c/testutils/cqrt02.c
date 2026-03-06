/**
 * @file cqrt02.c
 * @brief CQRT02 tests CUNGQR for generating a partial Q matrix.
 *
 * Given the QR factorization of an m-by-n matrix A, CQRT02 generates
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
 * @param[in]     AF      The QR factorization from CGEQRF.
 * @param[out]    Q       The m-by-n matrix Q.
 * @param[out]    R       Workspace, dimension (lda, n).
 * @param[in]     lda     Leading dimension >= m.
 * @param[in]     tau     Array of dimension n. Scalar factors from CGEQRF.
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void cqrt02(const INT m, const INT n, const INT k,
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
    const c64 ROGUE = CMPLXF(-1.0e+10f, -1.0e+10f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    f32 eps = slamch("E");
    INT info;

    /* Copy the first k columns of the factorization to Q */
    claset("F", m, n, ROGUE, ROGUE, Q, lda);
    if (m > 1) {
        clacpy("L", m - 1, k, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the first n columns of Q */
    cungqr(m, n, k, Q, lda, tau, work, lwork, &info);

    /* Copy R(0:n-1, 0:k-1) */
    claset("F", n, k, CZERO, CZERO, R, lda);
    clacpy("U", n, k, AF, lda, R, lda);

    /* Compute R - Q'*A(0:m-1, 0:k-1) */
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                n, k, m, &CNEGONE, Q, lda, A, lda, &CONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    f32 anorm = clange("1", m, k, A, lda, rwork);
    f32 resid = clange("1", n, k, R, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q'*Q */
    claset("F", n, n, CZERO, CONE, R, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                n, m, -1.0f, Q, lda, 1.0f, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = clansy("1", "U", n, R, lda, rwork);
    result[1] = (resid / (f32)(m > 1 ? m : 1)) / eps;
}
