/**
 * @file cqlt01.c
 * @brief CQLT01 tests CGEQLF and partially tests CUNGQL.
 *
 * Compares L with Q'*A, and checks that Q is unitary.
 *
 * RESULT(0) = norm( L - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * CQLT01 tests CGEQLF, which computes the QL factorization of an m-by-n
 * matrix A, and partially tests CUNGQL which forms the m-by-m
 * unitary matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, QL factorization.
 * @param[out]    Q       The m-by-m unitary matrix Q.
 * @param[out]    L       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2.
 */
void cqlt01(const INT m, const INT n,
            const c64* const restrict A,
            c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict L,
            const INT lda,
            c64* const restrict tau,
            c64* const restrict work, const INT lwork,
            f32* const restrict rwork,
            f32* restrict result)
{
    const c64 ROGUE = CMPLXF(-1.0e+10f, -1.0e+10f);
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 CNEGONE = CMPLXF(-1.0f, 0.0f);

    INT minmn = m < n ? m : n;
    f32 eps = slamch("E");
    INT info;

    /* Copy the matrix A to AF */
    clacpy("F", m, n, A, lda, AF, lda);

    /* Factorize A in AF */
    cgeqlf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q */
    claset("F", m, m, ROGUE, ROGUE, Q, lda);
    if (m >= n) {
        if (n < m && n > 0) {
            clacpy("F", m - n, n, AF, lda, &Q[0 + (m - n) * lda], lda);
        }
        if (n > 1) {
            clacpy("U", n - 1, n - 1, &AF[(m - n) + 1 * lda], lda,
                   &Q[(m - n) + (m - n + 1) * lda], lda);
        }
    } else {
        if (m > 1) {
            clacpy("U", m - 1, m - 1, &AF[0 + (n - m + 1) * lda], lda,
                   &Q[0 + 1 * lda], lda);
        }
    }

    /* Generate the m-by-m matrix Q */
    cungql(m, m, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy L */
    claset("F", m, n, CZERO, CZERO, L, lda);
    if (m >= n) {
        if (n > 0) {
            clacpy("L", n, n, &AF[(m - n) + 0 * lda], lda,
                   &L[(m - n) + 0 * lda], lda);
        }
    } else {
        if (n > m && m > 0) {
            clacpy("F", m, n - m, AF, lda, L, lda);
        }
        if (m > 0) {
            clacpy("L", m, m, &AF[0 + (n - m) * lda], lda,
                   &L[0 + (n - m) * lda], lda);
        }
    }

    /* Compute L - Q'*A */
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, lda, A, lda, &CONE, L, lda);

    /* Compute norm( L - Q'*A ) / ( M * norm(A) * EPS ) */
    f32 anorm = clange("1", m, n, A, lda, rwork);
    f32 resid = clange("1", m, n, L, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q'*Q */
    claset("F", m, m, CZERO, CONE, L, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -1.0f, Q, lda, 1.0f, L, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = clansy("1", "U", m, L, lda, rwork);
    result[1] = (resid / (f32)(m > 1 ? m : 1)) / eps;
}
