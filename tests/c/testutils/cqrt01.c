/**
 * @file cqrt01.c
 * @brief CQRT01 tests CGEQRF and partially tests CUNGQR.
 *
 * Compares R with Q'*A, and checks that Q is unitary.
 *
 * RESULT(0) = norm( R - Q'*A ) / ( M * norm(A) * EPS )
 * RESULT(1) = norm( I - Q'*Q ) / ( M * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * CQRT01 tests CGEQRF, which computes the QR factorization of an m-by-n
 * matrix A, and partially tests CUNGQR which forms the m-by-m
 * unitary matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, QR factorization of A.
 * @param[out]    Q       The m-by-m unitary matrix Q.
 * @param[out]    R       Workspace for R factor, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension of arrays. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace, dimension lwork.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension m.
 * @param[out]    result  Array of dimension 2 with test ratios.
 */
void cqrt01(const INT m, const INT n,
            const c64* const restrict A,
            c64* const restrict AF,
            c64* const restrict Q,
            c64* const restrict R,
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
    cgeqrf(m, n, AF, lda, tau, work, lwork, &info);

    /* Copy details of Q */
    claset("F", m, m, ROGUE, ROGUE, Q, lda);
    if (m > 1) {
        clacpy("L", m - 1, n, &AF[1 + 0 * lda], lda, &Q[1 + 0 * lda], lda);
    }

    /* Generate the m-by-m matrix Q */
    cungqr(m, m, minmn, Q, lda, tau, work, lwork, &info);

    /* Copy R */
    claset("F", m, n, CZERO, CZERO, R, lda);
    clacpy("U", m, n, AF, lda, R, lda);

    /* Compute R - Q'*A */
    cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                m, n, m, &CNEGONE, Q, lda, A, lda, &CONE, R, lda);

    /* Compute norm( R - Q'*A ) / ( M * norm(A) * EPS ) */
    f32 anorm = clange("1", m, n, A, lda, rwork);
    f32 resid = clange("1", m, n, R, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(m > 1 ? m : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    /* Compute I - Q'*Q */
    claset("F", m, m, CZERO, CONE, R, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasConjTrans,
                m, m, -1.0f, Q, lda, 1.0f, R, lda);

    /* Compute norm( I - Q'*Q ) / ( M * EPS ) */
    resid = clansy("1", "U", m, R, lda, rwork);
    result[1] = (resid / (f32)(m > 1 ? m : 1)) / eps;
}
