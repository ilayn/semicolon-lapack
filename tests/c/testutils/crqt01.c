/**
 * @file crqt01.c
 * @brief CRQT01 tests CGERQF and partially tests CUNGRQ.
 *
 * Compares R with A*Q', and checks that Q is orthogonal.
 *
 * RESULT(0) = norm( R - A*Q' ) / ( N * norm(A) * EPS )
 * RESULT(1) = norm( I - Q*Q' ) / ( N * EPS )
 */

#include <stdlib.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * CRQT01 tests CGERQF, which computes the RQ factorization of an m-by-n
 * matrix A, and partially tests CUNGRQ which forms the n-by-n
 * orthogonal matrix Q.
 *
 * @param[in]     m       Number of rows of A. m >= 0.
 * @param[in]     n       Number of columns of A. n >= 0.
 * @param[in]     A       The m-by-n matrix A (original).
 * @param[in,out] AF      On entry, copy of A. On exit, RQ factorization.
 * @param[out]    Q       The n-by-n orthogonal matrix Q.
 * @param[out]    R       Workspace, dimension (lda, max(m,n)).
 * @param[in]     lda     Leading dimension. lda >= max(m,n).
 * @param[out]    tau     Array of dimension min(m,n).
 * @param[out]    work    Workspace.
 * @param[in]     lwork   Dimension of work.
 * @param[out]    rwork   Workspace, dimension max(m,n).
 * @param[out]    result  Array of dimension 2.
 */
void crqt01(const INT m, const INT n,
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

    clacpy("F", m, n, A, lda, AF, lda);

    cgerqf(m, n, AF, lda, tau, work, lwork, &info);

    claset("F", n, n, ROGUE, ROGUE, Q, lda);
    if (m <= n) {
        if (m > 0 && m < n) {
            clacpy("F", m, n - m, AF, lda, &Q[(n - m) + 0 * lda], lda);
        }
        if (m > 1) {
            clacpy("L", m - 1, m - 1, &AF[1 + (n - m) * lda], lda,
                   &Q[(n - m + 1) + (n - m) * lda], lda);
        }
    } else {
        if (n > 1) {
            clacpy("L", n - 1, n - 1, &AF[(m - n + 1) + 0 * lda], lda,
                   &Q[1 + 0 * lda], lda);
        }
    }

    cungrq(n, n, minmn, Q, lda, tau, work, lwork, &info);

    claset("F", m, n, CZERO, CZERO, R, lda);
    if (m <= n) {
        if (m > 0) {
            clacpy("U", m, m, &AF[0 + (n - m) * lda], lda,
                   &R[0 + (n - m) * lda], lda);
        }
    } else {
        if (m > n && n > 0) {
            clacpy("F", m - n, n, AF, lda, R, lda);
        }
        if (n > 0) {
            clacpy("U", n, n, &AF[(m - n) + 0 * lda], lda,
                   &R[(m - n) + 0 * lda], lda);
        }
    }

    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                m, n, n, &CNEGONE, A, lda, Q, lda, &CONE, R, lda);

    f32 anorm = clange("1", m, n, A, lda, rwork);
    f32 resid = clange("1", m, n, R, lda, rwork);
    if (anorm > 0.0f) {
        result[0] = ((resid / (f32)(n > 1 ? n : 1)) / anorm) / eps;
    } else {
        result[0] = 0.0f;
    }

    claset("F", n, n, CZERO, CONE, R, lda);
    cblas_cherk(CblasColMajor, CblasUpper, CblasNoTrans,
                n, n, -1.0f, Q, lda, 1.0f, R, lda);

    resid = clansy("1", "U", n, R, lda, rwork);
    result[1] = (resid / (f32)(n > 1 ? n : 1)) / eps;
}
