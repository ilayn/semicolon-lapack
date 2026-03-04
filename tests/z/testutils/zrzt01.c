/**
 * @file zrzt01.c
 * @brief ZRZT01 returns || A - R*Q || / (M * eps * ||A||) for ZTZRZF.
 *
 * Port of LAPACK TESTING/LIN/zrzt01.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/**
 * ZRZT01 returns
 *    || A - R*Q || / (M * eps * ||A||)
 * for an upper trapezoidal A that was factored with ZTZRZF.
 *
 * @param[in]  m     The number of rows of the matrices A and AF.
 * @param[in]  n     The number of columns of the matrices A and AF.
 * @param[in]  A     Array (lda, n). The original upper trapezoidal M by N matrix A.
 * @param[in]  AF    Array (lda, n). The output of ZTZRZF for input matrix A.
 *                   The lower triangle is not referenced.
 * @param[in]  lda   The leading dimension of the arrays A and AF.
 * @param[in]  tau   Array (m). Details of the Householder transformations.
 * @param[out] work  Array (lwork). Workspace.
 * @param[in]  lwork The length of the array work. lwork >= m*n + m.
 *
 * @return || A - R*Q || / (M * eps * ||A||).
 */
f64 zrzt01(const INT m, const INT n, const c128* A, c128* AF,
              const INT lda, const c128* tau, c128* work, const INT lwork)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);
    const f64 ZERO = 0.0;
    INT i, j, info;
    f64 norma;
    f64 rwork[1];

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return ZERO;
    }

    /* Test for sufficient workspace */
    if (lwork < m * n + m) {
        return ZERO;
    }

    /* Compute ||A|| */
    norma = zlange("1", m, n, A, lda, rwork);

    /* Copy upper triangle R from AF into work.
     * AF(0:m-1, 0:m-1) contains the upper triangular R. */
    zlaset("F", m, n, CZERO, CZERO, work, m);
    for (j = 0; j < m; j++) {
        for (i = 0; i <= j; i++) {
            work[j * m + i] = AF[j * lda + i];
        }
    }

    /* R = R * P(0) * ... * P(m-1) = R * Q */
    zunmrz("R", "N", m, n, m, n - m, AF, lda, tau,
           work, m, &work[m * n], lwork - m * n, &info);

    /* R = R - A */
    for (i = 0; i < n; i++) {
        cblas_zaxpy(m, &CNEGONE, &A[i * lda], 1, &work[i * m], 1);
    }

    /* Compute ||R - A|| / (eps * max(M,N)) */
    f64 result = zlange("1", m, n, work, m, rwork);
    result /= dlamch("E") * (f64)((m > n) ? m : n);
    if (norma != ZERO) {
        result /= norma;
    }

    return result;
}
