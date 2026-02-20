/**
 * @file srzt01.c
 * @brief SRZT01 returns || A - R*Q || / (M * eps * ||A||) for STZRZF.
 *
 * Port of LAPACK TESTING/LIN/srzt01.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern void sormrz(const char* side, const char* trans,
                   const int m, const int n, const int k, const int l,
                   const f32* A, const int lda, const f32* tau,
                   f32* C, const int ldc, f32* work, const int lwork,
                   int* info);

/**
 * SRZT01 returns
 *    || A - R*Q || / (M * eps * ||A||)
 * for an upper trapezoidal A that was factored with STZRZF.
 *
 * @param[in]  m     The number of rows of the matrices A and AF.
 * @param[in]  n     The number of columns of the matrices A and AF.
 * @param[in]  A     Array (lda, n). The original upper trapezoidal M by N matrix A.
 * @param[in]  AF    Array (lda, n). The output of STZRZF for input matrix A.
 *                   The lower triangle is not referenced.
 * @param[in]  lda   The leading dimension of the arrays A and AF.
 * @param[in]  tau   Array (m). Details of the Householder transformations.
 * @param[out] work  Array (lwork). Workspace.
 * @param[in]  lwork The length of the array work. lwork >= m*n + m.
 *
 * @return || A - R*Q || / (M * eps * ||A||).
 */
f32 srzt01(const int m, const int n, const f32* A, const f32* AF,
              const int lda, const f32* tau, f32* work, const int lwork)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    int i, j, info;
    f32 norma;
    f32 rwork[1];

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return ZERO;
    }

    /* Test for sufficient workspace */
    if (lwork < m * n + m) {
        return ZERO;
    }

    /* Compute ||A|| */
    norma = slange("1", m, n, A, lda, rwork);

    /* Copy upper triangle R from AF into work.
     * AF(0:m-1, 0:m-1) contains the upper triangular R. */
    slaset("F", m, n, ZERO, ZERO, work, m);
    for (j = 0; j < m; j++) {
        for (i = 0; i <= j; i++) {
            work[j * m + i] = AF[j * lda + i];
        }
    }

    /* R = R * P(0) * ... * P(m-1) = R * Q
     * Using sormrz with side='R' and trans='N'. */
    sormrz("R", "N", m, n, m, n - m, AF, lda, tau,
           work, m, &work[m * n], lwork - m * n, &info);

    /* R = R - A */
    for (i = 0; i < n; i++) {
        cblas_saxpy(m, -ONE, &A[i * lda], 1, &work[i * m], 1);
    }

    /* Compute ||R - A|| / (eps * max(M,N)) */
    f32 result = slange("1", m, n, work, m, rwork);
    result /= slamch("E") * (f32)((m > n) ? m : n);
    if (norma != ZERO) {
        result /= norma;
    }

    return result;
}
