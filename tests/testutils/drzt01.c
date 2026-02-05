/**
 * @file drzt01.c
 * @brief DRZT01 returns || A - R*Q || / (M * eps * ||A||) for DTZRZF.
 *
 * Port of LAPACK TESTING/LIN/drzt01.f to C.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern void dormrz(const char* side, const char* trans,
                   const int m, const int n, const int k, const int l,
                   const double* A, const int lda, const double* tau,
                   double* C, const int ldc, double* work, const int lwork,
                   int* info);

/**
 * DRZT01 returns
 *    || A - R*Q || / (M * eps * ||A||)
 * for an upper trapezoidal A that was factored with DTZRZF.
 *
 * @param[in]  m     The number of rows of the matrices A and AF.
 * @param[in]  n     The number of columns of the matrices A and AF.
 * @param[in]  A     Array (lda, n). The original upper trapezoidal M by N matrix A.
 * @param[in]  AF    Array (lda, n). The output of DTZRZF for input matrix A.
 *                   The lower triangle is not referenced.
 * @param[in]  lda   The leading dimension of the arrays A and AF.
 * @param[in]  tau   Array (m). Details of the Householder transformations.
 * @param[out] work  Array (lwork). Workspace.
 * @param[in]  lwork The length of the array work. lwork >= m*n + m.
 *
 * @return || A - R*Q || / (M * eps * ||A||).
 */
double drzt01(const int m, const int n, const double* A, const double* AF,
              const int lda, const double* tau, double* work, const int lwork)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    int i, j, info;
    double norma;
    double rwork[1];

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return ZERO;
    }

    /* Test for sufficient workspace */
    if (lwork < m * n + m) {
        return ZERO;
    }

    /* Compute ||A|| */
    norma = dlange("1", m, n, A, lda, rwork);

    /* Copy upper triangle R from AF into work.
     * AF(0:m-1, 0:m-1) contains the upper triangular R. */
    dlaset("F", m, n, ZERO, ZERO, work, m);
    for (j = 0; j < m; j++) {
        for (i = 0; i <= j; i++) {
            work[j * m + i] = AF[j * lda + i];
        }
    }

    /* R = R * P(0) * ... * P(m-1) = R * Q
     * Using dormrz with side='R' and trans='N'. */
    dormrz("R", "N", m, n, m, n - m, AF, lda, tau,
           work, m, &work[m * n], lwork - m * n, &info);

    /* R = R - A */
    for (i = 0; i < n; i++) {
        cblas_daxpy(m, -ONE, &A[i * lda], 1, &work[i * m], 1);
    }

    /* Compute ||R - A|| / (eps * max(M,N)) */
    double result = dlange("1", m, n, work, m, rwork);
    result /= dlamch("E") * (double)((m > n) ? m : n);
    if (norma != ZERO) {
        result /= norma;
    }

    return result;
}
