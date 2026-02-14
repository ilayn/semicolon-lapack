/**
 * @file dqpt01.c
 * @brief DQPT01 tests the QR-factorization with pivoting of a matrix A.
 *
 * Port of LAPACK TESTING/LIN/dqpt01.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include <cblas.h>
#include "verify.h"

/* External declarations */
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern void dormqr(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f64* A, const int lda, const f64* tau,
                   f64* C, const int ldc, f64* work, const int lwork,
                   int* info);

/**
 * DQPT01 tests the QR-factorization with pivoting of a matrix A. The
 * array AF contains the (possibly partial) QR-factorization of A, where
 * the upper triangle of AF(1:K,1:K) is a partial triangular factor,
 * the entries below the diagonal in the first K columns are the
 * Householder vectors, and the rest of AF contains a partially updated
 * matrix.
 *
 * This function returns ||A*P - Q*R|| / (||norm(A)|| * eps * max(M,N)),
 * where ||.|| is matrix one norm.
 *
 * @param[in]  m     The number of rows of the matrices A and AF.
 * @param[in]  n     The number of columns of the matrices A and AF.
 * @param[in]  k     The number of columns of AF that have been reduced
 *                   to upper triangular form.
 * @param[in]  A     Array (lda, n). The original matrix A.
 * @param[in]  AF    Array (lda, n). The (possibly partial) output of DGEQPF.
 * @param[in]  lda   The leading dimension of the arrays A and AF.
 * @param[in]  tau   Array (k). Details of the Householder transformations.
 * @param[in]  jpvt  Array (n). Pivot information (0-based in C).
 * @param[out] work  Array (lwork). Workspace.
 * @param[in]  lwork The length of the array work. lwork >= m*n + n.
 *
 * @return ||A*P - Q*R|| / (||norm(A)|| * eps * max(M,N)).
 */
f64 dqpt01(const int m, const int n, const int k,
              const f64* A, const f64* AF, const int lda,
              const f64* tau, const int* jpvt,
              f64* work, const int lwork)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    int i, j, info;
    f64 norma;
    f64 rwork[1];

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return ZERO;
    }

    /* Test if there is enough workspace */
    if (lwork < m * n + n) {
        return ZERO;
    }

    /* Compute ||A|| (one-norm) */
    norma = dlange("1", m, n, A, lda, rwork);

    /* Copy the upper triangular part of the factor R stored
     * in AF(0:k-1, 0:k-1) into the work array WORK.
     * Also copy columns (k:n-1) from AF into work. */
    for (j = 0; j < k; j++) {
        /* Copy upper triangular part of column j */
        int imax = (j + 1 < m) ? (j + 1) : m;
        for (i = 0; i < imax; i++) {
            work[j * m + i] = AF[j * lda + i];
        }
        /* Zero out elements below the diagonal */
        for (i = j + 1; i < m; i++) {
            work[j * m + i] = ZERO;
        }
    }

    /* Copy columns (k:n-1) from AF into work */
    for (j = k; j < n; j++) {
        for (i = 0; i < m; i++) {
            work[j * m + i] = AF[j * lda + i];
        }
    }

    /* Form Q*R using dormqr */
    dormqr("L", "N", m, n, k, AF, lda, tau, work, m, &work[m * n], lwork - m * n, &info);

    /* Compute Q*R - A*P: for each column j, subtract A(:, jpvt[j]) */
    for (j = 0; j < n; j++) {
        /* Subtract A(:, jpvt[j]) from work(:, j) */
        cblas_daxpy(m, -ONE, &A[jpvt[j] * lda], 1, &work[j * m], 1);
    }

    /* Compute ||Q*R - A*P|| / (max(M,N) * ||A|| * eps) */
    f64 result = dlange("1", m, n, work, m, rwork) /
                    ((f64)((m > n) ? m : n) * dlamch("E"));
    if (norma != ZERO) {
        result /= norma;
    }

    return result;
}
