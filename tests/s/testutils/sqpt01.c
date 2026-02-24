/**
 * @file sqpt01.c
 * @brief SQPT01 tests the QR-factorization with pivoting of a matrix A.
 *
 * Port of LAPACK TESTING/LIN/sqpt01.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* External declarations */
/**
 * SQPT01 tests the QR-factorization with pivoting of a matrix A. The
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
f32 sqpt01(const INT m, const INT n, const INT k,
              const f32* A, const f32* AF, const INT lda,
              const f32* tau, const INT* jpvt,
              f32* work, const INT lwork)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    INT i, j, info;
    f32 norma;
    f32 rwork[1];

    /* Quick return if possible */
    if (m <= 0 || n <= 0) {
        return ZERO;
    }

    /* Test if there is enough workspace */
    if (lwork < m * n + n) {
        return ZERO;
    }

    /* Compute ||A|| (one-norm) */
    norma = slange("1", m, n, A, lda, rwork);

    /* Copy the upper triangular part of the factor R stored
     * in AF(0:k-1, 0:k-1) into the work array WORK.
     * Also copy columns (k:n-1) from AF into work. */
    for (j = 0; j < k; j++) {
        /* Copy upper triangular part of column j */
        INT imax = (j + 1 < m) ? (j + 1) : m;
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

    /* Form Q*R using sormqr */
    sormqr("L", "N", m, n, k, AF, lda, tau, work, m, &work[m * n], lwork - m * n, &info);

    /* Compute Q*R - A*P: for each column j, subtract A(:, jpvt[j]) */
    for (j = 0; j < n; j++) {
        /* Subtract A(:, jpvt[j]) from work(:, j) */
        cblas_saxpy(m, -ONE, &A[jpvt[j] * lda], 1, &work[j * m], 1);
    }

    /* Compute ||Q*R - A*P|| / (max(M,N) * ||A|| * eps) */
    f32 result = slange("1", m, n, work, m, rwork) /
                    ((f32)((m > n) ? m : n) * slamch("E"));
    if (norma != ZERO) {
        result /= norma;
    }

    return result;
}
