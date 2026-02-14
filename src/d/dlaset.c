/**
 * @file dlaset.c
 * @brief DLASET initializes the off-diagonal elements and the diagonal
 *        elements of a matrix to given values.
 */

#include "semicolon_lapack_double.h"

/**
 * DLASET initializes an m-by-n matrix A to BETA on the diagonal and
 * ALPHA on the offdiagonals.
 *
 * @param[in]  uplo   Specifies the part of A to be set.
 *                    = 'U': Upper triangular part is set; the strictly lower
 *                           triangular part of A is not changed.
 *                    = 'L': Lower triangular part is set; the strictly upper
 *                           triangular part of A is not changed.
 *                    Otherwise: All of the matrix A is set.
 * @param[in]  m      The number of rows of A. m >= 0.
 * @param[in]  n      The number of columns of A. n >= 0.
 * @param[in]  alpha  The constant to which the offdiagonal elements are set.
 * @param[in]  beta   The constant to which the diagonal elements are set.
 * @param[out] A      Double precision array, dimension (lda, n).
 *                    On exit, the leading m-by-n submatrix of A is set as
 *                    described above.
 * @param[in]  lda    The leading dimension of A. lda >= max(1, m).
 */
void dlaset(const char* uplo, const int m, const int n,
            const f64 alpha, const f64 beta,
            f64 * const restrict A, const int lda)
{
    int i, j;
    int minmn = m < n ? m : n;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        /* Set the strictly upper triangular or trapezoidal part to ALPHA */
        for (j = 1; j < n; j++) {
            int imax = j < m ? j : m;
            for (i = 0; i < imax; i++) {
                A[i + j * lda] = alpha;
            }
        }
    } else if (uplo[0] == 'L' || uplo[0] == 'l') {
        /* Set the strictly lower triangular or trapezoidal part to ALPHA */
        for (j = 0; j < minmn; j++) {
            for (i = j + 1; i < m; i++) {
                A[i + j * lda] = alpha;
            }
        }
    } else {
        /* Set the leading m-by-n submatrix to ALPHA */
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                A[i + j * lda] = alpha;
            }
        }
    }

    /* Set the first min(m, n) diagonal elements to BETA */
    for (i = 0; i < minmn; i++) {
        A[i + i * lda] = beta;
    }
}
