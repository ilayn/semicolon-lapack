/**
 * @file slacpy.c
 * @brief SLACPY copies all or part of one two-dimensional array to another.
 */

#include "semicolon_lapack_single.h"

/**
 * SLACPY copies all or part of a two-dimensional matrix A to another
 * matrix B.
 *
 * @param[in]     uplo    Specifies the part of the matrix A to be copied to B.
 *                        = 'U': Upper triangular part
 *                        = 'L': Lower triangular part
 *                        Otherwise: All of the matrix A
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     A       The m by n matrix A. If uplo = "U", only the upper
 *                        triangle or trapezoid is accessed; if uplo = "L",
 *                        only the lower triangle or trapezoid is accessed.
 *                        Array of dimension (lda, n).
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    B       On exit, B = A in the locations specified by uplo.
 *                        Array of dimension (ldb, n).
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,m).
 */
void slacpy(
    const char* uplo,
    const int m,
    const int n,
    const f32* restrict A,
    const int lda,
    f32* restrict B,
    const int ldb)
{
    int i, j;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        // Copy upper triangular part
        for (j = 0; j < n; j++) {
            int imax = (j + 1 < m) ? j + 1 : m;
            for (i = 0; i < imax; i++) {
                B[i + j * ldb] = A[i + j * lda];
            }
        }
    } else if (uplo[0] == 'L' || uplo[0] == 'l') {
        // Copy lower triangular part
        for (j = 0; j < n; j++) {
            for (i = j; i < m; i++) {
                B[i + j * ldb] = A[i + j * lda];
            }
        }
    } else {
        // Copy all of matrix A
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                B[i + j * ldb] = A[i + j * lda];
            }
        }
    }
}
