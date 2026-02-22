/**
 * @file clacp2.c
 * @brief CLACP2 copies all or part of a real two-dimensional array to a complex array.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLACP2 copies all or part of a real two-dimensional matrix A to a
 * complex matrix B.
 *
 * @param[in]     uplo    Specifies the part of the matrix A to be copied to B.
 *                        = 'U': Upper triangular part
 *                        = 'L': Lower triangular part
 *                        Otherwise: All of the matrix A
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     A       Single precision array, dimension (lda, n).
 *                        The m by n matrix A. If uplo = "U", only the upper
 *                        trapezium is accessed; if uplo = "L", only the lower
 *                        trapezium is accessed.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    B       Complex*16 array, dimension (ldb, n).
 *                        On exit, B = A in the locations specified by uplo.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,m).
 */
void clacp2(
    const char* uplo,
    const INT m,
    const INT n,
    const f32* restrict A,
    const INT lda,
    c64* restrict B,
    const INT ldb)
{
    INT i, j;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            INT imax = (j + 1 < m) ? j + 1 : m;
            for (i = 0; i < imax; i++) {
                B[i + j * ldb] = A[i + j * lda];
            }
        }
    } else if (uplo[0] == 'L' || uplo[0] == 'l') {
        for (j = 0; j < n; j++) {
            for (i = j; i < m; i++) {
                B[i + j * ldb] = A[i + j * lda];
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            for (i = 0; i < m; i++) {
                B[i + j * ldb] = A[i + j * lda];
            }
        }
    }
}
