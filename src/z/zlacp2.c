/**
 * @file zlacp2.c
 * @brief ZLACP2 copies all or part of a real two-dimensional array to a complex array.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZLACP2 copies all or part of a real two-dimensional matrix A to a
 * complex matrix B.
 *
 * @param[in]     uplo    Specifies the part of the matrix A to be copied to B.
 *                        = 'U': Upper triangular part
 *                        = 'L': Lower triangular part
 *                        Otherwise: All of the matrix A
 * @param[in]     m       The number of rows of the matrix A. m >= 0.
 * @param[in]     n       The number of columns of the matrix A. n >= 0.
 * @param[in]     A       Double precision array, dimension (lda, n).
 *                        The m by n matrix A. If uplo = "U", only the upper
 *                        trapezium is accessed; if uplo = "L", only the lower
 *                        trapezium is accessed.
 * @param[in]     lda     The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    B       Complex*16 array, dimension (ldb, n).
 *                        On exit, B = A in the locations specified by uplo.
 * @param[in]     ldb     The leading dimension of the array B. ldb >= max(1,m).
 */
void zlacp2(
    const char* uplo,
    const int m,
    const int n,
    const double* const restrict A,
    const int lda,
    double complex* const restrict B,
    const int ldb)
{
    int i, j;

    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (j = 0; j < n; j++) {
            int imax = (j + 1 < m) ? j + 1 : m;
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
