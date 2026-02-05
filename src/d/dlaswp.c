/**
 * @file dlaswp.c
 * @brief Row interchanges for pivoting.
 */

#include "semicolon_lapack_double.h"

/**
 * Performs a series of row interchanges on a general rectangular matrix.
 *
 * Interchanges row i with row ipiv[k1 + (i - k1) * |incx|] for each of
 * rows k1 through k2 of A.
 *
 * @param[in]     n     The number of columns of the matrix A.
 * @param[in,out] A     On entry, the M-by-N matrix to which the row
 *                      interchanges will be applied.
 *                      On exit, the permuted matrix.
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,m)).
 * @param[in]     k1    The first row of A to which a row interchange will be
 *                      applied (0-based).
 * @param[in]     k2    The last row of A to which a row interchange will be
 *                      applied (0-based, k2 >= k1).
 * @param[in]     ipiv  The vector of pivot indices. Row i is interchanged with
 *                      row ipiv[k1 + (i - k1) * |incx|]. Indices are 0-based.
 * @param[in]     incx  The increment between successive values of ipiv.
 *                      If incx > 0, pivots are applied from k1 to k2.
 *                      If incx < 0, pivots are applied from k2 to k1.
 *                      If incx = 0, returns immediately.
 *
 * @rst
 * .. note::
 *
 *    This function is typically called after :c:func:`dgetrf` or
 *    :c:func:`dgetrf2` to apply the same row permutations to other matrices
 *    (e.g., the right-hand side matrix B when solving Ax = B).
 *
 *    The implementation processes columns in blocks of 32 for cache efficiency.
 * @endrst
 */
void dlaswp(
    const int n,
    double * const restrict A,
    const int lda,
    const int k1,
    const int k2,
    const int * const restrict ipiv,
    const int incx)
{
    int i, i1, i2, inc, ip, ix, ix0, j, k, n32;
    double temp;

    // Interchange row i with row ipiv[k1 + (i - k1) * abs(incx)] for each of
    // rows k1 through k2. All indices are 0-based.

    if (incx > 0) {
        ix0 = k1;
        i1 = k1;
        i2 = k2;
        inc = 1;
    } else if (incx < 0) {
        ix0 = k1 + (k1 - k2) * incx;
        i1 = k2;
        i2 = k1;
        inc = -1;
    } else {
        return;
    }

    // Process columns in blocks of 32 for cache efficiency
    n32 = n & ~31;

    // Main loop over column blocks
    for (j = 0; j < n32; j += 32) {
        ix = ix0;
        for (i = i1; inc > 0 ? i <= i2 : i >= i2; i += inc) {
            ip = ipiv[ix];
            if (ip != i) {
                for (k = j; k < j + 32; k++) {
                    temp = A[i + k * lda];
                    A[i + k * lda] = A[ip + k * lda];
                    A[ip + k * lda] = temp;
                }
            }
            ix += incx;
        }
    }

    // Handle remaining columns, if any
    if (n32 < n) {
        ix = ix0;
        for (i = i1; inc > 0 ? i <= i2 : i >= i2; i += inc) {
            ip = ipiv[ix];
            if (ip != i) {
                for (k = n32; k < n; k++) {
                    temp = A[i + k * lda];
                    A[i + k * lda] = A[ip + k * lda];
                    A[ip + k * lda] = temp;
                }
            }
            ix += incx;
        }
    }
}
