/**
 * @file zgetrf2.c
 * @brief Recursive LU factorization with threshold.
 */

#include <math.h>
#include <complex.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * Recursion threshold: for panels with n <= this value, use unblocked code.
 * This avoids function call overhead for small matrices.
 */
#define RECURSION_THRESHOLD 16

/**
 * Computes an LU factorization of a general M-by-N matrix A using partial
 * pivoting with row interchanges (recursive algorithm).
 *
 * The factorization has the form:
 *
 *     A = P * L * U
 *
 * where P is a permutation matrix, L is lower triangular with unit diagonal
 * elements, and U is upper triangular.
 *
 * This is a recursive version that achieves better cache utilization than the
 * iterative unblocked algorithm for medium-sized matrices. For small panels
 * (n <= RECURSION_THRESHOLD), it falls back to the unblocked zgetf2.
 *
 * @param[in]     m     The number of rows of the matrix A (m >= 0).
 * @param[in]     n     The number of columns of the matrix A (n >= 0).
 * @param[in,out] A     On entry, the M-by-N matrix to be factored.
 *                      On exit, the factors L and U from the factorization;
 *                      the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,m)).
 * @param[out]    ipiv  The pivot indices; row i was interchanged with row
 *                      ipiv[i]. Array of dimension min(m,n), 0-based.
 *
 * @param[out]   info
 *                           Exit status.
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but U is exactly singular.
 */
void zgetrf2(
    const int m,
    const int n,
    c128* const restrict A,
    const int lda,
    int* const restrict ipiv,
    int* info)
{
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);

    int i, iinfo, n1, n2;
    int minmn = m < n ? m : n;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZGETRF2", -(*info));
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    // Base case: use unblocked code for small panels
    // Check minmn, not n, because that's the actual factorization size
    if (minmn <= RECURSION_THRESHOLD) {
        zgetf2(m, n, A, lda, ipiv, info);
        return;
    }

    // Recursive case: split and conquer
    n1 = minmn / 2;
    n2 = n - n1;

    //        [ A11 ]
    // Factor [ --- ]
    //        [ A21 ]
    zgetrf2(m, n1, A, lda, ipiv, &iinfo);

    if (*info == 0 && iinfo > 0) {
        *info = iinfo;
    }

    //                       [ A12 ]
    // Apply interchanges to [ --- ]
    //                       [ A22 ]
    zlaswp(n2, A + n1 * lda, lda, 0, n1 - 1, ipiv, 1);

    // Solve A12
    cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                n1, n2, &ONE, A, lda, A + n1 * lda, lda);

    // Update A22
    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                m - n1, n2, n1, &NEG_ONE, A + n1, lda,
                A + n1 * lda, lda, &ONE, A + n1 * lda + n1, lda);

    // Factor A22
    zgetrf2(m - n1, n2, A + n1 * lda + n1, lda, &ipiv[n1], &iinfo);

    // Adjust INFO and the pivot indices
    if (*info == 0 && iinfo > 0) {
        *info = iinfo + n1;
    }
    for (i = n1; i < minmn; i++) {
        ipiv[i] = ipiv[i] + n1;
    }

    // Apply interchanges to A21
    zlaswp(n1, A, lda, n1, minmn - 1, ipiv, 1);
}
