/**
 * @file cgeql2.c
 * @brief CGEQL2 computes a QL factorization of a general M-by-N matrix
 *        using an unblocked algorithm.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGEQL2 computes a QL factorization of a complex m by n matrix A:
 *    A = Q * L.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *    Q = H(k) . . . H(2) H(1),   where k = min(m, n).
 *
 * Each H(i) has the form
 *    H(i) = I - tau * v * v**H
 * where tau is a complex scalar, and v is a complex vector with
 * v(m-k+i+1:m-1) = 0 and v(m-k+i) = 1; v(0:m-k+i-1) is stored on exit
 * in A(0:m-k+i-1, n-k+i), and tau in TAU(i).
 *
 * @param[in]     m     The number of rows of A. m >= 0.
 * @param[in]     n     The number of columns of A. n >= 0.
 * @param[in,out] A     On entry, the m-by-n matrix A.
 *                      On exit, if m >= n, the lower triangle of the subarray
 *                      A(m-n:m-1, 0:n-1) contains the n-by-n lower triangular
 *                      matrix L; if m <= n, the elements on and below the
 *                      (n-m)-th superdiagonal contain the m-by-n lower
 *                      trapezoidal matrix L; the remaining elements, with TAU,
 *                      represent Q as a product of elementary reflectors.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau   Array of dimension min(m, n). The scalar factors of the
 *                      elementary reflectors.
 * @param[out]    work  Workspace, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cgeql2(const int m, const int n,
            c64* restrict A, const int lda,
            c64* restrict tau,
            c64* restrict work,
            int* info)
{
    int i, k;
    c64 aii;

    /* Parameter validation */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CGEQL2", -(*info));
        return;
    }

    k = m < n ? m : n;

    for (i = k - 1; i >= 0; i--) {
        /* Indices into the matrix for this reflector:
         * Column: n-k+i
         * Diagonal element: row m-k+i, col n-k+i
         * Reflector annihilates: rows 0:m-k+i-1 */

        /* Generate elementary reflector H(i) to annihilate A(0:m-k+i-1, n-k+i) */
        clarfg(m - k + i + 1, &A[(m - k + i) + (n - k + i) * lda],
               &A[0 + (n - k + i) * lda], 1, &tau[i]);

        if (n - k + i > 0) {
            /* Apply H(i)**H to A(0:m-k+i, 0:n-k+i-1) from the left */
            aii = A[(m - k + i) + (n - k + i) * lda];
            A[(m - k + i) + (n - k + i) * lda] = CMPLXF(1.0f, 0.0f);
            clarf1l("L", m - k + i + 1, n - k + i,
                    &A[0 + (n - k + i) * lda], 1, conjf(tau[i]),
                    A, lda, work);
            A[(m - k + i) + (n - k + i) * lda] = aii;
        }
    }
}
