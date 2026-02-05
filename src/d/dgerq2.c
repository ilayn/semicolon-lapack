/**
 * @file dgerq2.c
 * @brief DGERQ2 computes an RQ factorization of a general M-by-N matrix
 *        using an unblocked algorithm.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DGERQ2 computes an RQ factorization of a real m by n matrix A:
 *    A = R * Q.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *    Q = H(1) H(2) . . . H(k),   where k = min(m, n).
 *
 * Each H(i) has the form
 *    H(i) = I - tau * v * v**T
 * where tau is a real scalar, and v is a real vector with
 * v(n-k+i+1:n-1) = 0 and v(n-k+i) = 1; v(0:n-k+i-1) is stored on exit
 * in A(m-k+i, 0:n-k+i-1), and tau in TAU(i).
 *
 * @param[in]     m     The number of rows of A. m >= 0.
 * @param[in]     n     The number of columns of A. n >= 0.
 * @param[in,out] A     On entry, the m-by-n matrix A.
 *                      On exit, if m <= n, the upper triangle of the subarray
 *                      A(0:m-1, n-m:n-1) contains the m-by-m upper triangular
 *                      matrix R; if m >= n, the elements on and above the
 *                      (m-n)-th subdiagonal contain the m-by-n upper trapezoidal
 *                      matrix R; the remaining elements, with TAU, represent Q
 *                      as a product of elementary reflectors.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau   Array of dimension min(m, n). The scalar factors of the
 *                      elementary reflectors.
 * @param[out]    work  Workspace, dimension (m).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 */
void dgerq2(const int m, const int n,
            double * const restrict A, const int lda,
            double * const restrict tau,
            double * const restrict work,
            int *info)
{
    int i, k;
    double aii;

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
        xerbla("DGERQ2", -(*info));
        return;
    }

    k = m < n ? m : n;

    for (i = k - 1; i >= 0; i--) {
        /* Indices into the matrix for this reflector:
         * Row: m-k+i
         * Diagonal element: row m-k+i, col n-k+i
         * Reflector annihilates: cols 0:n-k+i-1 */

        /* Generate elementary reflector H(i) to annihilate A(m-k+i, 0:n-k+i-1) */
        dlarfg(n - k + i + 1, &A[(m - k + i) + (n - k + i) * lda],
               &A[(m - k + i) + 0 * lda], lda, &tau[i]);

        if (m - k + i > 0) {
            /* Apply H(i) to A(0:m-k+i-1, 0:n-k+i) from the right */
            aii = A[(m - k + i) + (n - k + i) * lda];
            A[(m - k + i) + (n - k + i) * lda] = 1.0;
            dlarf1l("R", m - k + i, n - k + i + 1,
                    &A[(m - k + i) + 0 * lda], lda, tau[i],
                    A, lda, work);
            A[(m - k + i) + (n - k + i) * lda] = aii;
        }
    }
}
