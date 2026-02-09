/**
 * @file sgeqr2p.c
 * @brief SGEQR2P computes a QR factorization of a general M-by-N matrix
 *        with non-negative diagonal elements using an unblocked algorithm.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGEQR2P computes a QR factorization of a real m-by-n matrix A:
 *
 *    A = Q * ( R ),
 *            ( 0 )
 *
 * where:
 *    Q is a m-by-m orthogonal matrix;
 *    R is an upper-triangular n-by-n matrix with nonnegative diagonal
 *    entries;
 *    0 is a (m-n)-by-n zero matrix, if m > n.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *    Q = H(1) H(2) . . . H(k),   where k = min(m, n).
 *
 * Each H(i) has the form
 *    H(i) = I - tau * v * v**T
 * where tau is a real scalar, and v is a real vector with
 * v(0:i-1) = 0 and v(i) = 1; v(i+1:m-1) is stored on exit in A(i+1:m-1, i),
 * and tau in TAU(i).
 *
 * See Lapack Working Note 203 for details.
 *
 * @param[in]     m     The number of rows of A. m >= 0.
 * @param[in]     n     The number of columns of A. n >= 0.
 * @param[in,out] A     On entry, the m-by-n matrix A.
 *                      On exit, the elements on and above the diagonal contain
 *                      the min(m,n)-by-n upper trapezoidal matrix R (R is upper
 *                      triangular if m >= n). The diagonal entries of R are
 *                      nonnegative; the elements below the diagonal, with the
 *                      array TAU, represent the orthogonal matrix Q as a product
 *                      of elementary reflectors.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau   Array of dimension min(m, n). The scalar factors of the
 *                      elementary reflectors.
 * @param[out]    work  Workspace, dimension (n).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgeqr2p(const int m, const int n,
             float * const restrict A, const int lda,
             float * const restrict tau,
             float * const restrict work,
             int *info)
{
    int i, k;

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
        xerbla("SGEQR2P", -(*info));
        return;
    }

    k = m < n ? m : n;

    for (i = 0; i < k; i++) {
        /* Generate elementary reflector H(i) to annihilate A(i+1:m-1, i)
         * with non-negative diagonal (beta >= 0) */
        slarfgp(m - i, &A[i + i * lda],
                &A[((i + 1) < m ? (i + 1) : i) + i * lda], 1, &tau[i]);

        if (i < n - 1) {
            /* Apply H(i) to A(i:m-1, i+1:n-1) from the left */
            slarf1f("L", m - i, n - i - 1,
                    &A[i + i * lda], 1, tau[i],
                    &A[i + (i + 1) * lda], lda, work);
        }
    }
}
