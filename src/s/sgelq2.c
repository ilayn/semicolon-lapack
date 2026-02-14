/**
 * @file sgelq2.c
 * @brief SGELQ2 computes an LQ factorization of a general M-by-N matrix
 *        using an unblocked algorithm.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGELQ2 computes an LQ factorization of a real m by n matrix A:
 *    A = L * Q.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *    Q = H(k) . . . H(2) H(1),   where k = min(m, n).
 *
 * Each H(i) has the form
 *    H(i) = I - tau * v * v**T
 * where tau is a real scalar, and v is a real vector with
 * v(0:i-1) = 0 and v(i) = 1; v(i+1:n-1) is stored on exit in A(i, i+1:n-1),
 * and tau in TAU(i).
 *
 * @param[in]     m     The number of rows of A. m >= 0.
 * @param[in]     n     The number of columns of A. n >= 0.
 * @param[in,out] A     On entry, the m-by-n matrix A.
 *                      On exit, the elements on and below the diagonal contain
 *                      the m-by-min(m,n) lower trapezoidal matrix L; the elements
 *                      above the diagonal, with the array TAU, represent the
 *                      orthogonal matrix Q as a product of elementary reflectors.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau   Array of dimension min(m, n). The scalar factors of the
 *                      elementary reflectors.
 * @param[out]    work  Workspace, dimension (m).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sgelq2(const int m, const int n,
            f32* restrict A, const int lda,
            f32* restrict tau,
            f32* restrict work,
            int* info)
{
    int i, k;
    f32 aii;

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
        xerbla("SGELQ2", -(*info));
        return;
    }

    k = m < n ? m : n;

    for (i = 0; i < k; i++) {
        /* Generate elementary reflector H(i) to annihilate A(i, i+1:n-1) */
        slarfg(n - i, &A[i + i * lda],
               &A[i + ((i + 1) < n ? (i + 1) : i) * lda], lda, &tau[i]);

        if (i < m - 1) {
            /* Apply H(i) to A(i+1:m-1, i:n-1) from the right */
            aii = A[i + i * lda];
            A[i + i * lda] = 1.0f;
            slarf1f("R", m - i - 1, n - i,
                    &A[i + i * lda], lda, tau[i],
                    &A[(i + 1) + i * lda], lda, work);
            A[i + i * lda] = aii;
        }
    }
}
