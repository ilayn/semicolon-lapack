/**
 * @file cgerq2.c
 * @brief CGERQ2 computes an RQ factorization of a general M-by-N matrix
 *        using an unblocked algorithm.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGERQ2 computes an RQ factorization of a complex m by n matrix A:
 *    A = R * Q.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *    Q = H(1)**H H(2)**H . . . H(k),   where k = min(m, n).
 *
 * Each H(i) has the form
 *    H(i) = I - tau * v * v**H
 * where tau is a complex scalar, and v is a complex vector with
 * v(n-k+i+1:n-1) = 0 and v(n-k+i) = 1; conjg(v(0:n-k+i-1)) is stored on
 * exit in A(m-k+i, 0:n-k+i-1), and tau in TAU(i).
 *
 * @param[in]     m     The number of rows of A. m >= 0.
 * @param[in]     n     The number of columns of A. n >= 0.
 * @param[in,out] A     On entry, the m-by-n matrix A.
 *                      On exit, if m <= n, the upper triangle of the subarray
 *                      A(0:m-1, n-m:n-1) contains the m-by-m upper triangular
 *                      matrix R; if m >= n, the elements on and above the
 *                      (m-n)-th subdiagonal contain the m-by-n upper trapezoidal
 *                      matrix R; the remaining elements, with TAU, represent the
 *                      unitary matrix Q as a product of elementary reflectors.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau   Array of dimension min(m, n). The scalar factors of the
 *                      elementary reflectors.
 * @param[out]    work  Workspace, dimension (m).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cgerq2(const INT m, const INT n,
            c64* restrict A, const INT lda,
            c64* restrict tau,
            c64* restrict work,
            INT* info)
{
    INT i, k;

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
        xerbla("CGERQ2", -(*info));
        return;
    }

    k = m < n ? m : n;

    for (i = k - 1; i >= 0; i--) {

        clacgv(n - k + i + 1, &A[(m - k + i) + 0 * lda], lda);

        /* Generate elementary reflector H(i) to annihilate A(m-k+i, 0:n-k+i-1) */
        clarfg(n - k + i + 1, &A[(m - k + i) + (n - k + i) * lda],
               &A[(m - k + i) + 0 * lda], lda, &tau[i]);

        /* Apply H(i) to A(0:m-k+i-1, 0:n-k+i) from the right */
        clarf1l("R", m - k + i, n - k + i + 1,
                &A[(m - k + i) + 0 * lda], lda, tau[i],
                A, lda, work);

        clacgv(n - k + i, &A[(m - k + i) + 0 * lda], lda);
    }
}
