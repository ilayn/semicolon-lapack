/**
 * @file cgeqr2.c
 * @brief CGEQR2 computes a QR factorization of a general M-by-N matrix
 *        using an unblocked algorithm.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CGEQR2 computes a QR factorization of a complex m by n matrix A:
 *    A = Q * R.
 *
 * The matrix Q is represented as a product of elementary reflectors
 *    Q = H(1) H(2) . . . H(k),   where k = min(m, n).
 *
 * Each H(i) has the form
 *    H(i) = I - tau * v * v**H
 * where tau is a complex scalar, and v is a complex vector with
 * v(0:i-1) = 0 and v(i) = 1; v(i+1:m-1) is stored on exit in A(i+1:m-1, i),
 * and tau in TAU(i).
 *
 * @param[in]     m     The number of rows of A. m >= 0.
 * @param[in]     n     The number of columns of A. n >= 0.
 * @param[in,out] A     On entry, the m-by-n matrix A.
 *                      On exit, the elements on and above the diagonal contain
 *                      the min(m,n)-by-n upper trapezoidal matrix R; the elements
 *                      below the diagonal, with the array TAU, represent the
 *                      unitary matrix Q as a product of elementary reflectors.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[out]    tau   Array of dimension min(m, n). The scalar factors of the
 *                      elementary reflectors.
 * @param[out]    work  Workspace, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cgeqr2(const INT m, const INT n,
            c64* restrict A, const INT lda,
            c64* restrict tau,
            c64* restrict work,
            INT* info)
{
    INT i, k;
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
        xerbla("CGEQR2", -(*info));
        return;
    }

    k = m < n ? m : n;

    for (i = 0; i < k; i++) {
        /* Generate elementary reflector H(i) to annihilate A(i+1:m-1, i) */
        clarfg(m - i, &A[i + i * lda],
               &A[((i + 1) < m ? (i + 1) : i) + i * lda], 1, &tau[i]);

        if (i < n - 1) {
            /* Apply H(i)**H to A(i:m-1, i+1:n-1) from the left */
            aii = A[i + i * lda];
            A[i + i * lda] = 1.0f;
            c64 conjtau = conjf(tau[i]);
            clarf1f("L", m - i, n - i - 1,
                    &A[i + i * lda], 1, conjtau,
                    &A[i + (i + 1) * lda], lda, work);
            A[i + i * lda] = aii;
        }
    }
}
