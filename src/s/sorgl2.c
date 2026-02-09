/**
 * @file sorgl2.c
 * @brief SORGL2 generates all or part of the orthogonal matrix Q from
 *        an LQ factorization determined by SGELQF (unblocked algorithm).
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SORGL2 generates an m by n real matrix Q with orthonormal rows,
 * which is defined as the first m rows of a product of k elementary
 * reflectors of order n
 *
 *    Q = H(k-1) . . . H(1) H(0)
 *
 * as returned by SGELQF.
 *
 * @param[in]     m     The number of rows of Q. m >= 0.
 * @param[in]     n     The number of columns of Q. n >= m.
 * @param[in]     k     The number of elementary reflectors whose product
 *                      defines Q. m >= k >= 0.
 * @param[in,out] A     On entry, the i-th row must contain the vector which
 *                      defines the elementary reflector H(i), for
 *                      i = 0,1,...,k-1, as returned by SGELQF in the first
 *                      k rows of its array argument A.
 *                      On exit, the m-by-n matrix Q.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau   Array of dimension (k). TAU(i) must contain the scalar
 *                      factor of the elementary reflector H(i), as returned
 *                      by SGELQF.
 * @param[out]    work  Workspace, dimension (m).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 */
void sorgl2(const int m, const int n, const int k,
            float * const restrict A, const int lda,
            const float * const restrict tau,
            float * const restrict work,
            int *info)
{
    int i, j, l;
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    /* Parameter validation */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < m) {
        *info = -2;
    } else if (k < 0 || k > m) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("SORGL2", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m <= 0) {
        return;
    }

    if (k < m) {
        /* Initialise rows k:m-1 to rows of the unit matrix */
        for (j = 0; j < n; j++) {
            for (l = k; l < m; l++) {
                A[l + j * lda] = ZERO;
            }
            if (j >= k && j < m) {
                A[j + j * lda] = ONE;
            }
        }
    }

    for (i = k - 1; i >= 0; i--) {
        /* Apply H(i) to A(i+1:m-1, i:n-1) from the right */
        if (i < n - 1) {
            if (i < m - 1) {
                slarf1f("R", m - i - 1, n - i, &A[i + i * lda], lda,
                        tau[i], &A[(i + 1) + i * lda], lda, work);
            }
            cblas_sscal(n - i - 1, -tau[i], &A[i + (i + 1) * lda], lda);
        }
        A[i + i * lda] = ONE - tau[i];

        /* Set A(i, 0:i-1) to zero */
        for (l = 0; l < i; l++) {
            A[i + l * lda] = ZERO;
        }
    }
}
