/**
 * @file dorg2l.c
 * @brief DORG2L generates all or part of the orthogonal matrix Q from
 *        a QL factorization determined by DGEQLF (unblocked algorithm).
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DORG2L generates an m by n real matrix Q with orthonormal columns,
 * which is defined as the last n columns of a product of k elementary
 * reflectors of order m
 *
 *    Q = H(k-1) . . . H(1) H(0)
 *
 * as returned by DGEQLF.
 *
 * @param[in]     m     The number of rows of Q. m >= 0.
 * @param[in]     n     The number of columns of Q. m >= n >= 0.
 * @param[in]     k     The number of elementary reflectors whose product
 *                      defines Q. n >= k >= 0.
 * @param[in,out] A     On entry, the (n-k+i)-th column must contain the
 *                      vector which defines the elementary reflector H(i),
 *                      for i = 0,1,...,k-1, as returned by DGEQLF in the
 *                      last k columns of its array argument A.
 *                      On exit, the m-by-n matrix Q.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau   Array of dimension (k). TAU(i) must contain the scalar
 *                      factor of the elementary reflector H(i), as returned
 *                      by DGEQLF.
 * @param[out]    work  Workspace, dimension (n).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value.
 */
void dorg2l(const int m, const int n, const int k,
            double * const restrict A, const int lda,
            const double * const restrict tau,
            double * const restrict work,
            int *info)
{
    int i, ii, j, l;
    const double ZERO = 0.0;
    const double ONE = 1.0;

    /* Parameter validation */
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > m) {
        *info = -2;
    } else if (k < 0 || k > n) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("DORG2L", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    /* Initialise columns 0:n-k-1 to columns of the unit matrix */
    for (j = 0; j < n - k; j++) {
        for (l = 0; l < m; l++) {
            A[l + j * lda] = ZERO;
        }
        A[(m - n + j) + j * lda] = ONE;
    }

    for (i = 0; i < k; i++) {
        /* ii is the column index (0-based) for reflector H(i) */
        ii = n - k + i;

        /* Apply H(i) to A(0:m-n+ii, 0:ii-1) from the left */
        dlarf1l("L", m - n + ii + 1, ii, &A[0 + ii * lda], 1, tau[i],
                A, lda, work);
        cblas_dscal(m - n + ii, -tau[i], &A[0 + ii * lda], 1);
        A[(m - n + ii) + ii * lda] = ONE - tau[i];

        /* Set A(m-n+ii+1:m-1, ii) to zero */
        for (l = m - n + ii + 1; l < m; l++) {
            A[l + ii * lda] = ZERO;
        }
    }
}
