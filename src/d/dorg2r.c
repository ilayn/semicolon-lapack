/**
 * @file dorg2r.c
 * @brief DORG2R generates all or part of the orthogonal matrix Q from
 *        a QR factorization determined by DGEQRF (unblocked algorithm).
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DORG2R generates an m by n real matrix Q with orthonormal columns,
 * which is defined as the first n columns of a product of k elementary
 * reflectors of order m
 *
 *    Q = H(0) H(1) . . . H(k-1)
 *
 * as returned by DGEQRF.
 *
 * @param[in]     m     The number of rows of Q. m >= 0.
 * @param[in]     n     The number of columns of Q. m >= n >= 0.
 * @param[in]     k     The number of elementary reflectors whose product
 *                      defines Q. n >= k >= 0.
 * @param[in,out] A     On entry, the i-th column must contain the vector
 *                      which defines the elementary reflector H(i), for
 *                      i = 0,1,...,k-1, as returned by DGEQRF in the first
 *                      k columns of its array argument A.
 *                      On exit, the m-by-n matrix Q.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau   Array of dimension (k). TAU(i) must contain the scalar
 *                      factor of the elementary reflector H(i), as returned
 *                      by DGEQRF.
 * @param[out]    work  Workspace, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void dorg2r(const int m, const int n, const int k,
            f64 * const restrict A, const int lda,
            const f64 * const restrict tau,
            f64 * const restrict work,
            int *info)
{
    int i, j, l;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

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
        xerbla("DORG2R", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n <= 0) {
        return;
    }

    /* Initialise columns k:n-1 to columns of the unit matrix */
    for (j = k; j < n; j++) {
        for (l = 0; l < m; l++) {
            A[l + j * lda] = ZERO;
        }
        A[j + j * lda] = ONE;
    }

    for (i = k - 1; i >= 0; i--) {
        /* Apply H(i) to A(i:m-1, i+1:n-1) from the left */
        if (i < n - 1) {
            dlarf1f("L", m - i, n - i - 1, &A[i + i * lda], 1, tau[i],
                    &A[i + (i + 1) * lda], lda, work);
        }
        if (i < m - 1) {
            cblas_dscal(m - i - 1, -tau[i], &A[(i + 1) + i * lda], 1);
        }
        A[i + i * lda] = ONE - tau[i];

        /* Set A(0:i-1, i) to zero */
        for (l = 0; l < i; l++) {
            A[l + i * lda] = ZERO;
        }
    }
}
