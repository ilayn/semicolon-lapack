/**
 * @file sorgr2.c
 * @brief SORGR2 generates all or part of the orthogonal matrix Q from
 *        an RQ factorization determined by SGERQF (unblocked algorithm).
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SORGR2 generates an m by n real matrix Q with orthonormal rows,
 * which is defined as the last m rows of a product of k elementary
 * reflectors of order n
 *
 *    Q = H(0) H(1) . . . H(k-1)
 *
 * as returned by SGERQF.
 *
 * @param[in]     m     The number of rows of Q. m >= 0.
 * @param[in]     n     The number of columns of Q. n >= m.
 * @param[in]     k     The number of elementary reflectors whose product
 *                      defines Q. m >= k >= 0.
 * @param[in,out] A     On entry, the (m-k+i)-th row must contain the vector
 *                      which defines the elementary reflector H(i), for
 *                      i = 0,1,...,k-1, as returned by SGERQF in the last
 *                      k rows of its array argument A.
 *                      On exit, the m-by-n matrix Q.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau   Array of dimension (k). TAU(i) must contain the scalar
 *                      factor of the elementary reflector H(i), as returned
 *                      by SGERQF.
 * @param[out]    work  Workspace, dimension (m).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void sorgr2(const INT m, const INT n, const INT k,
            f32* restrict A, const INT lda,
            const f32* restrict tau,
            f32* restrict work,
            INT* info)
{
    INT i, ii, j, l;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

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
        xerbla("SORGR2", -(*info));
        return;
    }

    /* Quick return if possible */
    if (m <= 0) {
        return;
    }

    if (k < m) {
        /* Initialise rows 0:m-k-1 to rows of the unit matrix */
        for (j = 0; j < n; j++) {
            for (l = 0; l < m - k; l++) {
                A[l + j * lda] = ZERO;
            }
            if (j > n - m - 1 && j < n - k) {
                A[(m - n + j) + j * lda] = ONE;
            }
        }
    }

    for (i = 0; i < k; i++) {
        /* ii is the row index (0-based) for reflector H(i) */
        ii = m - k + i;

        /* Apply H(i) to A(0:ii-1, 0:n-m+ii) from the right */
        slarf1l("R", ii, n - m + ii + 1, &A[ii + 0 * lda], lda,
                tau[i], A, lda, work);
        cblas_sscal(n - m + ii, -tau[i], &A[ii + 0 * lda], lda);
        A[ii + (n - m + ii) * lda] = ONE - tau[i];

        /* Set A(ii, n-m+ii+1:n-1) to zero */
        for (l = n - m + ii + 1; l < n; l++) {
            A[ii + l * lda] = ZERO;
        }
    }
}
