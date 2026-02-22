/**
 * @file zung2l.c
 * @brief ZUNG2L generates all or part of the unitary matrix Q from
 *        a QL factorization determined by ZGEQLF (unblocked algorithm).
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZUNG2L generates an m by n complex matrix Q with orthonormal columns,
 * which is defined as the last n columns of a product of k elementary
 * reflectors of order m
 *
 *    Q = H(k-1) . . . H(1) H(0)
 *
 * as returned by ZGEQLF.
 *
 * @param[in]     m     The number of rows of Q. m >= 0.
 * @param[in]     n     The number of columns of Q. m >= n >= 0.
 * @param[in]     k     The number of elementary reflectors whose product
 *                      defines Q. n >= k >= 0.
 * @param[in,out] A     On entry, the (n-k+i)-th column must contain the
 *                      vector which defines the elementary reflector H(i),
 *                      for i = 0,1,...,k-1, as returned by ZGEQLF in the
 *                      last k columns of its array argument A.
 *                      On exit, the m-by-n matrix Q.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau   Array of dimension (k). TAU(i) must contain the scalar
 *                      factor of the elementary reflector H(i), as returned
 *                      by ZGEQLF.
 * @param[out]    work  Workspace, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zung2l(const INT m, const INT n, const INT k,
            c128* restrict A, const INT lda,
            const c128* restrict tau,
            c128* restrict work,
            INT* info)
{
    INT i, ii, j, l;
    const c128 ZERO = CMPLX(0.0, 0.0);
    const c128 ONE = CMPLX(1.0, 0.0);

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
        xerbla("ZUNG2L", -(*info));
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
        ii = n - k + i;

        /* Apply H(i) to A(0:m-n+ii, 0:ii-1) from the left */
        A[(m - n + ii) + ii * lda] = ONE;
        zlarf1l("L", m - n + ii + 1, ii, &A[0 + ii * lda], 1, tau[i],
                A, lda, work);
        c128 neg_tau = -tau[i];
        cblas_zscal(m - n + ii, &neg_tau, &A[0 + ii * lda], 1);
        A[(m - n + ii) + ii * lda] = ONE - tau[i];

        /* Set A(m-n+ii+1:m-1, ii) to zero */
        for (l = m - n + ii + 1; l < m; l++) {
            A[l + ii * lda] = ZERO;
        }
    }
}
