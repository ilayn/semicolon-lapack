/**
 * @file cungr2.c
 * @brief CUNGR2 generates all or part of the unitary matrix Q from
 *        an RQ factorization determined by CGERQF (unblocked algorithm).
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CUNGR2 generates an m by n complex matrix Q with orthonormal rows,
 * which is defined as the last m rows of a product of k elementary
 * reflectors of order n
 *
 *    Q = H(0)**H H(1)**H . . . H(k-1)**H
 *
 * as returned by CGERQF.
 *
 * @param[in]     m     The number of rows of Q. m >= 0.
 * @param[in]     n     The number of columns of Q. n >= m.
 * @param[in]     k     The number of elementary reflectors whose product
 *                      defines Q. m >= k >= 0.
 * @param[in,out] A     On entry, the (m-k+i)-th row must contain the vector
 *                      which defines the elementary reflector H(i), for
 *                      i = 0,1,...,k-1, as returned by CGERQF in the last
 *                      k rows of its array argument A.
 *                      On exit, the m-by-n matrix Q.
 * @param[in]     lda   The leading dimension of A. lda >= max(1, m).
 * @param[in]     tau   Array of dimension (k). TAU(i) must contain the scalar
 *                      factor of the elementary reflector H(i), as returned
 *                      by CGERQF.
 * @param[out]    work  Workspace, dimension (m).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cungr2(const int m, const int n, const int k,
            c64* restrict A, const int lda,
            const c64* restrict tau,
            c64* restrict work,
            int* info)
{
    int i, ii, j, l;
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const c64 ONE = CMPLXF(1.0f, 0.0f);

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
        xerbla("CUNGR2", -(*info));
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
            if (j >= n - m && j < n - k) {
                A[(m - n + j) + j * lda] = ONE;
            }
        }
    }

    for (i = 0; i < k; i++) {
        ii = m - k + i;

        /* Apply H(i)**H to A(0:ii-1, 0:n-m+ii) from the right */
        clacgv(n - m + ii, &A[ii + 0 * lda], lda);
        c64 conjtau = conjf(tau[i]);
        clarf1l("R", ii, n - m + ii + 1, &A[ii + 0 * lda], lda,
                conjtau, A, lda, work);
        c64 neg_tau = -tau[i];
        cblas_cscal(n - m + ii, &neg_tau, &A[ii + 0 * lda], lda);
        clacgv(n - m + ii, &A[ii + 0 * lda], lda);
        A[ii + (n - m + ii) * lda] = ONE - conjf(tau[i]);

        /* Set A(ii, n-m+ii+1:n-1) to zero */
        for (l = n - m + ii + 1; l < n; l++) {
            A[ii + l * lda] = ZERO;
        }
    }
}
