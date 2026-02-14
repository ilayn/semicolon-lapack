/**
 * @file zungl2.c
 * @brief ZUNGL2 generates all or part of the unitary matrix Q from an LQ
 *        factorization determined by ZGELQF (unblocked algorithm).
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZUNGL2 generates an m-by-n complex matrix Q with orthonormal rows,
 * which is defined as the first m rows of a product of k elementary
 * reflectors of order n
 *
 *       Q  =  H(k)**H . . . H(2)**H H(1)**H
 *
 * as returned by ZGELQF.
 *
 * @param[in]     m     The number of rows of the matrix Q. m >= 0.
 * @param[in]     n     The number of columns of the matrix Q. n >= m.
 * @param[in]     k     The number of elementary reflectors whose product defines the
 *                      matrix Q. m >= k >= 0.
 * @param[in,out] A     Double complex array, dimension (lda, n).
 *                      On entry, the i-th row must contain the vector which defines
 *                      the elementary reflector H(i), for i = 0,1,...,k-1, as returned
 *                      by ZGELQF in the first k rows of its array argument A.
 *                      On exit, the m by n matrix Q.
 * @param[in]     lda   The first dimension of the array A. lda >= max(1, m).
 * @param[in]     tau   Double complex array, dimension (k).
 *                      TAU(i) must contain the scalar factor of the elementary
 *                      reflector H(i), as returned by ZGELQF.
 * @param[out]    work  Double complex array, dimension (m).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zungl2(const int m, const int n, const int k,
            c128* restrict A, const int lda,
            const c128* restrict tau,
            c128* restrict work,
            int* info)
{
    int i, j, l;
    const c128 ONE = CMPLX(1.0, 0.0);
    const c128 ZERO = CMPLX(0.0, 0.0);

    /* Test the input arguments */
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
        xerbla("ZUNGL2", -(*info));
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
        /* Apply H(i)**H to A(i:m-1, i:n-1) from the right */
        if (i < n - 1) {
            zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
            if (i < m - 1) {
                c128 conj_tau = conj(tau[i]);
                zlarf1f("R", m - i - 1, n - i, &A[i + i * lda], lda,
                        conj_tau, &A[(i + 1) + i * lda], lda, work);
            }
            c128 neg_tau = -tau[i];
            cblas_zscal(n - i - 1, &neg_tau, &A[i + (i + 1) * lda], lda);
            zlacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
        }
        A[i + i * lda] = ONE - conj(tau[i]);

        /* Set A(i, 0:i-1) to zero */
        for (l = 0; l < i; l++) {
            A[i + l * lda] = ZERO;
        }
    }
}
