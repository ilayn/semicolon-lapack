/**
 * @file cungl2.c
 * @brief CUNGL2 generates all or part of the unitary matrix Q from an LQ
 *        factorization determined by CGELQF (unblocked algorithm).
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CUNGL2 generates an m-by-n complex matrix Q with orthonormal rows,
 * which is defined as the first m rows of a product of k elementary
 * reflectors of order n
 *
 *       Q  =  H(k)**H . . . H(2)**H H(1)**H
 *
 * as returned by CGELQF.
 *
 * @param[in]     m     The number of rows of the matrix Q. m >= 0.
 * @param[in]     n     The number of columns of the matrix Q. n >= m.
 * @param[in]     k     The number of elementary reflectors whose product defines the
 *                      matrix Q. m >= k >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the i-th row must contain the vector which defines
 *                      the elementary reflector H(i), for i = 0,1,...,k-1, as returned
 *                      by CGELQF in the first k rows of its array argument A.
 *                      On exit, the m by n matrix Q.
 * @param[in]     lda   The first dimension of the array A. lda >= max(1, m).
 * @param[in]     tau   Single complex array, dimension (k).
 *                      TAU(i) must contain the scalar factor of the elementary
 *                      reflector H(i), as returned by CGELQF.
 * @param[out]    work  Single complex array, dimension (m).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void cungl2(const INT m, const INT n, const INT k,
            c64* restrict A, const INT lda,
            const c64* restrict tau,
            c64* restrict work,
            INT* info)
{
    INT i, j, l;
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const c64 ZERO = CMPLXF(0.0f, 0.0f);

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
        xerbla("CUNGL2", -(*info));
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
            clacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
            if (i < m - 1) {
                c64 conj_tau = conjf(tau[i]);
                clarf1f("R", m - i - 1, n - i, &A[i + i * lda], lda,
                        conj_tau, &A[(i + 1) + i * lda], lda, work);
            }
            c64 neg_tau = -tau[i];
            cblas_cscal(n - i - 1, &neg_tau, &A[i + (i + 1) * lda], lda);
            clacgv(n - i - 1, &A[i + (i + 1) * lda], lda);
        }
        A[i + i * lda] = ONE - conjf(tau[i]);

        /* Set A(i, 0:i-1) to zero */
        for (l = 0; l < i; l++) {
            A[i + l * lda] = ZERO;
        }
    }
}
