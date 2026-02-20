/**
 * @file dlarge.c
 * @brief DLARGE pre- and post-multiplies a real general n by n matrix A
 *        with a random orthogonal matrix: A = U*A*U'.
 *
 * Faithful port of LAPACK TESTING/MATGEN/dlarge.f
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declarations */
extern void xerbla(const char* srname, const int info);

/**
 * DLARGE pre- and post-multiplies a real general n by n matrix A
 * with a random orthogonal matrix: A = U*A*U'.
 *
 * @param[in] n
 *     The order of the matrix A.  n >= 0.
 *
 * @param[in,out] A
 *     Double precision array, dimension (lda, n).
 *     On entry, the original n by n matrix A.
 *     On exit, A is overwritten by U*A*U' for some random
 *     orthogonal matrix U.
 *
 * @param[in] lda
 *     The leading dimension of the array A.  lda >= n.
 *
 * @param[out] work
 *     Double precision array, dimension (2*n).
 *
 * @param[out] info
 *     = 0: successful exit
 *     < 0: if info = -i, the i-th argument had an illegal value
 *
 * @param[in,out] state
 *     On entry, the state of the random number generator.
 *     On exit, the state is updated.
 */
void dlarge(const int n, f64* A, const int lda,
            f64* work, int* info, uint64_t state[static 4])
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int i;
    f64 tau, wa, wb, wn;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -3;
    }
    if (*info < 0) {
        xerbla("DLARGE", -*info);
        return;
    }

    for (i = n - 1; i >= 0; i--) {

        dlarnv_rng(3, n - i, work, state);
        wn = cblas_dnrm2(n - i, work, 1);
        wa = (work[0] >= 0.0) ? wn : -wn;
        if (wn == ZERO) {
            tau = ZERO;
        } else {
            wb = work[0] + wa;
            cblas_dscal(n - i - 1, ONE / wb, work + 1, 1);
            work[0] = ONE;
            tau = wb / wa;
        }

        cblas_dgemv(CblasColMajor, CblasTrans, n - i, n, ONE,
                    A + i, lda, work, 1, ZERO, work + n, 1);
        cblas_dger(CblasColMajor, n - i, n, -tau, work, 1,
                   work + n, 1, A + i, lda);

        cblas_dgemv(CblasColMajor, CblasNoTrans, n, n - i, ONE,
                    A + i * lda, lda, work, 1, ZERO, work + n, 1);
        cblas_dger(CblasColMajor, n, n - i, -tau, work + n, 1,
                   work, 1, A + i * lda, lda);
    }
}
