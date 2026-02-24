/**
 * @file zlarge.c
 * @brief ZLARGE pre- and post-multiplies a complex general n by n matrix A
 *        with a random unitary matrix: A = U*D*U'.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlarge.f
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

extern void xerbla(const char* srname, const INT info);
extern void zlarnv_rng(const INT idist, const INT n, c128* x, uint64_t state[static 4]);

/**
 * ZLARGE pre- and post-multiplies a complex general n by n matrix A
 * with a random unitary matrix: A = U*A*U'.
 *
 * @param[in] n        The order of the matrix A. n >= 0.
 * @param[in,out] A    Complex array, dimension (lda, n).
 * @param[in] lda      The leading dimension of the array A. lda >= n.
 * @param[out] work    Complex workspace array of dimension (2*n).
 * @param[out] info    = 0: successful exit.
 * @param[in,out] state  RNG state array.
 */
void zlarge(const INT n, c128* A, const INT lda,
            c128* work, INT* info, uint64_t state[static 4])
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

    INT i;
    f64 wn, tau;
    c128 wa, wb;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -3;
    }
    if (*info < 0) {
        xerbla("ZLARGE", -(*info));
        return;
    }

    for (i = n - 1; i >= 0; i--) {
        INT len = n - i;

        zlarnv_rng(3, len, work, state);
        wn = cblas_dznrm2(len, work, 1);
        wa = (wn / cabs(work[0])) * work[0];
        if (wn == 0.0) {
            tau = 0.0;
        } else {
            wb = work[0] + wa;
            c128 scale = CONE / wb;
            cblas_zscal(len - 1, &scale, &work[1], 1);
            work[0] = CONE;
            tau = creal(wb / wa);
        }

        c128 neg_tau = CMPLX(-tau, 0.0);
        cblas_zgemv(CblasColMajor, CblasConjTrans, len, n, &CONE,
                    &A[i], lda, work, 1, &CZERO, &work[n], 1);
        cblas_zgerc(CblasColMajor, len, n, &neg_tau, work, 1,
                    &work[n], 1, &A[i], lda);

        cblas_zgemv(CblasColMajor, CblasNoTrans, n, len, &CONE,
                    &A[i * lda], lda, work, 1, &CZERO, &work[n], 1);
        cblas_zgerc(CblasColMajor, n, len, &neg_tau, &work[n], 1,
                    work, 1, &A[i * lda], lda);
    }
}
