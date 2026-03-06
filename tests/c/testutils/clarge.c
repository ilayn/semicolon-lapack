/**
 * @file clarge.c
 * @brief CLARGE pre- and post-multiplies a complex general n by n matrix A
 *        with a random unitary matrix: A = U*D*U'.
 *
 * Faithful port of LAPACK TESTING/MATGEN/clarge.f
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * CLARGE pre- and post-multiplies a complex general n by n matrix A
 * with a random unitary matrix: A = U*A*U'.
 *
 * @param[in] n        The order of the matrix A. n >= 0.
 * @param[in,out] A    Complex array, dimension (lda, n).
 * @param[in] lda      The leading dimension of the array A. lda >= n.
 * @param[out] work    Complex workspace array of dimension (2*n).
 * @param[out] info    = 0: successful exit.
 * @param[in,out] state  RNG state array.
 */
void clarge(const INT n, c64* A, const INT lda,
            c64* work, INT* info, uint64_t state[static 4])
{
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);

    INT i;
    f32 wn, tau;
    c64 wa, wb;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -3;
    }
    if (*info < 0) {
        xerbla("CLARGE", -(*info));
        return;
    }

    for (i = n - 1; i >= 0; i--) {
        INT len = n - i;

        clarnv_rng(3, len, work, state);
        wn = cblas_scnrm2(len, work, 1);
        wa = (wn / cabsf(work[0])) * work[0];
        if (wn == 0.0f) {
            tau = 0.0f;
        } else {
            wb = work[0] + wa;
            c64 scale = CONE / wb;
            cblas_cscal(len - 1, &scale, &work[1], 1);
            work[0] = CONE;
            tau = crealf(wb / wa);
        }

        c64 neg_tau = CMPLXF(-tau, 0.0f);
        cblas_cgemv(CblasColMajor, CblasConjTrans, len, n, &CONE,
                    &A[i], lda, work, 1, &CZERO, &work[n], 1);
        cblas_cgerc(CblasColMajor, len, n, &neg_tau, work, 1,
                    &work[n], 1, &A[i], lda);

        cblas_cgemv(CblasColMajor, CblasNoTrans, n, len, &CONE,
                    &A[i * lda], lda, work, 1, &CZERO, &work[n], 1);
        cblas_cgerc(CblasColMajor, n, len, &neg_tau, &work[n], 1,
                    work, 1, &A[i * lda], lda);
    }
}
