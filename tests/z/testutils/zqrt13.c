/**
 * @file zqrt13.c
 * @brief ZQRT13 generates a full-rank matrix that may be scaled to have
 *        large or small norm.
 *
 * Faithful port of LAPACK TESTING/LIN/zqrt13.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * ZQRT13 generates a full-rank matrix that may be scaled to have large
 * or small norm.
 *
 * @param[in] scale
 *     SCALE = 1: normally scaled matrix
 *     SCALE = 2: matrix scaled up
 *     SCALE = 3: matrix scaled down
 *
 * @param[in] m
 *     The number of rows of the matrix A.
 *
 * @param[in] n
 *     The number of columns of A.
 *
 * @param[out] A
 *     The M-by-N matrix A.
 *
 * @param[in] lda
 *     The leading dimension of the array A.
 *
 * @param[out] norma
 *     The one-norm of A.
 */
void zqrt13(const INT scale, const INT m, const INT n,
            c128* A, const INT lda, f64* norma,
            uint64_t state[static 4])
{
    const f64 ONE = 1.0;

    INT info, j;
    f64 bignum, smlnum;
    f64 dummy[1];

    if (m <= 0 || n <= 0) {
        return;
    }

    /* benign matrix */
    for (j = 0; j < n; j++) {
        zlarnv_rng(2, m, &A[j * lda], state);
        if (j < m) {
            f64 asum = cblas_dzasum(m, &A[j * lda], 1);
            f64 re_ajj = creal(A[j + j * lda]);
            if (re_ajj >= 0.0) {
                A[j + j * lda] = A[j + j * lda] + CMPLX(asum, 0.0);
            } else {
                A[j + j * lda] = A[j + j * lda] - CMPLX(asum, 0.0);
            }
        }
    }

    /* scaled versions */
    if (scale != 1) {
        *norma = zlange("M", m, n, A, lda, dummy);
        smlnum = dlamch("S");
        bignum = ONE / smlnum;
        smlnum = smlnum / dlamch("E");
        bignum = ONE / smlnum;

        if (scale == 2) {
            /* matrix scaled up */
            zlascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
        } else if (scale == 3) {
            /* matrix scaled down */
            zlascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
        }
    }

    *norma = zlange("O", m, n, A, lda, dummy);
}
