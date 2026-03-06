/**
 * @file cqrt13.c
 * @brief CQRT13 generates a full-rank matrix that may be scaled to have
 *        large or small norm.
 *
 * Faithful port of LAPACK TESTING/LIN/cqrt13.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * CQRT13 generates a full-rank matrix that may be scaled to have large
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
void cqrt13(const INT scale, const INT m, const INT n,
            c64* A, const INT lda, f32* norma,
            uint64_t state[static 4])
{
    const f32 ONE = 1.0f;

    INT info, j;
    f32 bignum, smlnum;
    f32 dummy[1];

    if (m <= 0 || n <= 0) {
        return;
    }

    /* benign matrix */
    for (j = 0; j < n; j++) {
        clarnv_rng(2, m, &A[j * lda], state);
        if (j < m) {
            f32 asum = cblas_scasum(m, &A[j * lda], 1);
            f32 re_ajj = crealf(A[j + j * lda]);
            if (re_ajj >= 0.0f) {
                A[j + j * lda] = A[j + j * lda] + CMPLXF(asum, 0.0f);
            } else {
                A[j + j * lda] = A[j + j * lda] - CMPLXF(asum, 0.0f);
            }
        }
    }

    /* scaled versions */
    if (scale != 1) {
        *norma = clange("M", m, n, A, lda, dummy);
        smlnum = slamch("S");
        bignum = ONE / smlnum;
        smlnum = smlnum / slamch("E");
        bignum = ONE / smlnum;

        if (scale == 2) {
            /* matrix scaled up */
            clascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
        } else if (scale == 3) {
            /* matrix scaled down */
            clascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
        }
    }

    *norma = clange("O", m, n, A, lda, dummy);
}
