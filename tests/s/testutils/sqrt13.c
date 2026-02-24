/**
 * @file sqrt13.c
 * @brief SQRT13 generates a full-rank matrix that may be scaled to have
 *        large or small norm.
 *
 * Faithful port of LAPACK TESTING/LIN/sqrt13.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
#include "test_rng.h"

/**
 * SQRT13 generates a full-rank matrix that may be scaled to have large
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
void sqrt13(const INT scale, const INT m, const INT n,
            f32* A, const INT lda, f32* norma,
            uint64_t state[static 4])
{
    const f32 ONE = 1.0f;

    INT info, j;
    f32 bignum, smlnum;
    (void)(m < n);  /* minmn was computed but unused */
    f32 dummy[1];

    if (m <= 0 || n <= 0) {
        return;
    }

    /* Generate benign matrix: random entries with diagonal dominance */
    for (j = 0; j < n; j++) {
        /* Fill column j with uniform(-1, 1) random values */
        rng_fill_f32(state, 2, m, &A[j * lda]);

        /* Make diagonal dominant for well-conditioning */
        if (j < m) {
            f32 asum = cblas_sasum(m, &A[j * lda], 1);
            if (A[j + j * lda] >= 0.0f) {
                A[j + j * lda] += asum;
            } else {
                A[j + j * lda] -= asum;
            }
        }
    }

    /* Scale the matrix if requested */
    if (scale != 1) {
        *norma = slange("M", m, n, A, lda, dummy);
        smlnum = slamch("S");  /* Safe minimum */
        bignum = ONE / smlnum;
        smlnum = smlnum / slamch("E");  /* Safe minimum / epsilon */
        bignum = ONE / smlnum;

        if (scale == 2) {
            /* Matrix scaled up */
            slascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
        } else if (scale == 3) {
            /* Matrix scaled down */
            slascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
        }
    }

    *norma = slange("O", m, n, A, lda, dummy);
}
