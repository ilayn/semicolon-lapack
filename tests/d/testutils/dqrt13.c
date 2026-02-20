/**
 * @file dqrt13.c
 * @brief DQRT13 generates a full-rank matrix that may be scaled to have
 *        large or small norm.
 *
 * Faithful port of LAPACK TESTING/LIN/dqrt13.f
 * Uses xoshiro256+ RNG instead of LAPACK's ISEED array.
 */

#include <math.h>
#include <cblas.h>
#include "verify.h"
#include "test_rng.h"

/* Forward declarations */
extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern void dlascl(const char* type, const int kl, const int ku,
                   const f64 cfrom, const f64 cto,
                   const int m, const int n, f64* A, const int lda,
                   int* info);

/**
 * DQRT13 generates a full-rank matrix that may be scaled to have large
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
void dqrt13(const int scale, const int m, const int n,
            f64* A, const int lda, f64* norma,
            uint64_t state[static 4])
{
    const f64 ONE = 1.0;

    int info, j;
    f64 bignum, smlnum;
    (void)(m < n);  /* minmn was computed but unused */
    f64 dummy[1];

    if (m <= 0 || n <= 0) {
        return;
    }

    /* Generate benign matrix: random entries with diagonal dominance */
    for (j = 0; j < n; j++) {
        /* Fill column j with uniform(-1, 1) random values */
        rng_fill(state, 2, m, &A[j * lda]);

        /* Make diagonal dominant for well-conditioning */
        if (j < m) {
            f64 asum = cblas_dasum(m, &A[j * lda], 1);
            if (A[j + j * lda] >= 0.0) {
                A[j + j * lda] += asum;
            } else {
                A[j + j * lda] -= asum;
            }
        }
    }

    /* Scale the matrix if requested */
    if (scale != 1) {
        *norma = dlange("M", m, n, A, lda, dummy);
        smlnum = dlamch("S");  /* Safe minimum */
        bignum = ONE / smlnum;
        smlnum = smlnum / dlamch("E");  /* Safe minimum / epsilon */
        bignum = ONE / smlnum;

        if (scale == 2) {
            /* Matrix scaled up */
            dlascl("G", 0, 0, *norma, bignum, m, n, A, lda, &info);
        } else if (scale == 3) {
            /* Matrix scaled down */
            dlascl("G", 0, 0, *norma, smlnum, m, n, A, lda, &info);
        }
    }

    *norma = dlange("O", m, n, A, lda, dummy);
}
