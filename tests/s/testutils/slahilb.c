/**
 * @file slahilb.c
 * @brief SLAHILB generates an N by N scaled Hilbert matrix in A along with
 *        NRHS right-hand sides in B and solutions in X such that A*X=B.
 */

#include "verify.h"


/*
 * NMAX_EXACT   the largest dimension where the generated data is exact.
 * NMAX_APPROX  the largest dimension where the generated data has
 *              a small componentwise relative error.
 */
#define NMAX_EXACT   6
#define NMAX_APPROX  11

/**
 * SLAHILB generates an N by N scaled Hilbert matrix in A along with
 * NRHS right-hand sides in B and solutions in X such that A*X=B.
 *
 * The Hilbert matrix is scaled by M = LCM(1, 2, ..., 2*N-1) so that all
 * entries are integers.  The right-hand sides are the first NRHS
 * columns of M * the identity matrix, and the solutions are the
 * first NRHS columns of the inverse Hilbert matrix.
 *
 * The condition number of the Hilbert matrix grows exponentially with
 * its size, roughly as O(e ** (3.5*N)).  Additionally, the inverse
 * Hilbert matrices beyond a relatively small dimension cannot be
 * generated exactly without extra precision.  Precision is exhausted
 * when the largest entry in the inverse Hilbert matrix is greater than
 * 2 to the power of the number of bits in the fraction of the data type
 * used plus one, which is 24 for single precision.
 *
 * In f64, the generated solution is exact for N <= 6 and has
 * small componentwise error for 7 <= N <= 11.
 *
 * @param[in] n
 *     The dimension of the matrix A.
 *
 * @param[in] nrhs
 *     The requested number of right-hand sides.
 *
 * @param[out] A
 *     Double precision array, dimension (lda, n).
 *     The generated scaled Hilbert matrix.
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= n.
 *
 * @param[out] X
 *     Double precision array, dimension (ldx, nrhs).
 *     The generated exact solutions. Currently, the first NRHS
 *     columns of the inverse Hilbert matrix.
 *
 * @param[in] ldx
 *     The leading dimension of the array X. ldx >= n.
 *
 * @param[out] B
 *     Double precision array, dimension (ldb, nrhs).
 *     The generated right-hand sides. Currently, the first NRHS
 *     columns of LCM(1, 2, ..., 2*N-1) * the identity matrix.
 *
 * @param[in] ldb
 *     The leading dimension of the array B. ldb >= n.
 *
 * @param[out] work
 *     Double precision array, dimension (n).
 *
 * @param[out] info
 *     = 0: successful exit
 *     = 1: N is too large; the data is still generated but may not
 *          be not exact.
 *     < 0: if info = -i, the i-th argument had an illegal value
 */
void slahilb(const INT n, const INT nrhs,
             f32* A, const INT lda,
             f32* X, const INT ldx,
             f32* B, const INT ldb,
             f32* work, INT* info)
{
    INT tm, ti, r;
    INT m;
    INT i, j;

    *info = 0;
    if (n < 0 || n > NMAX_APPROX) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (lda < n) {
        *info = -4;
    } else if (ldx < n) {
        *info = -6;
    } else if (ldb < n) {
        *info = -8;
    }
    if (*info < 0) {
        xerbla("SLAHILB", -(*info));
        return;
    }
    if (n > NMAX_EXACT) {
        *info = 1;
    }

    /* Compute M = the LCM of the integers [1, 2*N-1]. The largest
     * reasonable N is small enough that integers suffice (up to N = 11). */
    m = 1;
    for (i = 2; i <= 2 * n - 1; i++) {
        tm = m;
        ti = i;
        r = tm % ti;
        while (r != 0) {
            tm = ti;
            ti = r;
            r = tm % ti;
        }
        m = (m / ti) * i;
    }

    /* Generate the scaled Hilbert matrix in A */
    for (j = 0; j < n; j++) {
        for (i = 0; i < n; i++) {
            A[i + j * lda] = (f32)m / (i + j + 1);
        }
    }

    /* Generate matrix B as simply the first NRHS columns of M * the
     * identity. */
    slaset("Full", n, nrhs, 0.0f, (f32)m, B, ldb);

    /* Generate the true solutions in X. Because B = the first NRHS
     * columns of M*I, the true solutions are just the first NRHS columns
     * of the inverse Hilbert matrix. */
    work[0] = (f32)n;
    for (j = 1; j < n; j++) {
        /* Fortran: WORK(J) = ((WORK(J-1)/(J-1)) * (J-1 - N) / (J-1)) * (N+J-1)
         * With 0-based indexing: j_f = j+1, so j_f-1 = j
         * WORK(j+1) = ((WORK(j)/(j)) * (j - N) / (j)) * (N + j)
         */
        work[j] = ((work[j - 1] / (f32)j) * (f32)(j - n) / (f32)j)
                  * (f32)(n + j);
    }

    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            X[i + j * ldx] = (work[i] * work[j]) / (i + j + 1);
        }
    }
}
