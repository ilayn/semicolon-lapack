/**
 * @file zlahilb.c
 * @brief ZLAHILB generates an N by N scaled Hilbert matrix in A along with
 *        NRHS right-hand sides in B and solutions in X such that A*X=B.
 *
 * Faithful port of LAPACK TESTING/MATGEN/zlahilb.f
 */

#include "verify.h"

#define NMAX_EXACT   6
#define NMAX_APPROX  11
#define SIZE_D       8

static const c128 D1[8] = {
    CMPLX(-1.0, 0.0),  CMPLX(0.0, 1.0), CMPLX(-1.0, -1.0),
     CMPLX(0.0, -1.0),  CMPLX(1.0, 0.0), CMPLX(-1.0, 1.0),
     CMPLX(1.0, 1.0),  CMPLX(1.0, -1.0)
};

static const c128 D2[8] = {
    CMPLX(-1.0, 0.0),  CMPLX(0.0, -1.0), CMPLX(-1.0, 1.0),
     CMPLX(0.0, 1.0),  CMPLX(1.0, 0.0), CMPLX(-1.0, -1.0),
     CMPLX(1.0, -1.0),  CMPLX(1.0, 1.0)
};

static const c128 INVD1[8] = {
    CMPLX(-1.0, 0.0),  CMPLX(0.0, -1.0), CMPLX(-0.5, 0.5),
     CMPLX(0.0, 1.0),  CMPLX(1.0, 0.0), CMPLX(-0.5, -0.5),
     CMPLX(0.5, -0.5),  CMPLX(0.5, 0.5)
};

static const c128 INVD2[8] = {
    CMPLX(-1.0, 0.0),  CMPLX(0.0, 1.0), CMPLX(-0.5, -0.5),
     CMPLX(0.0, -1.0),  CMPLX(1.0, 0.0), CMPLX(-0.5, 0.5),
     CMPLX(0.5, 0.5),  CMPLX(0.5, -0.5)
};

/**
 * ZLAHILB generates an N by N scaled Hilbert matrix in A along with
 * NRHS right-hand sides in B and solutions in X such that A*X=B.
 *
 * The Hilbert matrix is scaled by M = LCM(1, 2, ..., 2*N-1) so that all
 * entries are integers.  The right-hand sides are the first NRHS
 * columns of M * the identity matrix, and the solutions are the
 * first NRHS columns of the inverse Hilbert matrix.
 *
 * @param[in] n
 *     The dimension of the matrix A.
 *
 * @param[in] nrhs
 *     The requested number of right-hand sides.
 *
 * @param[out] A
 *     Complex array, dimension (lda, n).
 *     The generated scaled Hilbert matrix.
 *
 * @param[in] lda
 *     The leading dimension of the array A. lda >= n.
 *
 * @param[out] X
 *     Complex array, dimension (ldx, nrhs).
 *     The generated exact solutions.
 *
 * @param[in] ldx
 *     The leading dimension of the array X. ldx >= n.
 *
 * @param[out] B
 *     Complex array, dimension (ldb, nrhs).
 *     The generated right-hand sides.
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
 *
 * @param[in] path
 *     The LAPACK path name (3 characters).
 */
void zlahilb(const INT n, const INT nrhs,
             c128* A, const INT lda,
             c128* X, const INT ldx,
             c128* B, const INT ldb,
             f64* work, INT* info,
             const char* path)
{
    INT tm, ti, r;
    INT m;
    INT i, j;
    c128 tmp;
    INT is_sy;

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
        xerbla("ZLAHILB", -(*info));
        return;
    }
    if (n > NMAX_EXACT) {
        *info = 1;
    }

    /* Compute M = the LCM of the integers [1, 2*N-1]. */
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

    /* Check if we are testing SY routines: path(2:3) == 'SY' */
    is_sy = (path[1] == 'S' || path[1] == 's') &&
            (path[2] == 'Y' || path[2] == 'y');

    /* Generate the scaled Hilbert matrix in A
     * If we are testing SY routines, take D1_i = D2_i,
     * else, D1_i = D2_i* */
    if (is_sy) {
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                A[i + j * lda] = D1[(j + 1) % SIZE_D] *
                    ((f64)m / (i + j + 1)) *
                    D1[(i + 1) % SIZE_D];
            }
        }
    } else {
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                A[i + j * lda] = D1[(j + 1) % SIZE_D] *
                    ((f64)m / (i + j + 1)) *
                    D2[(i + 1) % SIZE_D];
            }
        }
    }

    /* Generate matrix B as simply the first NRHS columns of M * the
     * identity. */
    tmp = (f64)m;
    zlaset("Full", n, nrhs, CMPLX(0.0, 0.0), tmp, B, ldb);

    /* Generate the true solutions in X. Because B = the first NRHS
     * columns of M*I, the true solutions are just the first NRHS columns
     * of the inverse Hilbert matrix. */
    work[0] = (f64)n;
    for (j = 1; j < n; j++) {
        work[j] = ((work[j - 1] / (f64)j) * (f64)(j - n) / (f64)j)
                  * (f64)(n + j);
    }

    if (is_sy) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                X[i + j * ldx] = INVD1[(j + 1) % SIZE_D] *
                    ((work[i] * work[j]) / (i + j + 1)) *
                    INVD1[(i + 1) % SIZE_D];
            }
        }
    } else {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                X[i + j * ldx] = INVD2[(j + 1) % SIZE_D] *
                    ((work[i] * work[j]) / (i + j + 1)) *
                    INVD1[(i + 1) % SIZE_D];
            }
        }
    }
}
