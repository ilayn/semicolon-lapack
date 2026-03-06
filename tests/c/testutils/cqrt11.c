/**
 * @file cqrt11.c
 * @brief CQRT11 computes the test ratio || Q'*Q - I || / (eps * m).
 *
 * Port of LAPACK TESTING/LIN/cqrt11.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include "verify.h"

/**
 * CQRT11 computes the test ratio
 *
 *    || Q'*Q - I || / (eps * m)
 *
 * where the orthogonal matrix Q is represented as a product of
 * elementary transformations. Each transformation has the form
 *
 *    H(k) = I - tau(k) v(k) v(k)'
 *
 * where tau(k) is stored in TAU(k) and v(k) is an m-vector of the form
 * [ 0 ... 0 1 x(k) ]', where x(k) is a vector of length m-k stored
 * in A(k+1:m,k).
 *
 * @param[in]  m     The number of rows of the matrix A.
 * @param[in]  k     The number of columns of A whose subdiagonal entries
 *                   contain information about orthogonal transformations.
 * @param[in]  A     Array (lda, k). The (possibly partial) output of a
 *                   QR reduction routine.
 * @param[in]  lda   The leading dimension of the array A.
 * @param[in]  tau   Array (k). The scaling factors tau for the elementary
 *                   transformations as computed by the QR factorization routine.
 * @param[out] work  Array (lwork). Workspace.
 * @param[in]  lwork The length of the array work. lwork >= m*m + m.
 *
 * @return The test ratio || Q'*Q - I || / (eps * m).
 */
f32 cqrt11(const INT m, const INT k, const c64* A, const INT lda,
              const c64* tau, c64* work, const INT lwork)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CZERO = CMPLXF(0.0f, 0.0f);
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    INT info, j;
    f32 rdummy[1];

    /* Quick return if possible */
    if (lwork < m * m + m) {
        xerbla("CQRT11", 7);
        return ZERO;
    }

    if (m <= 0) {
        return ZERO;
    }

    /* Set work to identity matrix */
    claset("F", m, m, CZERO, CONE, work, m);

    /* Form Q by applying transformations from the left with 'N' */
    cunm2r("L", "N", m, m, k, A, lda, tau, work, m, &work[m * m], &info);

    /* Form Q'*Q by applying transformations from the left with 'C' */
    cunm2r("L", "C", m, m, k, A, lda, tau, work, m, &work[m * m], &info);

    /* Subtract identity: Q'*Q - I */
    for (j = 0; j < m; j++) {
        work[j * m + j] -= ONE;
    }

    /* Return || Q'*Q - I || / (eps * m) */
    return clange("1", m, m, work, m, rdummy) / ((f32)m * slamch("E"));
}
