/**
 * @file sqrt11.c
 * @brief SQRT11 computes the test ratio || Q'*Q - I || / (eps * m).
 *
 * Port of LAPACK TESTING/LIN/sqrt11.f to C.
 */

#include <math.h>
#include <stdlib.h>
#include "verify.h"

/* External declarations */
/**
 * SQRT11 computes the test ratio
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
f32 sqrt11(const INT m, const INT k, const f32* A, const INT lda,
              const f32* tau, f32* work, const INT lwork)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    INT info, j;
    f32 rdummy[1];

    /* Quick return if possible */
    if (m <= 0) {
        return ZERO;
    }

    /* Test for sufficient workspace */
    if (lwork < m * m + m) {
        return ZERO;
    }

    /* Set work to identity matrix */
    slaset("F", m, m, ZERO, ONE, work, m);

    /* Form Q by applying transformations from the left with 'N' */
    sorm2r("L", "N", m, m, k, A, lda, tau, work, m, &work[m * m], &info);

    /* Form Q'*Q by applying transformations from the left with 'T' */
    sorm2r("L", "T", m, m, k, A, lda, tau, work, m, &work[m * m], &info);

    /* Subtract identity: Q'*Q - I */
    for (j = 0; j < m; j++) {
        work[j * m + j] -= ONE;
    }

    /* Return || Q'*Q - I || / (eps * m) */
    return slange("1", m, m, work, m, rdummy) / ((f32)m * slamch("E"));
}
