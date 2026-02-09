/**
 * @file slaqz1.c
 * @brief SLAQZ1 computes the first column of the product for QZ shifts.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * Given a 3-by-3 matrix pencil (A,B), SLAQZ1 sets v to a scalar multiple
 * of the first column of the product
 *
 *     K = (A - (beta2*sr2 - i*si)*B)*B^(-1)*(beta1*A - (sr1 + i*si)*B)*B^(-1).
 *
 * It is assumed that either sr1 = sr2 or si = 0.
 *
 * This is useful for starting double implicit shift bulges in the QZ algorithm.
 *
 * @param[in]     A       3x3 matrix A.
 * @param[in]     lda     Leading dimension of A.
 * @param[in]     B       3x3 matrix B.
 * @param[in]     ldb     Leading dimension of B.
 * @param[in]     sr1     First shift real part.
 * @param[in]     sr2     Second shift real part.
 * @param[in]     si      Shift imaginary part.
 * @param[in]     beta1   First shift beta.
 * @param[in]     beta2   Second shift beta.
 * @param[out]    v       Output vector of length 3.
 */
void slaqz1(
    const float* const restrict A,
    const int lda,
    const float* const restrict B,
    const int ldb,
    const float sr1,
    const float sr2,
    const float si,
    const float beta1,
    const float beta2,
    float* const restrict v)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;

    float w[2];
    float safmin, safmax, scale1, scale2;

    safmin = slamch("S");
    safmax = ONE / safmin;

    /* Calculate first shifted vector */
    w[0] = beta1 * A[0 + 0 * lda] - sr1 * B[0 + 0 * ldb];
    w[1] = beta1 * A[1 + 0 * lda] - sr1 * B[1 + 0 * ldb];
    scale1 = sqrtf(fabsf(w[0])) * sqrtf(fabsf(w[1]));
    if (scale1 >= safmin && scale1 <= safmax) {
        w[0] = w[0] / scale1;
        w[1] = w[1] / scale1;
    }

    /* Solve linear system */
    w[1] = w[1] / B[1 + 1 * ldb];
    w[0] = (w[0] - B[0 + 1 * ldb] * w[1]) / B[0 + 0 * ldb];
    scale2 = sqrtf(fabsf(w[0])) * sqrtf(fabsf(w[1]));
    if (scale2 >= safmin && scale2 <= safmax) {
        w[0] = w[0] / scale2;
        w[1] = w[1] / scale2;
    }

    /* Apply second shift */
    v[0] = beta2 * (A[0 + 0 * lda] * w[0] + A[0 + 1 * lda] * w[1])
         - sr2 * (B[0 + 0 * ldb] * w[0] + B[0 + 1 * ldb] * w[1]);
    v[1] = beta2 * (A[1 + 0 * lda] * w[0] + A[1 + 1 * lda] * w[1])
         - sr2 * (B[1 + 0 * ldb] * w[0] + B[1 + 1 * ldb] * w[1]);
    v[2] = beta2 * (A[2 + 0 * lda] * w[0] + A[2 + 1 * lda] * w[1])
         - sr2 * (B[2 + 0 * ldb] * w[0] + B[2 + 1 * ldb] * w[1]);

    /* Account for imaginary part */
    v[0] = v[0] + si * si * B[0 + 0 * ldb] / scale1 / scale2;

    /* Check for overflow */
    if (fabsf(v[0]) > safmax || fabsf(v[1]) > safmax ||
        fabsf(v[2]) > safmax || sisnan(v[0]) ||
        sisnan(v[1]) || sisnan(v[2])) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
    }
}
