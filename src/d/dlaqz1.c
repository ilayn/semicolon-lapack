/**
 * @file dlaqz1.c
 * @brief DLAQZ1 computes the first column of the product for QZ shifts.
 */

#include "internal_build_defs.h"
#include <math.h>
#include "semicolon_lapack_double.h"

/**
 * Given a 3-by-3 matrix pencil (A,B), DLAQZ1 sets v to a scalar multiple
 * of the first column of the product
 *
 *     K = (A - (beta2*sr2 - i*si)*B)*B^(-1)*(beta1*A - (sr1 + i*si)*B)*B^(-1).
 *
 * It is assumed that either sr1 = sr2 or si = 0.
 *
 * This is useful for starting f64 implicit shift bulges in the QZ algorithm.
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
void dlaqz1(
    const f64* restrict A,
    const INT lda,
    const f64* restrict B,
    const INT ldb,
    const f64 sr1,
    const f64 sr2,
    const f64 si,
    const f64 beta1,
    const f64 beta2,
    f64* restrict v)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    f64 w[2];
    f64 safmin, safmax, scale1, scale2;

    safmin = dlamch("S");
    safmax = ONE / safmin;

    /* Calculate first shifted vector */
    w[0] = beta1 * A[0 + 0 * lda] - sr1 * B[0 + 0 * ldb];
    w[1] = beta1 * A[1 + 0 * lda] - sr1 * B[1 + 0 * ldb];
    scale1 = sqrt(fabs(w[0])) * sqrt(fabs(w[1]));
    if (scale1 >= safmin && scale1 <= safmax) {
        w[0] = w[0] / scale1;
        w[1] = w[1] / scale1;
    }

    /* Solve linear system */
    w[1] = w[1] / B[1 + 1 * ldb];
    w[0] = (w[0] - B[0 + 1 * ldb] * w[1]) / B[0 + 0 * ldb];
    scale2 = sqrt(fabs(w[0])) * sqrt(fabs(w[1]));
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
    if (fabs(v[0]) > safmax || fabs(v[1]) > safmax ||
        fabs(v[2]) > safmax || disnan(v[0]) ||
        disnan(v[1]) || disnan(v[2])) {
        v[0] = ZERO;
        v[1] = ZERO;
        v[2] = ZERO;
    }
}
