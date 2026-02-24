/**
 * @file slaptm.c
 * @brief SLAPTM multiplies an N by NRHS matrix X by a symmetric tridiagonal
 *        matrix A and stores the result in a matrix B.
 *
 * Port of LAPACK's TESTING/LIN/slaptm.f to C.
 */

#include "verify.h"

/**
 * SLAPTM multiplies an N by NRHS matrix X by a symmetric tridiagonal
 * matrix A and stores the result in a matrix B.  The operation has the
 * form
 *
 *    B := alpha * A * X + beta * B
 *
 * where alpha may be either 1. or -1. and beta may be 0., 1., or -1.
 *
 * @param[in]     n      The order of the matrix A. N >= 0.
 * @param[in]     nrhs   The number of right hand sides.
 * @param[in]     alpha  The scalar alpha. ALPHA must be 1. or -1.
 * @param[in]     D      The n diagonal elements of A.
 * @param[in]     E      The (n-1) subdiagonal/superdiagonal elements of A.
 * @param[in]     X      The N by NRHS matrix X (ldx x nrhs).
 * @param[in]     ldx    Leading dimension of X. LDX >= max(N,1).
 * @param[in]     beta   The scalar beta. BETA must be 0., 1., or -1.
 * @param[in,out] B      On entry, the N by NRHS matrix B.
 *                       On exit, B := alpha * A * X + beta * B.
 * @param[in]     ldb    Leading dimension of B. LDB >= max(N,1).
 */
void slaptm(
    const INT n,
    const INT nrhs,
    const f32 alpha,
    const f32* const restrict D,
    const f32* const restrict E,
    const f32* const restrict X,
    const INT ldx,
    const f32 beta,
    f32* const restrict B,
    const INT ldb)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    if (n == 0) {
        return;
    }

    /* Multiply B by BETA if BETA != 1. */
    if (beta == ZERO) {
        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                B[i + j * ldb] = ZERO;
            }
        }
    } else if (beta == -ONE) {
        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                B[i + j * ldb] = -B[i + j * ldb];
            }
        }
    }

    if (alpha == ONE) {
        /* Compute B := B + A*X */
        for (INT j = 0; j < nrhs; j++) {
            if (n == 1) {
                B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx];
            } else {
                B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx] +
                            E[0] * X[1 + j * ldx];
                B[n - 1 + j * ldb] = B[n - 1 + j * ldb] +
                                     E[n - 2] * X[n - 2 + j * ldx] +
                                     D[n - 1] * X[n - 1 + j * ldx];
                for (INT i = 1; i < n - 1; i++) {
                    B[i + j * ldb] = B[i + j * ldb] +
                                    E[i - 1] * X[i - 1 + j * ldx] +
                                    D[i] * X[i + j * ldx] +
                                    E[i] * X[i + 1 + j * ldx];
                }
            }
        }
    } else if (alpha == -ONE) {
        /* Compute B := B - A*X */
        for (INT j = 0; j < nrhs; j++) {
            if (n == 1) {
                B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx];
            } else {
                B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx] -
                            E[0] * X[1 + j * ldx];
                B[n - 1 + j * ldb] = B[n - 1 + j * ldb] -
                                     E[n - 2] * X[n - 2 + j * ldx] -
                                     D[n - 1] * X[n - 1 + j * ldx];
                for (INT i = 1; i < n - 1; i++) {
                    B[i + j * ldb] = B[i + j * ldb] -
                                    E[i - 1] * X[i - 1 + j * ldx] -
                                    D[i] * X[i + j * ldx] -
                                    E[i] * X[i + 1 + j * ldx];
                }
            }
        }
    }
}
