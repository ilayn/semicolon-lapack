/**
 * @file dlaptm.c
 * @brief DLAPTM multiplies an N by NRHS matrix X by a symmetric tridiagonal
 *        matrix A and stores the result in a matrix B.
 *
 * Port of LAPACK's TESTING/LIN/dlaptm.f to C.
 */

#include "verify.h"

/**
 * DLAPTM multiplies an N by NRHS matrix X by a symmetric tridiagonal
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
void dlaptm(
    const int n,
    const int nrhs,
    const double alpha,
    const double* const restrict D,
    const double* const restrict E,
    const double* const restrict X,
    const int ldx,
    const double beta,
    double* const restrict B,
    const int ldb)
{
    const double ONE = 1.0;
    const double ZERO = 0.0;

    if (n == 0) {
        return;
    }

    /* Multiply B by BETA if BETA != 1. */
    if (beta == ZERO) {
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                B[i + j * ldb] = ZERO;
            }
        }
    } else if (beta == -ONE) {
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                B[i + j * ldb] = -B[i + j * ldb];
            }
        }
    }

    if (alpha == ONE) {
        /* Compute B := B + A*X */
        for (int j = 0; j < nrhs; j++) {
            if (n == 1) {
                B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx];
            } else {
                B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx] +
                            E[0] * X[1 + j * ldx];
                B[n - 1 + j * ldb] = B[n - 1 + j * ldb] +
                                     E[n - 2] * X[n - 2 + j * ldx] +
                                     D[n - 1] * X[n - 1 + j * ldx];
                for (int i = 1; i < n - 1; i++) {
                    B[i + j * ldb] = B[i + j * ldb] +
                                    E[i - 1] * X[i - 1 + j * ldx] +
                                    D[i] * X[i + j * ldx] +
                                    E[i] * X[i + 1 + j * ldx];
                }
            }
        }
    } else if (alpha == -ONE) {
        /* Compute B := B - A*X */
        for (int j = 0; j < nrhs; j++) {
            if (n == 1) {
                B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx];
            } else {
                B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx] -
                            E[0] * X[1 + j * ldx];
                B[n - 1 + j * ldb] = B[n - 1 + j * ldb] -
                                     E[n - 2] * X[n - 2 + j * ldx] -
                                     D[n - 1] * X[n - 1 + j * ldx];
                for (int i = 1; i < n - 1; i++) {
                    B[i + j * ldb] = B[i + j * ldb] -
                                    E[i - 1] * X[i - 1 + j * ldx] -
                                    D[i] * X[i + j * ldx] -
                                    E[i] * X[i + 1 + j * ldx];
                }
            }
        }
    }
}
