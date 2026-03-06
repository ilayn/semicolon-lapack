/**
 * @file claptm.c
 * @brief CLAPTM multiplies an N by NRHS matrix X by a Hermitian tridiagonal
 *        matrix A and stores the result in a matrix B.
 *
 * Port of LAPACK's TESTING/LIN/claptm.f to C.
 */

#include "verify.h"

/**
 * CLAPTM multiplies an N by NRHS matrix X by a Hermitian tridiagonal
 * matrix A and stores the result in a matrix B.  The operation has the
 * form
 *
 *    B := alpha * A * X + beta * B
 *
 * where alpha may be either 1. or -1. and beta may be 0., 1., or -1.
 *
 * @param[in]     uplo   Specifies whether the superdiagonal or the subdiagonal
 *                       of the tridiagonal matrix A is stored.
 *                       = 'U':  Upper, E is the superdiagonal of A.
 *                       = 'L':  Lower, E is the subdiagonal of A.
 * @param[in]     n      The order of the matrix A. N >= 0.
 * @param[in]     nrhs   The number of right hand sides.
 * @param[in]     alpha  The scalar alpha. ALPHA must be 1. or -1.
 * @param[in]     D      The n diagonal elements of A (real).
 * @param[in]     E      The (n-1) subdiagonal or superdiagonal elements of A.
 * @param[in]     X      The N by NRHS matrix X (ldx x nrhs).
 * @param[in]     ldx    Leading dimension of X. LDX >= max(N,1).
 * @param[in]     beta   The scalar beta. BETA must be 0., 1., or -1.
 * @param[in,out] B      On entry, the N by NRHS matrix B.
 *                       On exit, B := alpha * A * X + beta * B.
 * @param[in]     ldb    Leading dimension of B. LDB >= max(N,1).
 */
void claptm(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f32 alpha,
    const f32* const restrict D,
    const c64* const restrict E,
    const c64* const restrict X,
    const INT ldx,
    const f32 beta,
    c64* const restrict B,
    const INT ldb)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    if (n == 0) {
        return;
    }

    if (beta == ZERO) {
        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                B[i + j * ldb] = CMPLXF(0.0f, 0.0f);
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
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Compute B := B + A*X, where E is the superdiagonal of A. */
            for (INT j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx];
                } else {
                    B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx] +
                                 E[0] * X[1 + j * ldx];
                    B[n - 1 + j * ldb] = B[n - 1 + j * ldb] +
                                          conjf(E[n - 2]) * X[n - 2 + j * ldx] +
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] +
                                         conjf(E[i - 1]) * X[i - 1 + j * ldx] +
                                         D[i] * X[i + j * ldx] +
                                         E[i] * X[i + 1 + j * ldx];
                    }
                }
            }
        } else {
            /* Compute B := B + A*X, where E is the subdiagonal of A. */
            for (INT j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx];
                } else {
                    B[j * ldb] = B[j * ldb] + D[0] * X[j * ldx] +
                                 conjf(E[0]) * X[1 + j * ldx];
                    B[n - 1 + j * ldb] = B[n - 1 + j * ldb] +
                                          E[n - 2] * X[n - 2 + j * ldx] +
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] +
                                         E[i - 1] * X[i - 1 + j * ldx] +
                                         D[i] * X[i + j * ldx] +
                                         conjf(E[i]) * X[i + 1 + j * ldx];
                    }
                }
            }
        }
    } else if (alpha == -ONE) {
        if (uplo[0] == 'U' || uplo[0] == 'u') {
            /* Compute B := B - A*X, where E is the superdiagonal of A. */
            for (INT j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx];
                } else {
                    B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx] -
                                 E[0] * X[1 + j * ldx];
                    B[n - 1 + j * ldb] = B[n - 1 + j * ldb] -
                                          conjf(E[n - 2]) * X[n - 2 + j * ldx] -
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] -
                                         conjf(E[i - 1]) * X[i - 1 + j * ldx] -
                                         D[i] * X[i + j * ldx] -
                                         E[i] * X[i + 1 + j * ldx];
                    }
                }
            }
        } else {
            /* Compute B := B - A*X, where E is the subdiagonal of A. */
            for (INT j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx];
                } else {
                    B[j * ldb] = B[j * ldb] - D[0] * X[j * ldx] -
                                 conjf(E[0]) * X[1 + j * ldx];
                    B[n - 1 + j * ldb] = B[n - 1 + j * ldb] -
                                          E[n - 2] * X[n - 2 + j * ldx] -
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] -
                                         E[i - 1] * X[i - 1 + j * ldx] -
                                         D[i] * X[i + j * ldx] -
                                         conjf(E[i]) * X[i + 1 + j * ldx];
                    }
                }
            }
        }
    }
}
