/**
 * @file zlaptm.c
 * @brief ZLAPTM multiplies an N by NRHS matrix X by a Hermitian tridiagonal
 *        matrix A and stores the result in a matrix B.
 *
 * Port of LAPACK's TESTING/LIN/zlaptm.f to C.
 */

#include "verify.h"

/**
 * ZLAPTM multiplies an N by NRHS matrix X by a Hermitian tridiagonal
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
void zlaptm(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64 alpha,
    const f64* const restrict D,
    const c128* const restrict E,
    const c128* const restrict X,
    const INT ldx,
    const f64 beta,
    c128* const restrict B,
    const INT ldb)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    if (n == 0) {
        return;
    }

    if (beta == ZERO) {
        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                B[i + j * ldb] = CMPLX(0.0, 0.0);
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
                                          conj(E[n - 2]) * X[n - 2 + j * ldx] +
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] +
                                         conj(E[i - 1]) * X[i - 1 + j * ldx] +
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
                                 conj(E[0]) * X[1 + j * ldx];
                    B[n - 1 + j * ldb] = B[n - 1 + j * ldb] +
                                          E[n - 2] * X[n - 2 + j * ldx] +
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] +
                                         E[i - 1] * X[i - 1 + j * ldx] +
                                         D[i] * X[i + j * ldx] +
                                         conj(E[i]) * X[i + 1 + j * ldx];
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
                                          conj(E[n - 2]) * X[n - 2 + j * ldx] -
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] -
                                         conj(E[i - 1]) * X[i - 1 + j * ldx] -
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
                                 conj(E[0]) * X[1 + j * ldx];
                    B[n - 1 + j * ldb] = B[n - 1 + j * ldb] -
                                          E[n - 2] * X[n - 2 + j * ldx] -
                                          D[n - 1] * X[n - 1 + j * ldx];
                    for (INT i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb] -
                                         E[i - 1] * X[i - 1 + j * ldx] -
                                         D[i] * X[i + j * ldx] -
                                         conj(E[i]) * X[i + 1 + j * ldx];
                    }
                }
            }
        }
    }
}
