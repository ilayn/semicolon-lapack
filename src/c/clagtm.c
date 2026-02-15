/**
 * @file clagtm.c
 * @brief CLAGTM performs a matrix-matrix product of the form C = alpha*A*B + beta*C,
 *        where A is a tridiagonal matrix.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLAGTM performs a matrix-matrix product of the form
 *
 *    B := alpha * A * X + beta * B
 *
 * where A is a tridiagonal matrix of order N, B and X are N by NRHS
 * matrices, and alpha and beta are real scalars, each of which may be
 * 0., 1., or -1.
 *
 * @param[in] trans   Specifies the operation applied to A.
 *                    = 'N': No transpose, B := alpha * A * X + beta * B
 *                    = 'T': Transpose,    B := alpha * A**T * X + beta * B
 *                    = 'C': Conjugate transpose, B := alpha * A**H * X + beta * B
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in] nrhs    The number of right hand sides, i.e., the number of columns
 *                    of the matrices X and B.
 * @param[in] alpha   The scalar alpha. ALPHA must be 0., 1., or -1.; otherwise,
 *                    it is assumed to be 0.
 * @param[in] DL      The (n-1) sub-diagonal elements of T. Array of dimension (n-1).
 * @param[in] D       The diagonal elements of T. Array of dimension (n).
 * @param[in] DU      The (n-1) super-diagonal elements of T. Array of dimension (n-1).
 * @param[in] X       The N by NRHS matrix X. Array of dimension (ldx, nrhs).
 * @param[in] ldx     The leading dimension of the array X. ldx >= max(n, 1).
 * @param[in] beta    The scalar beta. BETA must be 0., 1., or -1.; otherwise,
 *                    it is assumed to be 1.
 * @param[in,out] B   On entry, the N by NRHS matrix B.
 *                    On exit, B is overwritten by the matrix expression
 *                    B := alpha * A * X + beta * B.
 *                    Array of dimension (ldb, nrhs).
 * @param[in] ldb     The leading dimension of the array B. ldb >= max(n, 1).
 */
void clagtm(
    const char* trans,
    const int n,
    const int nrhs,
    const f32 alpha,
    const c64* restrict DL,
    const c64* restrict D,
    const c64* restrict DU,
    const c64* restrict X,
    const int ldx,
    const f32 beta,
    c64* restrict B,
    const int ldb)
{
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    int i, j;

    if (n == 0) {
        return;
    }

    /* Multiply B by BETA if BETA != 1 */
    if (beta == ZERO) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = CMPLXF(0.0f, 0.0f);
            }
        }
    } else if (beta == -ONE) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = -B[i + j * ldb];
            }
        }
    }

    if (alpha == ONE) {
        if (trans[0] == 'N' || trans[0] == 'n') {
            /* Compute B := B + A*X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx];
                } else {
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx]
                                     + DU[0] * X[1 + j * ldx];
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           + DL[n - 2] * X[(n - 2) + j * ldx]
                                           + D[n - 1] * X[(n - 1) + j * ldx];
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         + DL[i - 1] * X[(i - 1) + j * ldx]
                                         + D[i] * X[i + j * ldx]
                                         + DU[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        } else if (trans[0] == 'T' || trans[0] == 't') {
            /* Compute B := B + A**T * X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx];
                } else {
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx]
                                     + DL[0] * X[1 + j * ldx];
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           + DU[n - 2] * X[(n - 2) + j * ldx]
                                           + D[n - 1] * X[(n - 1) + j * ldx];
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         + DU[i - 1] * X[(i - 1) + j * ldx]
                                         + D[i] * X[i + j * ldx]
                                         + DL[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        } else if (trans[0] == 'C' || trans[0] == 'c') {
            /* Compute B := B + A**H * X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] + conjf(D[0]) * X[0 + j * ldx];
                } else {
                    B[0 + j * ldb] = B[0 + j * ldb] + conjf(D[0]) * X[0 + j * ldx]
                                     + conjf(DL[0]) * X[1 + j * ldx];
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           + conjf(DU[n - 2]) * X[(n - 2) + j * ldx]
                                           + conjf(D[n - 1]) * X[(n - 1) + j * ldx];
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         + conjf(DU[i - 1]) * X[(i - 1) + j * ldx]
                                         + conjf(D[i]) * X[i + j * ldx]
                                         + conjf(DL[i]) * X[(i + 1) + j * ldx];
                    }
                }
            }
        }
    } else if (alpha == -ONE) {
        if (trans[0] == 'N' || trans[0] == 'n') {
            /* Compute B := B - A*X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx];
                } else {
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx]
                                     - DU[0] * X[1 + j * ldx];
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           - DL[n - 2] * X[(n - 2) + j * ldx]
                                           - D[n - 1] * X[(n - 1) + j * ldx];
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         - DL[i - 1] * X[(i - 1) + j * ldx]
                                         - D[i] * X[i + j * ldx]
                                         - DU[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        } else if (trans[0] == 'T' || trans[0] == 't') {
            /* Compute B := B - A**T *X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx];
                } else {
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx]
                                     - DL[0] * X[1 + j * ldx];
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           - DU[n - 2] * X[(n - 2) + j * ldx]
                                           - D[n - 1] * X[(n - 1) + j * ldx];
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         - DU[i - 1] * X[(i - 1) + j * ldx]
                                         - D[i] * X[i + j * ldx]
                                         - DL[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        } else if (trans[0] == 'C' || trans[0] == 'c') {
            /* Compute B := B - A**H *X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] - conjf(D[0]) * X[0 + j * ldx];
                } else {
                    B[0 + j * ldb] = B[0 + j * ldb] - conjf(D[0]) * X[0 + j * ldx]
                                     - conjf(DL[0]) * X[1 + j * ldx];
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           - conjf(DU[n - 2]) * X[(n - 2) + j * ldx]
                                           - conjf(D[n - 1]) * X[(n - 1) + j * ldx];
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         - conjf(DU[i - 1]) * X[(i - 1) + j * ldx]
                                         - conjf(D[i]) * X[i + j * ldx]
                                         - conjf(DL[i]) * X[(i + 1) + j * ldx];
                    }
                }
            }
        }
    }
}
