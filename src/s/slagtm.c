/**
 * @file slagtm.c
 * @brief SLAGTM performs a matrix-matrix product of the form C = alpha*A*B + beta*C,
 *        where A is a tridiagonal matrix.
 */

#include "semicolon_lapack_single.h"

/**
 * SLAGTM performs a matrix-matrix product of the form
 *
 *    B := alpha * A * X + beta * B
 *
 * where A is a tridiagonal matrix of order N, B and X are N by NRHS
 * matrices, and alpha and beta are real scalars, each of which may be
 * 0., 1., or -1.
 *
 * @param[in] trans   Specifies the operation applied to A.
 *                    = 'N': No transpose, B := alpha * A * X + beta * B
 *                    = 'T': Transpose,    B := alpha * A' * X + beta * B
 *                    = 'C': Conjugate transpose = Transpose
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in] nrhs    The number of right hand sides, i.e., the number of columns
 *                    of the matrices X and B.
 * @param[in] alpha   The scalar alpha. ALPHA must be 0., 1., or -1.; otherwise,
 *                    it is assumed to be 0.
 * @param[in] DL      The (n-1) sub-diagonal elements of A. Array of dimension (n-1).
 * @param[in] D       The diagonal elements of A. Array of dimension (n).
 * @param[in] DU      The (n-1) super-diagonal elements of A. Array of dimension (n-1).
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
void slagtm(
    const char* trans,
    const int n,
    const int nrhs,
    const float alpha,
    const float * const restrict DL,
    const float * const restrict D,
    const float * const restrict DU,
    const float * const restrict X,
    const int ldx,
    const float beta,
    float * const restrict B,
    const int ldb)
{
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float NEG_ONE = -1.0f;

    int i, j;

    if (n == 0) {
        return;
    }

    /* Multiply B by BETA if BETA != 1 */
    if (beta == ZERO) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = ZERO;
            }
        }
    } else if (beta == NEG_ONE) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = -B[i + j * ldb];
            }
        }
    }
    /* If beta == 1, B remains unchanged */

    if (alpha == ONE) {
        if (trans[0] == 'N' || trans[0] == 'n') {
            /* Compute B := B + A*X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx];
                } else {
                    /* First row */
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx]
                                     + DU[0] * X[1 + j * ldx];
                    /* Last row */
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           + DL[n - 2] * X[(n - 2) + j * ldx]
                                           + D[n - 1] * X[(n - 1) + j * ldx];
                    /* Middle rows */
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         + DL[i - 1] * X[(i - 1) + j * ldx]
                                         + D[i] * X[i + j * ldx]
                                         + DU[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        } else {
            /* Compute B := B + A**T * X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx];
                } else {
                    /* First row: A^T has D[0] and DL[0] in first row */
                    B[0 + j * ldb] = B[0 + j * ldb] + D[0] * X[0 + j * ldx]
                                     + DL[0] * X[1 + j * ldx];
                    /* Last row: A^T has DU[n-2] and D[n-1] in last row */
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           + DU[n - 2] * X[(n - 2) + j * ldx]
                                           + D[n - 1] * X[(n - 1) + j * ldx];
                    /* Middle rows */
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         + DU[i - 1] * X[(i - 1) + j * ldx]
                                         + D[i] * X[i + j * ldx]
                                         + DL[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        }
    } else if (alpha == NEG_ONE) {
        if (trans[0] == 'N' || trans[0] == 'n') {
            /* Compute B := B - A*X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx];
                } else {
                    /* First row */
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx]
                                     - DU[0] * X[1 + j * ldx];
                    /* Last row */
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           - DL[n - 2] * X[(n - 2) + j * ldx]
                                           - D[n - 1] * X[(n - 1) + j * ldx];
                    /* Middle rows */
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         - DL[i - 1] * X[(i - 1) + j * ldx]
                                         - D[i] * X[i + j * ldx]
                                         - DU[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        } else {
            /* Compute B := B - A**T * X */
            for (j = 0; j < nrhs; j++) {
                if (n == 1) {
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx];
                } else {
                    /* First row */
                    B[0 + j * ldb] = B[0 + j * ldb] - D[0] * X[0 + j * ldx]
                                     - DL[0] * X[1 + j * ldx];
                    /* Last row */
                    B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb]
                                           - DU[n - 2] * X[(n - 2) + j * ldx]
                                           - D[n - 1] * X[(n - 1) + j * ldx];
                    /* Middle rows */
                    for (i = 1; i < n - 1; i++) {
                        B[i + j * ldb] = B[i + j * ldb]
                                         - DU[i - 1] * X[(i - 1) + j * ldx]
                                         - D[i] * X[i + j * ldx]
                                         - DL[i] * X[(i + 1) + j * ldx];
                    }
                }
            }
        }
    }
    /* If alpha == 0, only the beta*B part applies, already done */
}
