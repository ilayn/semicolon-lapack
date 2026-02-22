/**
 * @file cgtts2.c
 * @brief CGTTS2 solves a system of linear equations with a tridiagonal matrix
 *        using the LU factorization computed by cgttrf.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CGTTS2 solves one of the systems of equations
 *    A * X = B,  A**T * X = B,  or  A**H * X = B,
 * with a tridiagonal matrix A using the LU factorization computed
 * by CGTTRF.
 *
 * @param[in] itrans  Specifies the form of the system of equations.
 *                    = 0: A * X = B  (No transpose)
 *                    = 1: A**T * X = B  (Transpose)
 *                    = 2: A**H * X = B  (Conjugate transpose)
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in] nrhs    The number of right hand sides, i.e., the number of columns
 *                    of the matrix B. nrhs >= 0.
 * @param[in] DL      The (n-1) multipliers that define the matrix L from the
 *                    LU factorization of A. Array of dimension (n-1).
 * @param[in] D       The n diagonal elements of the upper triangular matrix U from
 *                    the LU factorization of A. Array of dimension (n).
 * @param[in] DU      The (n-1) elements of the first super-diagonal of U.
 *                    Array of dimension (n-1).
 * @param[in] DU2     The (n-2) elements of the second super-diagonal of U.
 *                    Array of dimension (n-2).
 * @param[in] ipiv    The pivot indices; for 0 <= i < n, row i of the matrix was
 *                    interchanged with row ipiv[i]. ipiv[i] will always be either
 *                    i or i+1; ipiv[i] = i indicates a row interchange was not
 *                    required. Array of dimension (n).
 * @param[in,out] B   On entry, the matrix of right hand side vectors B.
 *                    On exit, B is overwritten by the solution vectors X.
 *                    Array of dimension (ldb, nrhs).
 * @param[in] ldb     The leading dimension of the array B. ldb >= max(1, n).
 */
void cgtts2(
    const INT itrans,
    const INT n,
    const INT nrhs,
    const c64* restrict DL,
    const c64* restrict D,
    const c64* restrict DU,
    const c64* restrict DU2,
    const INT* restrict ipiv,
    c64* restrict B,
    const INT ldb)
{
    INT i, j;
    c64 temp;

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return;
    }

    if (itrans == 0) {
        /*
         * Solve A*X = B using the LU factorization of A,
         * overwriting each right hand side vector with its solution.
         */
        if (nrhs <= 1) {
            for (j = 0; j < nrhs; j++) {
                /* Solve L*x = b. */
                for (i = 0; i < n - 1; i++) {
                    if (ipiv[i] == i) {
                        B[(i + 1) + j * ldb] = B[(i + 1) + j * ldb] - DL[i] * B[i + j * ldb];
                    } else {
                        temp = B[i + j * ldb];
                        B[i + j * ldb] = B[(i + 1) + j * ldb];
                        B[(i + 1) + j * ldb] = temp - DL[i] * B[i + j * ldb];
                    }
                }

                /* Solve U*x = b. */
                B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
                if (n > 1) {
                    B[(n - 2) + j * ldb] = (B[(n - 2) + j * ldb] - DU[n - 2] * B[(n - 1) + j * ldb]) / D[n - 2];
                }
                for (i = n - 3; i >= 0; i--) {
                    B[i + j * ldb] = (B[i + j * ldb] - DU[i] * B[(i + 1) + j * ldb]
                                      - DU2[i] * B[(i + 2) + j * ldb]) / D[i];
                }
            }
        } else {
            for (j = 0; j < nrhs; j++) {
                /* Solve L*x = b. */
                for (i = 0; i < n - 1; i++) {
                    if (ipiv[i] == i) {
                        B[(i + 1) + j * ldb] = B[(i + 1) + j * ldb] - DL[i] * B[i + j * ldb];
                    } else {
                        temp = B[i + j * ldb];
                        B[i + j * ldb] = B[(i + 1) + j * ldb];
                        B[(i + 1) + j * ldb] = temp - DL[i] * B[i + j * ldb];
                    }
                }

                /* Solve U*x = b. */
                B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
                if (n > 1) {
                    B[(n - 2) + j * ldb] = (B[(n - 2) + j * ldb] - DU[n - 2] * B[(n - 1) + j * ldb]) / D[n - 2];
                }
                for (i = n - 3; i >= 0; i--) {
                    B[i + j * ldb] = (B[i + j * ldb] - DU[i] * B[(i + 1) + j * ldb]
                                      - DU2[i] * B[(i + 2) + j * ldb]) / D[i];
                }
            }
        }
    } else if (itrans == 1) {
        /*
         * Solve A**T * X = B.
         */
        if (nrhs <= 1) {
            for (j = 0; j < nrhs; j++) {
                /* Solve U**T * x = b. */
                B[0 + j * ldb] = B[0 + j * ldb] / D[0];
                if (n > 1) {
                    B[1 + j * ldb] = (B[1 + j * ldb] - DU[0] * B[0 + j * ldb]) / D[1];
                }
                for (i = 2; i < n; i++) {
                    B[i + j * ldb] = (B[i + j * ldb] - DU[i - 1] * B[(i - 1) + j * ldb]
                                      - DU2[i - 2] * B[(i - 2) + j * ldb]) / D[i];
                }

                /* Solve L**T * x = b. */
                for (i = n - 2; i >= 0; i--) {
                    if (ipiv[i] == i) {
                        B[i + j * ldb] = B[i + j * ldb] - DL[i] * B[(i + 1) + j * ldb];
                    } else {
                        temp = B[(i + 1) + j * ldb];
                        B[(i + 1) + j * ldb] = B[i + j * ldb] - DL[i] * temp;
                        B[i + j * ldb] = temp;
                    }
                }
            }
        } else {
            for (j = 0; j < nrhs; j++) {
                /* Solve U**T * x = b. */
                B[0 + j * ldb] = B[0 + j * ldb] / D[0];
                if (n > 1) {
                    B[1 + j * ldb] = (B[1 + j * ldb] - DU[0] * B[0 + j * ldb]) / D[1];
                }
                for (i = 2; i < n; i++) {
                    B[i + j * ldb] = (B[i + j * ldb] - DU[i - 1] * B[(i - 1) + j * ldb]
                                      - DU2[i - 2] * B[(i - 2) + j * ldb]) / D[i];
                }

                /* Solve L**T * x = b. */
                for (i = n - 2; i >= 0; i--) {
                    if (ipiv[i] == i) {
                        B[i + j * ldb] = B[i + j * ldb] - DL[i] * B[(i + 1) + j * ldb];
                    } else {
                        temp = B[(i + 1) + j * ldb];
                        B[(i + 1) + j * ldb] = B[i + j * ldb] - DL[i] * temp;
                        B[i + j * ldb] = temp;
                    }
                }
            }
        }
    } else {
        /*
         * Solve A**H * X = B.
         */
        if (nrhs <= 1) {
            for (j = 0; j < nrhs; j++) {
                /* Solve U**H * x = b. */
                B[0 + j * ldb] = B[0 + j * ldb] / conjf(D[0]);
                if (n > 1) {
                    B[1 + j * ldb] = (B[1 + j * ldb] - conjf(DU[0]) * B[0 + j * ldb]) / conjf(D[1]);
                }
                for (i = 2; i < n; i++) {
                    B[i + j * ldb] = (B[i + j * ldb] - conjf(DU[i - 1]) * B[(i - 1) + j * ldb]
                                      - conjf(DU2[i - 2]) * B[(i - 2) + j * ldb]) / conjf(D[i]);
                }

                /* Solve L**H * x = b. */
                for (i = n - 2; i >= 0; i--) {
                    if (ipiv[i] == i) {
                        B[i + j * ldb] = B[i + j * ldb] - conjf(DL[i]) * B[(i + 1) + j * ldb];
                    } else {
                        temp = B[(i + 1) + j * ldb];
                        B[(i + 1) + j * ldb] = B[i + j * ldb] - conjf(DL[i]) * temp;
                        B[i + j * ldb] = temp;
                    }
                }
            }
        } else {
            for (j = 0; j < nrhs; j++) {
                /* Solve U**H * x = b. */
                B[0 + j * ldb] = B[0 + j * ldb] / conjf(D[0]);
                if (n > 1) {
                    B[1 + j * ldb] = (B[1 + j * ldb] - conjf(DU[0]) * B[0 + j * ldb]) / conjf(D[1]);
                }
                for (i = 2; i < n; i++) {
                    B[i + j * ldb] = (B[i + j * ldb] - conjf(DU[i - 1]) * B[(i - 1) + j * ldb]
                                      - conjf(DU2[i - 2]) * B[(i - 2) + j * ldb]) / conjf(D[i]);
                }

                /* Solve L**H * x = b. */
                for (i = n - 2; i >= 0; i--) {
                    if (ipiv[i] == i) {
                        B[i + j * ldb] = B[i + j * ldb] - conjf(DL[i]) * B[(i + 1) + j * ldb];
                    } else {
                        temp = B[(i + 1) + j * ldb];
                        B[(i + 1) + j * ldb] = B[i + j * ldb] - conjf(DL[i]) * temp;
                        B[i + j * ldb] = temp;
                    }
                }
            }
        }
    }
}
