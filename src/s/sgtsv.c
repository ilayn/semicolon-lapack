/**
 * @file sgtsv.c
 * @brief SGTSV computes the solution to system of linear equations A * X = B
 *        for GT (tridiagonal) matrices.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SGTSV solves the equation
 *
 *    A*X = B,
 *
 * where A is an n by n tridiagonal matrix, by Gaussian elimination with
 * partial pivoting.
 *
 * Note that the equation A**T*X = B may be solved by interchanging the
 * order of the arguments DU and DL.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of columns
 *                      of the matrix B. nrhs >= 0.
 * @param[in,out] DL    On entry, the (n-1) sub-diagonal elements of A.
 *                      On exit, DL is overwritten by the (n-2) elements of the
 *                      second super-diagonal of the upper triangular matrix U from
 *                      the LU factorization of A, in DL[0], ..., DL[n-3].
 *                      Array of dimension (n-1).
 * @param[in,out] D     On entry, the diagonal elements of A.
 *                      On exit, D is overwritten by the n diagonal elements of U.
 *                      Array of dimension (n).
 * @param[in,out] DU    On entry, the (n-1) super-diagonal elements of A.
 *                      On exit, DU is overwritten by the (n-1) elements of the first
 *                      super-diagonal of U.
 *                      Array of dimension (n-1).
 * @param[in,out] B     On entry, the N by NRHS matrix of right hand side matrix B.
 *                      On exit, if info = 0, the N by NRHS solution matrix X.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, U(i-1,i-1) is exactly zero (0-based), and the
 *                           solution has not been computed. The factorization has not
 *                           been completed unless i = n.
 */
void sgtsv(
    const INT n,
    const INT nrhs,
    f32* restrict DL,
    f32* restrict D,
    f32* restrict DU,
    f32* restrict B,
    const INT ldb,
    INT* info)
{
    const f32 ZERO = 0.0f;
    INT i, j;
    f32 fact, temp;
    INT ldb_min;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else {
        ldb_min = (n > 1) ? n : 1;
        if (ldb < ldb_min) {
            *info = -7;
        }
    }

    if (*info != 0) {
        xerbla("SGTSV ", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (nrhs == 1) {
        /* Optimized path for single RHS */
        for (i = 0; i < n - 2; i++) {
            if (fabsf(D[i]) >= fabsf(DL[i])) {
                /* No row interchange required */
                if (D[i] != ZERO) {
                    fact = DL[i] / D[i];
                    D[i + 1] = D[i + 1] - fact * DU[i];
                    B[i + 1] = B[i + 1] - fact * B[i];
                } else {
                    *info = i + 1;  /* 1-based */
                    return;
                }
                DL[i] = ZERO;
            } else {
                /* Interchange rows i and i+1 */
                fact = D[i] / DL[i];
                D[i] = DL[i];
                temp = D[i + 1];
                D[i + 1] = DU[i] - fact * temp;
                DL[i] = DU[i + 1];
                DU[i + 1] = -fact * DL[i];
                DU[i] = temp;
                temp = B[i];
                B[i] = B[i + 1];
                B[i + 1] = temp - fact * B[i + 1];
            }
        }

        if (n > 1) {
            i = n - 2;
            if (fabsf(D[i]) >= fabsf(DL[i])) {
                if (D[i] != ZERO) {
                    fact = DL[i] / D[i];
                    D[i + 1] = D[i + 1] - fact * DU[i];
                    B[i + 1] = B[i + 1] - fact * B[i];
                } else {
                    *info = i + 1;  /* 1-based */
                    return;
                }
            } else {
                fact = D[i] / DL[i];
                D[i] = DL[i];
                temp = D[i + 1];
                D[i + 1] = DU[i] - fact * temp;
                DU[i] = temp;
                temp = B[i];
                B[i] = B[i + 1];
                B[i + 1] = temp - fact * B[i + 1];
            }
        }

        if (D[n - 1] == ZERO) {
            *info = n;  /* 1-based */
            return;
        }
    } else {
        /* General path for multiple RHS */
        for (i = 0; i < n - 2; i++) {
            if (fabsf(D[i]) >= fabsf(DL[i])) {
                /* No row interchange required */
                if (D[i] != ZERO) {
                    fact = DL[i] / D[i];
                    D[i + 1] = D[i + 1] - fact * DU[i];
                    for (j = 0; j < nrhs; j++) {
                        B[(i + 1) + j * ldb] = B[(i + 1) + j * ldb] - fact * B[i + j * ldb];
                    }
                } else {
                    *info = i + 1;  /* 1-based */
                    return;
                }
                DL[i] = ZERO;
            } else {
                /* Interchange rows i and i+1 */
                fact = D[i] / DL[i];
                D[i] = DL[i];
                temp = D[i + 1];
                D[i + 1] = DU[i] - fact * temp;
                DL[i] = DU[i + 1];
                DU[i + 1] = -fact * DL[i];
                DU[i] = temp;
                for (j = 0; j < nrhs; j++) {
                    temp = B[i + j * ldb];
                    B[i + j * ldb] = B[(i + 1) + j * ldb];
                    B[(i + 1) + j * ldb] = temp - fact * B[(i + 1) + j * ldb];
                }
            }
        }

        if (n > 1) {
            i = n - 2;
            if (fabsf(D[i]) >= fabsf(DL[i])) {
                if (D[i] != ZERO) {
                    fact = DL[i] / D[i];
                    D[i + 1] = D[i + 1] - fact * DU[i];
                    for (j = 0; j < nrhs; j++) {
                        B[(i + 1) + j * ldb] = B[(i + 1) + j * ldb] - fact * B[i + j * ldb];
                    }
                } else {
                    *info = i + 1;  /* 1-based */
                    return;
                }
            } else {
                fact = D[i] / DL[i];
                D[i] = DL[i];
                temp = D[i + 1];
                D[i + 1] = DU[i] - fact * temp;
                DU[i] = temp;
                for (j = 0; j < nrhs; j++) {
                    temp = B[i + j * ldb];
                    B[i + j * ldb] = B[(i + 1) + j * ldb];
                    B[(i + 1) + j * ldb] = temp - fact * B[(i + 1) + j * ldb];
                }
            }
        }

        if (D[n - 1] == ZERO) {
            *info = n;  /* 1-based */
            return;
        }
    }

    /*
     * Back solve with the matrix U from the factorization.
     * Note: In sgtsv, DL is reused for DU2 (second superdiagonal).
     */
    if (nrhs <= 2) {
        for (j = 0; j < nrhs; j++) {
            B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
            if (n > 1) {
                B[(n - 2) + j * ldb] = (B[(n - 2) + j * ldb] - DU[n - 2] * B[(n - 1) + j * ldb]) / D[n - 2];
            }
            for (i = n - 3; i >= 0; i--) {
                B[i + j * ldb] = (B[i + j * ldb] - DU[i] * B[(i + 1) + j * ldb]
                                  - DL[i] * B[(i + 2) + j * ldb]) / D[i];
            }
        }
    } else {
        for (j = 0; j < nrhs; j++) {
            B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
            if (n > 1) {
                B[(n - 2) + j * ldb] = (B[(n - 2) + j * ldb] - DU[n - 2] * B[(n - 1) + j * ldb]) / D[n - 2];
            }
            for (i = n - 3; i >= 0; i--) {
                B[i + j * ldb] = (B[i + j * ldb] - DU[i] * B[(i + 1) + j * ldb]
                                  - DL[i] * B[(i + 2) + j * ldb]) / D[i];
            }
        }
    }
}
