/**
 * @file cgtsv.c
 * @brief CGTSV computes the solution to system of linear equations A * X = B
 *        for GT (tridiagonal) matrices.
 */

#include <complex.h>
#include <math.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGTSV solves the equation
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
void cgtsv(
    const INT n,
    const INT nrhs,
    c64* restrict DL,
    c64* restrict D,
    c64* restrict DU,
    c64* restrict B,
    const INT ldb,
    INT* info)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    INT j, k;
    c64 mult, temp;
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
        xerbla("CGTSV ", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    for (k = 0; k < n - 1; k++) {
        if (DL[k] == ZERO) {
            /*
             * Subdiagonal is zero, no elimination is required.
             */
            if (D[k] == ZERO) {
                /*
                 * Diagonal is zero: set INFO = K and return; a unique
                 * solution can not be found.
                 */
                *info = k + 1;
                return;
            }
        } else if (cabs1f(D[k]) >= cabs1f(DL[k])) {
            /*
             * No row interchange required
             */
            mult = DL[k] / D[k];
            D[k + 1] = D[k + 1] - mult * DU[k];
            for (j = 0; j < nrhs; j++) {
                B[(k + 1) + j * ldb] = B[(k + 1) + j * ldb] - mult * B[k + j * ldb];
            }
            if (k < (n - 2)) {
                DL[k] = ZERO;
            }
        } else {
            /*
             * Interchange rows K and K+1
             */
            mult = D[k] / DL[k];
            D[k] = DL[k];
            temp = D[k + 1];
            D[k + 1] = DU[k] - mult * temp;
            if (k < (n - 2)) {
                DL[k] = DU[k + 1];
                DU[k + 1] = -mult * DL[k];
            }
            DU[k] = temp;
            for (j = 0; j < nrhs; j++) {
                temp = B[k + j * ldb];
                B[k + j * ldb] = B[(k + 1) + j * ldb];
                B[(k + 1) + j * ldb] = temp - mult * B[(k + 1) + j * ldb];
            }
        }
    }
    if (D[n - 1] == ZERO) {
        *info = n;
        return;
    }

    /*
     * Back solve with the matrix U from the factorization.
     */
    for (j = 0; j < nrhs; j++) {
        B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
        if (n > 1) {
            B[(n - 2) + j * ldb] = (B[(n - 2) + j * ldb] - DU[n - 2] * B[(n - 1) + j * ldb]) / D[n - 2];
        }
        for (k = n - 3; k >= 0; k--) {
            B[k + j * ldb] = (B[k + j * ldb] - DU[k] * B[(k + 1) + j * ldb]
                              - DL[k] * B[(k + 2) + j * ldb]) / D[k];
        }
    }
}
