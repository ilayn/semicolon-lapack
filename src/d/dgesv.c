#include "semicolon_lapack_double.h"
/**
 * @file dgesv.c
 * @brief Solves a general system of linear equations A * X = B.
 */

/**
 * DGESV computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
 *
 * The LU decomposition with partial pivoting and row interchanges is
 * used to factor A as
 *    A = P * L * U,
 * where P is a permutation matrix, L is unit lower triangular, and U is
 * upper triangular. The factored form of A is then used to solve the
 * system of equations A * X = B.
 *
 * @param[in]     n     The number of linear equations, i.e., the order of the
 *                      matrix A (n >= 0).
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B (nrhs >= 0).
 * @param[in,out] A     On entry, the N-by-N coefficient matrix A.
 *                      On exit, the factors L and U from the factorization
 *                      A = P*L*U; the unit diagonal elements of L are not stored.
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,n)).
 * @param[out]    ipiv  The pivot indices that define the permutation matrix P;
 *                      row i of the matrix was interchanged with row ipiv[i].
 *                      Array of dimension n, 0-based.
 * @param[in,out] B     On entry, the N-by-NRHS matrix of right hand side matrix B.
 *                      On exit, if info = 0, the N-by-NRHS solution matrix X.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B (ldb >= max(1,n)).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but the factor U is exactly
 *                           singular, so the solution could not be computed.
 */
void dgesv(
    const int n,
    const int nrhs,
    f64* restrict A,
    const int lda,
    int* restrict ipiv,
    f64* restrict B,
    const int ldb,
    int* info)
{
    // Test the input parameters
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    }

    if (*info != 0) {
        xerbla("DGESV ", -(*info));
        return;
    }

    // Compute the LU factorization of A
    dgetrf(n, n, A, lda, ipiv, info);

    if (*info == 0) {
        // Solve the system A*X = B, overwriting B with X
        dgetrs("N", n, nrhs, A, lda, ipiv, B, ldb, info);
    }
}
