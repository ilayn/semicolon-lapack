#include "semicolon_lapack_complex_double.h"
#include <complex.h>
/**
 * @file zgesv.c
 * @brief Solves a general system of linear equations A * X = B.
 */

/**
 * ZGESV computes the solution to a complex system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where `A` is an `n`-by-`n` matrix and X and `B` are `n`-by-`nrhs` matrices.
 *
 * The LU decomposition with partial pivoting and row interchanges is
 * used to factor `A` as
 * @rst
 * .. code-block:: text
 *
 *     A = P * L * U
 * @endrst
 * where P is a permutation matrix, L is unit lower triangular, and U is
 * upper triangular. The factored form of `A` is then used to solve the
 * system of equations A * X = B.
 *
 * @param[in]     n     The number of linear equations, i.e., the order of the
 *                      matrix `A`. `n>=0`.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix `B`. `nrhs>=0`.
 * @param[in,out] A     Array of dimension (`lda`, `n`).
 *                      On entry, the `n`-by-`n` coefficient matrix `A`.
 *                      On exit, the factors L and U from the factorization
 *                      A = P*L*U; the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[out]    ipiv  Array of dimension `n`.
 *                      The pivot indices that define the permutation matrix P;
 *                      row i of the matrix was interchanged with row `ipiv[i]`.
 * @param[in,out] B     Array of dimension (`ldb`, `nrhs`).
 *                      On entry, the `n`-by-`nrhs` right hand side matrix `B`.
 *                      On exit, if `info=0`, the `n`-by-`nrhs` solution matrix X.
 * @param[in]     ldb   The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    info
 *                          - `info=0`: successful exit
 *                          - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                            value
 *                          - `info>0`: if `info=i`, U(i,i) is exactly zero. The
 *                            factorization has been completed, but the factor U is
 *                            exactly singular, so the solution could not be computed.
 */
void zgesv(
    const INT n,
    const INT nrhs,
    c128* restrict A,
    const INT lda,
    INT* restrict ipiv,
    c128* restrict B,
    const INT ldb,
    INT* info)
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
        xerbla("ZGESV ", -(*info));
        return;
    }

    // Compute the LU factorization of A
    zgetrf(n, n, A, lda, ipiv, info);

    if (*info == 0) {
        // Solve the system A*X = B, overwriting B with X
        zgetrs("N", n, nrhs, A, lda, ipiv, B, ldb, info);
    }
}
