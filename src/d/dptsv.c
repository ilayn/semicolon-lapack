/**
 * @file dptsv.c
 * @brief DPTSV computes the solution to a real system of linear equations
 *        A*X = B, where A is a symmetric positive definite tridiagonal matrix.
 */

#include "semicolon_lapack_double.h"

/**
 * DPTSV computes the solution to a real system of linear equations
 * A*X = B, where A is an N-by-N symmetric positive definite tridiagonal
 * matrix, and X and B are N-by-NRHS matrices.
 *
 * A is factored as A = L*D*L**T, and the factored form of A is then
 * used to solve the system of equations.
 *
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number
 *                      of columns of the matrix B. nrhs >= 0.
 * @param[in,out] D     Double precision array, dimension (n).
 *                      On entry, the n diagonal elements of the tridiagonal
 *                      matrix A.
 *                      On exit, the n diagonal elements of the diagonal
 *                      matrix D from the factorization A = L*D*L**T.
 * @param[in,out] E     Double precision array, dimension (n-1).
 *                      On entry, the (n-1) subdiagonal elements of the
 *                      tridiagonal matrix A.
 *                      On exit, the (n-1) subdiagonal elements of the unit
 *                      bidiagonal factor L from the L*D*L**T factorization
 *                      of A. (E can also be regarded as the superdiagonal
 *                      of the unit bidiagonal factor U from the U**T*D*U
 *                      factorization of A.)
 * @param[in,out] B     Double precision array, dimension (ldb, nrhs).
 *                      On entry, the N-by-NRHS right hand side matrix B.
 *                      On exit, if info = 0, the N-by-NRHS solution matrix X.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the leading principal minor of order i
 *                           is not positive, and the solution has not been
 *                           computed. The factorization has not been completed
 *                           unless i = n.
 */
void dptsv(
    const INT n,
    const INT nrhs,
    f64* restrict D,
    f64* restrict E,
    f64* restrict B,
    const INT ldb,
    INT* info)
{
    INT max_n_1 = (1 > n) ? 1 : n;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (ldb < max_n_1) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("DPTSV ", -(*info));
        return;
    }

    /* Compute the L*D*L**T (or U**T*D*U) factorization of A. */

    dpttrf(n, D, E, info);
    if (*info == 0) {

        /* Solve the system A*X = B, overwriting B with X. */

        dpttrs(n, nrhs, D, E, B, ldb, info);
    }
}
