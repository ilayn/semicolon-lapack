/**
 * @file dspsv.c
 * @brief DSPSV computes the solution to a real system of linear equations A * X = B.
 */

#include "semicolon_lapack_double.h"

/**
 * DSPSV computes the solution to a real system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B,
 * @endrst
 * where A is an N-by-N symmetric matrix stored in packed format and X
 * and B are N-by-NRHS matrices.
 *
 * The diagonal pivoting method is used to factor A as
 * @rst
 * .. code-block:: text
 *
 *     A = U * D * U**T,  if uplo = 'U', or
 *     A = L * D * L**T,  if uplo = 'L',
 * @endrst
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, D is symmetric and block diagonal with 1-by-1
 * and 2-by-2 diagonal blocks. The factored form of A is then used to
 * solve the system of equations A * X = B.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The number of linear equations, i.e., the order of
 *                       the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in,out] AP     Array of dimension `n*(n+1)/2`.
 *                       On entry, the upper or lower triangle of the
 *                       symmetric matrix A, packed columnwise in a linear
 *                       array. The j-th column of A is stored in the array
 *                       AP as follows: if `uplo='U'`,
 *                       `AP[i + j*(j+1)/2] = A(i,j)` for `0<=i<=j`; if
 *                       `uplo='L'`, `AP[i + j*(2*n-j-1)/2] = A(i,j)` for
 *                       `j<=i<=n-1`. See below for further details.
 *                       On exit, the block diagonal matrix D and the
 *                       multipliers used to obtain the factor U or L from
 *                       the factorization A = U*D*U**T or A = L*D*L**T as
 *                       computed by `dsptrf`, stored as a packed triangular
 *                       matrix in the same storage format as A.
 * @param[out]    ipiv   Array of dimension `n`. The pivot indices from
 *                       `dsptrf` (0-based).
 * @param[in,out] B      Array of dimension `(ldb,nrhs)`.
 *                       On entry, the n-by-nrhs right hand side matrix B.
 *                       On exit, if `info=0`, the n-by-nrhs solution matrix X.
 * @param[in]     ldb    The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, `D(i,i)` is exactly zero. The
 *                           factorization has been completed, but the block
 *                           diagonal matrix D is exactly singular, so the
 *                           solution could not be computed.
 *
 * @par Further Details:
 * @rst
 * The packed storage scheme is illustrated by the following example
 * when ``n=4``, ``uplo='U'``:
 *
 * Two-dimensional storage of the symmetric matrix A:
 *
 * .. code-block:: text
 *
 *     a00 a01 a02 a03
 *         a11 a12 a13
 *             a22 a23     (aij = aji)
 *                 a33
 *
 * Packed storage of the upper triangle of A:
 *
 * .. code-block:: text
 *
 *     AP = [ a00, a01, a11, a02, a12, a22, a03, a13, a23, a33 ]
 * @endrst
 */
void dspsv(
    const char* uplo,
    const INT n,
    const INT nrhs,
    f64* restrict AP,
    INT* restrict ipiv,
    f64* restrict B,
    const INT ldb,
    INT* info)
{
    *info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("DSPSV ", -(*info));
        return;
    }

    // Compute the factorization A = U*D*U**T or A = L*D*L**T
    dsptrf(uplo, n, AP, ipiv, info);
    if (*info == 0) {
        // Solve the system A*X = B, overwriting B with X
        dsptrs(uplo, n, nrhs, AP, ipiv, B, ldb, info);
    }
}
