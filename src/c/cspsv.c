/**
 * @file cspsv.c
 * @brief CSPSV computes the solution to a complex system of linear equations A * X = B.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CSPSV computes the solution to a complex system of linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric matrix stored in packed format and X
 * and B are N-by-NRHS matrices.
 *
 * The diagonal pivoting method is used to factor A as
 *    A = U * D * U**T,  if UPLO = 'U', or
 *    A = L * D * L**T,  if UPLO = 'L',
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, D is symmetric and block diagonal with 1-by-1
 * and 2-by-2 diagonal blocks.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] AP     On entry, the packed symmetric matrix A.
 *                       On exit, the factorization from CSPTRF.
 *                       Array of dimension (n*(n+1)/2).
 * @param[out]    ipiv   The pivot indices from CSPTRF. Array of dimension (n).
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) is exactly zero
 */
void cspsv(
    const char* uplo,
    const int n,
    const int nrhs,
    c64* restrict AP,
    int* restrict ipiv,
    c64* restrict B,
    const int ldb,
    int* info)
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
        xerbla("CSPSV ", -(*info));
        return;
    }

    // Compute the factorization A = U*D*U**T or A = L*D*L**T
    csptrf(uplo, n, AP, ipiv, info);
    if (*info == 0) {
        // Solve the system A*X = B, overwriting B with X
        csptrs(uplo, n, nrhs, AP, ipiv, B, ldb, info);
    }
}
