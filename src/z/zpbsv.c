/**
 * @file zpbsv.c
 * @brief ZPBSV computes the solution to a Hermitian positive definite banded system A * X = B.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZPBSV computes the solution to a complex system of linear equations
 *    A * X = B,
 * where A is an N-by-N Hermitian positive definite band matrix and X
 * and B are N-by-NRHS matrices.
 *
 * The Cholesky decomposition is used to factor A as
 *    A = U**H * U,  if UPLO = 'U', or
 *    A = L * L**H,  if UPLO = 'L',
 * where U is an upper triangular band matrix, and L is a lower
 * triangular band matrix.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] AB     On entry, the banded matrix A. On exit, the factor U or L.
 *                       Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the leading minor of order i is not
 *                           positive definite.
 */
void zpbsv(
    const char* uplo,
    const int n,
    const int kd,
    const int nrhs,
    c128* restrict AB,
    const int ldab,
    c128* restrict B,
    const int ldb,
    int* info)
{
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldab < kd + 1) {
        *info = -6;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("ZPBSV ", -(*info));
        return;
    }

    // Compute the Cholesky factorization A = U**H*U or A = L*L**H
    zpbtrf(uplo, n, kd, AB, ldab, info);
    if (*info == 0) {
        // Solve the system A*X = B
        zpbtrs(uplo, n, kd, nrhs, AB, ldab, B, ldb, info);
    }
}
