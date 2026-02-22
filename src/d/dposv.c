/**
 * @file dposv.c
 * @brief DPOSV computes the solution to a real system of linear equations
 *        A * X = B for symmetric positive definite matrices.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"

/**
 * DPOSV computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric positive definite matrix and X and B
 * are N-by-NRHS matrices.
 *
 * The Cholesky decomposition is used to factor A as
 *    A = U**T * U, if UPLO = 'U', or
 *    A = L * L**T, if UPLO = 'L',
 * where U is an upper triangular matrix and L is a lower triangular
 * matrix. The factored form of A is then used to solve the system of
 * equations A * X = B.
 *
 * @param[in]     uplo  = 'U': Upper triangle of A is stored
 *                       = 'L': Lower triangle of A is stored
 * @param[in]     n     The number of linear equations. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in,out] A     On entry, the symmetric matrix A.
 *                      On exit, if info = 0, the factor U or L from the
 *                      Cholesky factorization A = U**T*U or A = L*L**T.
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in,out] B     On entry, the N-by-NRHS right hand side matrix B.
 *                      On exit, if info = 0, the N-by-NRHS solution matrix X.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the leading principal minor of order k
 *                           of A is not positive, so the factorization could not
 *                           be completed, and the solution has not been computed.
 */
void dposv(
    const char* uplo,
    const INT n,
    const INT nrhs,
    f64* restrict A,
    const INT lda,
    f64* restrict B,
    const INT ldb,
    INT* info)
{
    // Test the input parameters
    *info = 0;
    if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
        !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("DPOSV", -(*info));
        return;
    }

    // Compute the Cholesky factorization A = U**T*U or A = L*L**T.
    dpotrf(uplo, n, A, lda, info);
    if (*info == 0) {
        // Solve the system A*X = B, overwriting B with X.
        dpotrs(uplo, n, nrhs, A, lda, B, ldb, info);
    }
}
