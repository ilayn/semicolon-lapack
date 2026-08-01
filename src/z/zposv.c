/**
 * @file zposv.c
 * @brief ZPOSV computes the solution to a complex system of linear equations
 *        A * X = B for Hermitian positive definite matrices.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZPOSV computes the solution to a complex system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where A is an `n`-by-`n` Hermitian positive definite matrix and X and `B`
 * are `n`-by-`nrhs` matrices.
 *
 * The Cholesky decomposition is used to factor A as
 * @rst
 * .. code-block:: text
 *
 *     A = U**H * U, if uplo = 'U', or
 *     A = L * L**H, if uplo = 'L',
 * @endrst
 * where U is an upper triangular matrix and L is a lower triangular
 * matrix. The factored form of A is then used to solve the system of
 * equations A * X = B.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangle of A is stored
 *                      - `'L'`: Lower triangle of A is stored
 * @param[in]     n     The number of linear equations. `n>=0`.
 * @param[in]     nrhs  The number of right hand sides. `nrhs>=0`.
 * @param[in,out] A     Array of dimension (`lda`, `n`).
 *                      On entry, the Hermitian matrix A. If `uplo='U'`, the
 *                      leading `n`-by-`n` upper triangular part of A contains the
 *                      upper triangular part of the matrix A, and the strictly
 *                      lower triangular part of A is not referenced. If `uplo='L'`,
 *                      the leading `n`-by-`n` lower triangular part of A contains
 *                      the lower triangular part of the matrix A, and the strictly
 *                      upper triangular part of A is not referenced.
 *                      On exit, if `info=0`, the factor U or L from the Cholesky
 *                      factorization A = U**H*U or A = L*L**H.
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[in,out] B     Array of dimension (`ldb`, `nrhs`).
 *                      On entry, the `n`-by-`nrhs` right hand side matrix `B`.
 *                      On exit, if `info=0`, the `n`-by-`nrhs` solution matrix X.
 * @param[in]     ldb   The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-k`, the k-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=k`, the leading principal minor of order
 *                           k of A is not positive, so the factorization could not
 *                           be completed, and the solution has not been computed.
 */
void zposv(
    const char* uplo,
    const INT n,
    const INT nrhs,
    c128* restrict A,
    const INT lda,
    c128* restrict B,
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
        xerbla("ZPOSV", -(*info));
        return;
    }

    // Compute the Cholesky factorization A = U**H *U or A = L*L**H.
    zpotrf(uplo, n, A, lda, info);
    if (*info == 0) {
        // Solve the system A*X = B, overwriting B with X.
        zpotrs(uplo, n, nrhs, A, lda, B, ldb, info);
    }
}
