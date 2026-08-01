/**
 * @file cpotrs.c
 * @brief CPOTRS solves a system of linear equations using the Cholesky factorization.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPOTRS solves a system of linear equations A*X = B with a Hermitian
 * positive definite matrix A using the Cholesky factorization
 * A = U**H*U or A = L*L**H computed by CPOTRF.
 *
 * @param[in]     uplo
 *                      - `'U'`: Upper triangle of A is stored
 *                      - `'L'`: Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. `n>=0`.
 * @param[in]     nrhs  The number of right hand sides. `nrhs>=0`.
 * @param[in]     A     Array of dimension (`lda`, `n`).
 *                      The triangular factor U or L from the Cholesky
 *                      factorization A = U**H*U or A = L*L**H, as computed
 *                      by `cpotrf`.
 * @param[in]     lda   The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[in,out] B     Array of dimension (`ldb`, `nrhs`).
 *                      On entry, the right hand side matrix `B`.
 *                      On exit, the solution matrix X.
 * @param[in]     ldb   The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-k`, the k-th argument had an illegal
 *                           value
 */
void cpotrs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c64* restrict A,
    const INT lda,
    c64* restrict B,
    const INT ldb,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);

    // Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
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
        xerbla("CPOTRS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) return;

    if (upper) {
        // Solve A*X = B where A = U**H * U.

        // Solve U**H * X = B, overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);

        // Solve U * X = B, overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);
    } else {
        // Solve A*X = B where A = L * L**H.

        // Solve L * X = B, overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);

        // Solve L**H * X = B, overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);
    }
}
