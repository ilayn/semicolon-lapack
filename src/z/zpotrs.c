/**
 * @file zpotrs.c
 * @brief ZPOTRS solves a system of linear equations using the Cholesky factorization.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPOTRS solves a system of linear equations A*X = B with a Hermitian
 * positive definite matrix A using the Cholesky factorization
 * A = U**H*U or A = L*L**H computed by ZPOTRF.
 *
 * @param[in]     uplo  Specifies whether the factor stored in A is upper or
 *                      lower triangular.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     A     The triangular factor U or L from the Cholesky
 *                      factorization A = U**H*U or A = L*L**H, as computed
 *                      by zpotrf. Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in,out] B     On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void zpotrs(
    const char* uplo,
    const int n,
    const int nrhs,
    const c128* restrict A,
    const int lda,
    c128* restrict B,
    const int ldb,
    int* info)
{
    const c128 ONE = CMPLX(1.0, 0.0);

    // Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
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
        xerbla("ZPOTRS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) return;

    if (upper) {
        // Solve A*X = B where A = U**H * U.

        // Solve U**H * X = B, overwriting B with X.
        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasConjTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);

        // Solve U * X = B, overwriting B with X.
        cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);
    } else {
        // Solve A*X = B where A = L * L**H.

        // Solve L * X = B, overwriting B with X.
        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);

        // Solve L**H * X = B, overwriting B with X.
        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasConjTrans,
                    CblasNonUnit, n, nrhs, &ONE, A, lda, B, ldb);
    }
}
