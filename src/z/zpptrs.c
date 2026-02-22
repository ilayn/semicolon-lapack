/**
 * @file zpptrs.c
 * @brief ZPPTRS solves a system of linear equations using the Cholesky factorization of a packed matrix.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZPPTRS solves a system of linear equations A*X = B with a Hermitian
 * positive definite matrix A in packed storage using the Cholesky
 * factorization A = U**H*U or A = L*L**H computed by ZPPTRF.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides, i.e., the number of
 *                       columns of the matrix B. nrhs >= 0.
 * @param[in]     AP     The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H, packed
 *                       columnwise in a linear array.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zpptrs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c128* restrict AP,
    c128* restrict B,
    const INT ldb,
    INT* info)
{
    // zpptrs.f lines 140-154: Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("ZPPTRS", -(*info));
        return;
    }

    // zpptrs.f lines 158-159: Quick return if possible
    if (n == 0 || nrhs == 0) {
        return;
    }

    if (upper) {
        // zpptrs.f lines 161-177: Solve A*X = B where A = U**H * U.
        for (INT i = 0; i < nrhs; i++) {  // zpptrs.f line 165: DO 10 I = 1, NRHS
            // zpptrs.f lines 169-171: Solve U**H *X = B, overwriting B with X.
            cblas_ztpsv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
            // zpptrs.f lines 175-176: Solve U*X = B, overwriting B with X.
            cblas_ztpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
        }
    } else {
        // zpptrs.f lines 178-195: Solve A*X = B where A = L * L**H.
        for (INT i = 0; i < nrhs; i++) {  // zpptrs.f line 182: DO 20 I = 1, NRHS
            // zpptrs.f lines 186-187: Solve L*Y = B, overwriting B with X.
            cblas_ztpsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
            // zpptrs.f lines 191-193: Solve L**H *X = Y, overwriting B with X.
            cblas_ztpsv(CblasColMajor, CblasLower, CblasConjTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
        }
    }
}
