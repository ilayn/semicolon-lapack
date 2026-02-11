/**
 * @file spptrs.c
 * @brief SPPTRS solves a system of linear equations using the Cholesky factorization of a packed matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SPPTRS solves a system of linear equations A*X = B with a symmetric
 * positive definite matrix A in packed storage using the Cholesky
 * factorization A = U**T*U or A = L*L**T computed by SPPTRF.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides, i.e., the number of
 *                       columns of the matrix B. nrhs >= 0.
 * @param[in]     AP     The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, packed
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
void spptrs(
    const char* uplo,
    const int n,
    const int nrhs,
    const float* const restrict AP,
    float* const restrict B,
    const int ldb,
    int* info)
{
    // spptrs.f lines 140-154: Test the input parameters
    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
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
        xerbla("SPPTRS", -(*info));
        return;
    }

    // spptrs.f lines 158-159: Quick return if possible
    if (n == 0 || nrhs == 0) {
        return;
    }

    if (upper) {
        // spptrs.f lines 161-176: Solve A*X = B where A = U**T * U.
        for (int i = 0; i < nrhs; i++) {  // spptrs.f line 165: DO 10 I = 1, NRHS
            // spptrs.f lines 169-170: Solve U**T *X = B, overwriting B with X.
            cblas_stpsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
            // spptrs.f lines 174-175: Solve U*X = B, overwriting B with X.
            cblas_stpsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
        }
    } else {
        // spptrs.f lines 177-193: Solve A*X = B where A = L * L**T.
        for (int i = 0; i < nrhs; i++) {  // spptrs.f line 181: DO 20 I = 1, NRHS
            // spptrs.f lines 185-186: Solve L*Y = B, overwriting B with X.
            cblas_stpsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
            // spptrs.f lines 190-191: Solve L**T *X = Y, overwriting B with X.
            cblas_stpsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                        n, AP, &B[i * ldb], 1);
        }
    }
}
