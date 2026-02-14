/**
 * @file spbtrs.c
 * @brief SPBTRS solves a system with a symmetric positive definite band matrix using Cholesky.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SPBTRS solves a system of linear equations A*X = B with a symmetric
 * positive definite band matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by SPBTRF.
 *
 * @param[in]     uplo   = 'U': Upper triangular factor stored in AB
 *                        = 'L': Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AB     The triangular factor from SPBTRF. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void spbtrs(
    const char* uplo,
    const int n,
    const int kd,
    const int nrhs,
    const f32* restrict AB,
    const int ldab,
    f32* restrict B,
    const int ldb,
    int* info)
{
    int upper;
    int j;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

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
        xerbla("SPBTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0)
        return;

    if (upper) {
        // Solve A*X = B where A = U**T * U
        for (j = 0; j < nrhs; j++) {
            // Solve U**T * X = B
            cblas_stbsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
            // Solve U * X = B
            cblas_stbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
        }
    } else {
        // Solve A*X = B where A = L * L**T
        for (j = 0; j < nrhs; j++) {
            // Solve L * X = B
            cblas_stbsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
            // Solve L**T * X = B
            cblas_stbsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
        }
    }
}
