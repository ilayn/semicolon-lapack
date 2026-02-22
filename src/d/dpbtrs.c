/**
 * @file dpbtrs.c
 * @brief DPBTRS solves a system with a symmetric positive definite band matrix using Cholesky.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DPBTRS solves a system of linear equations A*X = B with a symmetric
 * positive definite band matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by DPBTRF.
 *
 * @param[in]     uplo   = 'U': Upper triangular factor stored in AB
 *                        = 'L': Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AB     The triangular factor from DPBTRF. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dpbtrs(
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    const f64* restrict AB,
    const INT ldab,
    f64* restrict B,
    const INT ldb,
    INT* info)
{
    INT upper;
    INT j;

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
        xerbla("DPBTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0)
        return;

    if (upper) {
        // Solve A*X = B where A = U**T * U
        for (j = 0; j < nrhs; j++) {
            // Solve U**T * X = B
            cblas_dtbsv(CblasColMajor, CblasUpper, CblasTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
            // Solve U * X = B
            cblas_dtbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
        }
    } else {
        // Solve A*X = B where A = L * L**T
        for (j = 0; j < nrhs; j++) {
            // Solve L * X = B
            cblas_dtbsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
            // Solve L**T * X = B
            cblas_dtbsv(CblasColMajor, CblasLower, CblasTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
        }
    }
}
