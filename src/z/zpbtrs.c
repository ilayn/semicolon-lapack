/**
 * @file zpbtrs.c
 * @brief ZPBTRS solves a system with a Hermitian positive definite band matrix using Cholesky.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBTRS solves a system of linear equations A*X = B with a Hermitian
 * positive definite band matrix A using the Cholesky factorization
 * A = U**H*U or A = L*L**H computed by ZPBTRF.
 *
 * @param[in]     uplo   = 'U': Upper triangular factor stored in AB
 *                        = 'L': Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AB     The triangular factor from ZPBTRF. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zpbtrs(
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    const c128* restrict AB,
    const INT ldab,
    c128* restrict B,
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
        xerbla("ZPBTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0)
        return;

    if (upper) {
        // Solve A*X = B where A = U**H * U
        for (j = 0; j < nrhs; j++) {
            // Solve U**H * X = B
            cblas_ztbsv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
            // Solve U * X = B
            cblas_ztbsv(CblasColMajor, CblasUpper, CblasNoTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
        }
    } else {
        // Solve A*X = B where A = L * L**H
        for (j = 0; j < nrhs; j++) {
            // Solve L * X = B
            cblas_ztbsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
            // Solve L**H * X = B
            cblas_ztbsv(CblasColMajor, CblasLower, CblasConjTrans, CblasNonUnit,
                        n, kd, AB, ldab, &B[j * ldb], 1);
        }
    }
}
