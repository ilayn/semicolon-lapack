/**
 * @file spbtrs.c
 * @brief SPBTRS solves a system with a symmetric positive definite band matrix using Cholesky.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SPBTRS solves a system of linear equations A*X = B with a symmetric
 * positive definite band matrix A using the Cholesky factorization
 * A = U**T*U or A = L*L**T computed by SPBTRF.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular factor stored in AB
 *                       - `'L'`: Lower triangular factor stored in AB
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in]     AB     Array of dimension (`ldab`, `n`).
 *                       The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T of the band
 *                       matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of U or L is stored in the
 *                       j-th column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = U(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = L(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[in,out] B      Array of dimension (`ldb`, `nrhs`).
 *                       On entry, the right hand side matrix `B`.
 *                       On exit, the solution matrix X.
 * @param[in]     ldb    The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void spbtrs(
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    const f32* restrict AB,
    const INT ldab,
    f32* restrict B,
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
