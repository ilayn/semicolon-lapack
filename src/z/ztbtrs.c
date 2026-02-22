/**
 * @file ztbtrs.c
 * @brief ZTBTRS solves a triangular banded system of equations.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZTBTRS solves a triangular system of the form
 *
 *    A * X = B,  A**T * X = B,  or  A**H * X = B,
 *
 * where A is a triangular band matrix of order N, and B is an N-by-NRHS matrix.
 *
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     trans  = 'N': A * X = B (No transpose)
 *                        = 'T': A**T * X = B (Transpose)
 *                        = 'C': A**H * X = B (Conjugate transpose)
 * @param[in]     diag   = 'N': A is non-unit triangular
 *                        = 'U': A is unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of superdiagonals (if uplo='U') or
 *                       subdiagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AB     The triangular band matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element is zero,
 *                           indicating the matrix is singular
 */
void ztbtrs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const INT n,
    const INT kd,
    const INT nrhs,
    const c128* restrict AB,
    const INT ldab,
    c128* restrict B,
    const INT ldb,
    INT* info)
{
    const c128 ZERO = CMPLX(0.0, 0.0);

    INT nounit, upper;
    INT j;

    *info = 0;
    nounit = (diag[0] == 'N' || diag[0] == 'n');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!(trans[0] == 'N' || trans[0] == 'n') &&
               !(trans[0] == 'T' || trans[0] == 't') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (kd < 0) {
        *info = -5;
    } else if (nrhs < 0) {
        *info = -6;
    } else if (ldab < kd + 1) {
        *info = -8;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("ZTBTRS", -(*info));
        return;
    }

    if (n == 0)
        return;

    // Check for singularity
    if (nounit) {
        if (upper) {
            for (*info = 1; *info <= n; (*info)++) {
                if (AB[kd + (*info - 1) * ldab] == ZERO)
                    return;
            }
        } else {
            for (*info = 1; *info <= n; (*info)++) {
                if (AB[0 + (*info - 1) * ldab] == ZERO)
                    return;
            }
        }
    }
    *info = 0;

    // Solve A * X = B,  A**T * X = B,  or  A**H * X = B.
    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'N' || trans[0] == 'n') {
        cblas_trans = CblasNoTrans;
    } else if (trans[0] == 'T' || trans[0] == 't') {
        cblas_trans = CblasTrans;
    } else {
        cblas_trans = CblasConjTrans;
    }
    CBLAS_DIAG cblas_diag = nounit ? CblasNonUnit : CblasUnit;

    for (j = 0; j < nrhs; j++) {
        cblas_ztbsv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                    n, kd, AB, ldab, &B[j * ldb], 1);
    }
}
