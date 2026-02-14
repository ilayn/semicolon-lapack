/**
 * @file stptrs.c
 * @brief STPTRS solves a triangular system with a packed triangular matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * STPTRS solves a triangular system of the form
 *
 *    A * X = B  or  A**T * X = B,
 *
 * where A is a triangular matrix of order N stored in packed format,
 * and B is an N-by-NRHS matrix.
 *
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     trans  = 'N': A * X = B (No transpose)
 *                        = 'T': A**T * X = B (Transpose)
 *                        = 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     diag   = 'N': A is non-unit triangular
 *                        = 'U': A is unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The packed triangular matrix A. Array of dimension (n*(n+1)/2).
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, the solution matrix X.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element is exactly zero
 */
void stptrs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int nrhs,
    const f32* const restrict AP,
    f32* const restrict B,
    const int ldb,
    int* info)
{
    const f32 ZERO = 0.0f;

    int nounit, upper;
    int j, jc;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    nounit = (diag[0] == 'N' || diag[0] == 'n');

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
    } else if (nrhs < 0) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("STPTRS", -(*info));
        return;
    }

    if (n == 0)
        return;

    // Check for singularity
    if (nounit) {
        if (upper) {
            jc = 0;
            for (*info = 1; *info <= n; (*info)++) {
                if (AP[jc + *info - 1] == ZERO)
                    return;
                jc = jc + *info;
            }
        } else {
            jc = 0;
            for (*info = 1; *info <= n; (*info)++) {
                if (AP[jc] == ZERO)
                    return;
                jc = jc + n - *info + 1;
            }
        }
    }
    *info = 0;

    // Solve A * x = b or A**T * x = b
    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'N' || trans[0] == 'n') {
        cblas_trans = CblasNoTrans;
    } else {
        cblas_trans = CblasTrans;
    }
    CBLAS_DIAG cblas_siag = nounit ? CblasNonUnit : CblasUnit;

    for (j = 0; j < nrhs; j++) {
        cblas_stpsv(CblasColMajor, cblas_uplo, cblas_trans, cblas_siag,
                    n, AP, &B[j * ldb], 1);
    }
}
