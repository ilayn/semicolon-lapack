/**
 * @file dtrtrs.c
 * @brief DTRTRS solves a triangular system of equations.
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DTRTRS solves a triangular system of the form
 *
 *    A * X = B  or  A**T * X = B,
 *
 * where A is a triangular matrix of order N, and B is an N-by-NRHS matrix.
 *
 * This routine verifies that A is nonsingular, but callers should note
 * that only exact singularity is detected. It is conceivable for one or
 * more diagonal elements of A to be subnormally tiny numbers without this
 * routine signalling an error.
 *
 * If a possible loss of numerical precision due to near-singular matrices
 * is a concern, the caller should verify that A is nonsingular within some
 * tolerance before calling this routine.
 *
 * @param[in]     uplo  = 'U': A is upper triangular;
 *                        = 'L': A is lower triangular.
 * @param[in]     trans Specifies the form of the system of equations:
 *                      - 'N': A * X = B (No transpose)
 *                      - 'T': A**T * X = B (Transpose)
 *                      - 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     diag  = 'N': A is non-unit triangular;
 *                        = 'U': A is unit triangular.
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B. nrhs >= 0.
 * @param[in]     A     Double precision array, dimension (lda, n).
 *                      The triangular matrix A. If uplo = "U", the leading
 *                      N-by-N upper triangular part of the array A contains
 *                      the upper triangular matrix, and the strictly lower
 *                      triangular part of A is not referenced. If uplo = "L",
 *                      the leading N-by-N lower triangular part of the array A
 *                      contains the lower triangular matrix, and the strictly
 *                      upper triangular part of A is not referenced.
 *                      If diag = "U", the diagonal elements of A are also not
 *                      referenced and are assumed to be 1.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,n).
 * @param[in,out] B     Double precision array, dimension (ldb, nrhs).
 *                      On entry, the right hand side matrix B.
 *                      On exit, if info = 0, the solution matrix X.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the i-th diagonal element of A is exactly
 *                           zero, indicating that the matrix is singular and the
 *                           solutions X have not been computed.
 */
void dtrtrs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int nrhs,
    const f64* restrict A,
    const int lda,
    f64* restrict B,
    const int ldb,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int nounit = (diag[0] == 'N' || diag[0] == 'n');

    // Test the input parameters
    *info = 0;

    if (uplo[0] != 'U' && uplo[0] != 'u' && uplo[0] != 'L' && uplo[0] != 'l') {
        *info = -1;
    } else if (trans[0] != 'N' && trans[0] != 'n' && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -2;
    } else if (!nounit && diag[0] != 'U' && diag[0] != 'u') {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (nrhs < 0) {
        *info = -5;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    }

    if (*info != 0) {
        xerbla("DTRTRS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0) {
        return;
    }

    // Check for singularity
    if (nounit) {
        for (int i = 0; i < n; i++) {
            if (A[i + i * lda] == ZERO) {
                *info = i + 1;
                return;
            }
        }
    }
    *info = 0;

    // Map character arguments to CBLAS enums
    CBLAS_UPLO uplo_flag = (uplo[0] == 'U' || uplo[0] == 'u') ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE trans_flag = (trans[0] == 'N' || trans[0] == 'n') ? CblasNoTrans : CblasTrans;
    CBLAS_DIAG diag_flag = nounit ? CblasNonUnit : CblasUnit;

    // Solve A * X = B  or  A**T * X = B
    cblas_dtrsm(CblasColMajor, CblasLeft, uplo_flag, trans_flag, diag_flag,
                n, nrhs, ONE, A, lda, B, ldb);
}
