/**
 * @file sgetrs.c
 * @brief Solves a system of linear equations using LU factorization (single precision).
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGETRS solves a system of linear equations
 *    A * X = B  or  A**T * X = B
 * with a general N-by-N matrix A using the LU factorization computed
 * by SGETRF.
 *
 * @param[in]     trans Specifies the form of the system of equations:
 *                      - 'N': A * X = B (No transpose)
 *                      - 'T': A**T * X = B (Transpose)
 *                      - 'C': A**T * X = B (Conjugate transpose = Transpose)
 * @param[in]     n     The order of the matrix A (n >= 0).
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B (nrhs >= 0).
 * @param[in]     A     The factors L and U from the factorization A = P*L*U
 *                      as computed by sgetrf. Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,n)).
 * @param[in]     ipiv  The pivot indices from sgetrf; row i was interchanged
 *                      with row ipiv[i]. Array of dimension n, 0-based.
 * @param[in,out] B     On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B (ldb >= max(1,n)).
 * @param[out]    info  Exit status:
 *                      - = 0: successful exit
 *                      - < 0: if info = -i, the i-th argument had an illegal value
 */
void sgetrs(
    const char* trans,
    const int n,
    const int nrhs,
    const float * const restrict A,
    const int lda,
    const int * const restrict ipiv,
    float * const restrict B,
    const int ldb,
    int *info)
{
    const float ONE = 1.0f;
    int notran;

    // Test the input parameters
    *info = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    }

    if (*info != 0) {
        xerbla("SGETRS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        return;
    }

    if (notran) {
        // Solve A * X = B.

        // Apply row interchanges to the right hand sides.
        slaswp(nrhs, B, ldb, 0, n - 1, ipiv, 1);

        // Solve L*X = B, overwriting B with X.
        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n, nrhs, ONE, A, lda, B, ldb);

        // Solve U*X = B, overwriting B with X.
        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    n, nrhs, ONE, A, lda, B, ldb);
    } else {
        // Solve A**T * X = B.

        // Solve U**T * X = B, overwriting B with X.
        cblas_strsm(CblasColMajor, CblasLeft, CblasUpper, CblasTrans, CblasNonUnit,
                    n, nrhs, ONE, A, lda, B, ldb);

        // Solve L**T * X = B, overwriting B with X.
        cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasTrans, CblasUnit,
                    n, nrhs, ONE, A, lda, B, ldb);

        // Apply row interchanges to the solution vectors.
        slaswp(nrhs, B, ldb, 0, n - 1, ipiv, -1);
    }
}
