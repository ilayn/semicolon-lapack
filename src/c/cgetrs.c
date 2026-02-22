/**
 * @file cgetrs.c
 * @brief Solves a system of linear equations using LU factorization.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGETRS solves a system of linear equations
 *    A * X = B,  A**T * X = B,  or  A**H * X = B
 * with a general N-by-N matrix A using the LU factorization computed
 * by CGETRF.
 *
 * @param[in]     trans Specifies the form of the system of equations:
 *                      - 'N': A * X = B (No transpose)
 *                      - 'T': A**T * X = B (Transpose)
 *                      - 'C': A**H * X = B (Conjugate transpose)
 * @param[in]     n     The order of the matrix A (n >= 0).
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B (nrhs >= 0).
 * @param[in]     A     The factors L and U from the factorization A = P*L*U
 *                      as computed by cgetrf. Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,n)).
 * @param[in]     ipiv  The pivot indices from cgetrf; row i was interchanged
 *                      with row ipiv[i]. Array of dimension n, 0-based.
 * @param[in,out] B     On entry, the right hand side matrix B.
 *                      On exit, the solution matrix X.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B (ldb >= max(1,n)).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 */
void cgetrs(
    const char* trans,
    const INT n,
    const INT nrhs,
    const c64* restrict A,
    const INT lda,
    const INT* restrict ipiv,
    c64* restrict B,
    const INT ldb,
    INT* info)
{
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    INT notran;

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
        xerbla("CGETRS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        return;
    }

    if (notran) {
        // Solve A * X = B.

        // Apply row interchanges to the right hand sides.
        // Note: LAPACK uses 1-based k1=1, k2=N; we use 0-based k1=0, k2=n-1
        claswp(nrhs, B, ldb, 0, n - 1, ipiv, 1);

        // Solve L*X = B, overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        // Solve U*X = B, overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);
    } else {
        // Solve A**T * X = B  or A**H * X = B.

        enum CBLAS_TRANSPOSE cblas_trans = (trans[0] == 'C' || trans[0] == 'c')
                                            ? CblasConjTrans : CblasTrans;

        // Solve U**T *X = B or U**H *X = B, overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasUpper, cblas_trans, CblasNonUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        // Solve L**T *X = B, or L**H *X = B overwriting B with X.
        cblas_ctrsm(CblasColMajor, CblasLeft, CblasLower, cblas_trans, CblasUnit,
                    n, nrhs, &ONE, A, lda, B, ldb);

        // Apply row interchanges to the solution vectors.
        // Note: incx=-1 means apply pivots in reverse order
        claswp(nrhs, B, ldb, 0, n - 1, ipiv, -1);
    }
}
