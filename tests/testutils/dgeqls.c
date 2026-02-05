/**
 * @file dgeqls.c
 * @brief DGEQLS solves the least squares problem min || A*X - B || using the
 *        QL factorization A = Q*L computed by DGEQLF.
 */

#include <cblas.h>
#include "verify.h"
#include "semicolon_lapack_double.h"

/**
 * Solve the least squares problem
 *     min || A*X - B ||
 * using the QL factorization
 *     A = Q*L
 * computed by DGEQLF.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. m >= n >= 0.
 * @param[in]     nrhs   The number of columns of B. nrhs >= 0.
 * @param[in]     A      Details of the QL factorization of the original matrix A
 *                       as returned by DGEQLF. Double precision array, dimension
 *                       (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= m.
 * @param[in]     tau    Details of the orthogonal matrix Q. Double precision
 *                       array, dimension (n).
 * @param[in,out] B      On entry, the m-by-nrhs right hand side matrix B.
 *                       On exit, the n-by-nrhs solution matrix X, stored in
 *                       rows m-n:m-1 (0-indexed). Double precision array,
 *                       dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= m.
 * @param[out]    work   Workspace array, dimension (lwork).
 * @param[in]     lwork  The length of the array work. lwork must be at least
 *                       nrhs, and should be at least nrhs*NB.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 */
void dgeqls(
    const int m,
    const int n,
    const int nrhs,
    double* A,
    const int lda,
    const double* tau,
    double* B,
    const int ldb,
    double* work,
    const int lwork,
    int* info)
{
    const double ONE = 1.0;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || n > m) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldb < (m > 1 ? m : 1)) {
        *info = -8;
    } else if (lwork < 1 || (lwork < nrhs && m > 0 && n > 0)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DGEQLS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0 || m == 0)
        return;

    dormql("L", "T", m, nrhs, n, A, lda, tau, B, ldb, work, lwork, info);

    cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit,
                n, nrhs, ONE, &A[(m - n) + 0 * lda], lda, &B[m - n], ldb);
}
