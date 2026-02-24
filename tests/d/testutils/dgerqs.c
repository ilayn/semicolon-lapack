/**
 * @file dgerqs.c
 * @brief DGERQS computes a minimum-norm solution min || A*X - B || using the
 *        RQ factorization A = R*Q computed by DGERQF.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include "semicolon_lapack_double.h"

/**
 * Compute a minimum-norm solution
 *     min || A*X - B ||
 * using the RQ factorization
 *     A = R*Q
 * computed by DGERQF.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. n >= m >= 0.
 * @param[in]     nrhs   The number of columns of B. nrhs >= 0.
 * @param[in]     A      Details of the RQ factorization of the original matrix A
 *                       as returned by DGERQF. Double precision array, dimension
 *                       (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= m.
 * @param[in]     tau    Details of the orthogonal matrix Q. Double precision
 *                       array, dimension (m).
 * @param[in,out] B      On entry, the right hand side vectors for the linear
 *                       system. On exit, the solution vectors X. Each solution
 *                       vector is contained in rows 0:n-1 of a column of B.
 *                       Double precision array, dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    work   Workspace array, dimension (lwork).
 * @param[in]     lwork  The length of the array work. lwork must be at least
 *                       nrhs, and should be at least nrhs*NB.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 */
void dgerqs(
    const INT m,
    const INT n,
    const INT nrhs,
    f64* A,
    const INT lda,
    const f64* tau,
    f64* B,
    const INT ldb,
    f64* work,
    const INT lwork,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0 || m > n) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (lwork < 1 || (lwork < nrhs && m > 0 && n > 0)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DGERQS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0 || m == 0)
        return;

    cblas_dtrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                m, nrhs, ONE, &A[0 + (n - m) * lda], lda, &B[n - m], ldb);

    dlaset("F", n - m, nrhs, ZERO, ZERO, B, ldb);

    dormrq("L", "T", n, nrhs, m, A, lda, tau, B, ldb, work, lwork, info);
}
