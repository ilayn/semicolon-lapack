/**
 * @file zgerqs.c
 * @brief ZGERQS computes a minimum-norm solution min || A*X - B || using the
 *        RQ factorization A = R*Q computed by ZGERQF.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include "semicolon_lapack_complex_double.h"

/**
 * Compute a minimum-norm solution
 *     min || A*X - B ||
 * using the RQ factorization
 *     A = R*Q
 * computed by ZGERQF.
 *
 * @param[in]     m      The number of rows of the matrix A. m >= 0.
 * @param[in]     n      The number of columns of the matrix A. n >= m >= 0.
 * @param[in]     nrhs   The number of columns of B. nrhs >= 0.
 * @param[in]     A      Details of the RQ factorization of the original matrix A
 *                       as returned by ZGERQF. Complex*16 array, dimension
 *                       (lda, n).
 * @param[in]     lda    The leading dimension of the array A. lda >= m.
 * @param[in]     tau    Details of the unitary matrix Q. Complex*16
 *                       array, dimension (m).
 * @param[in,out] B      On entry, the right hand side vectors for the linear
 *                       system. On exit, the solution vectors X. Each solution
 *                       vector is contained in rows 0:n-1 of a column of B.
 *                       Complex*16 array, dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    work   Workspace array, dimension (lwork).
 * @param[in]     lwork  The length of the array work. lwork must be at least
 *                       nrhs, and should be at least nrhs*NB.
 * @param[out]    info   = 0: successful exit
 *                       < 0: if info = -i, the i-th argument had an illegal value
 */
void zgerqs(
    const INT m,
    const INT n,
    const INT nrhs,
    c128* A,
    const INT lda,
    const c128* tau,
    c128* B,
    const INT ldb,
    c128* work,
    const INT lwork,
    INT* info)
{
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CONE = CMPLX(1.0, 0.0);

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
        xerbla("ZGERQS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0 || m == 0)
        return;

    cblas_ztrsm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans, CblasNonUnit,
                m, nrhs, &CONE, &A[0 + (n - m) * lda], lda, &B[n - m], ldb);

    zlaset("F", n - m, nrhs, CZERO, CZERO, B, ldb);

    zunmrq("L", "C", n, nrhs, m, A, lda, tau, B, ldb, work, lwork, info);
}
