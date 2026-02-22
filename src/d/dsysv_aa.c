/**
 * @file dsysv_aa.c
 * @brief DSYSV_AA computes the solution to system of linear equations A * X = B for SY matrices using Aasen's algorithm.
 */

#include "semicolon_lapack_double.h"

/**
 * DSYSV_AA computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
 * matrices.
 *
 * Aasen's algorithm is used to factor A as
 *    A = U**T * T * U,  if UPLO = 'U', or
 *    A = L * T * L**T,  if UPLO = 'L',
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and T is symmetric tridiagonal. The factored
 * form of A is then used to solve the system of equations A * X = B.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The number of linear equations, i.e., the order of the
 *          matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B. nrhs >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, if info = 0, the tridiagonal matrix T and the
 *          multipliers used to obtain the factor U or L.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          On exit, it contains the details of the interchanges.
 *
 * @param[in,out] B
 *          Double precision array, dimension (ldb, nrhs).
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if info = 0, the N-by-NRHS solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] work
 *          Double precision array, dimension (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The length of work. lwork >= max(1, 2*n, 3*n-2).
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) is exactly zero.
 */
void dsysv_aa(
    const char* uplo,
    const INT n,
    const INT nrhs,
    f64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    f64* restrict B,
    const INT ldb,
    f64* restrict work,
    const INT lwork,
    INT* info)
{
    INT lquery;
    INT lwkmin, lwkopt, lwkopt_sytrf, lwkopt_sytrs;
    INT tmp1;

    *info = 0;
    lquery = (lwork == -1);

    tmp1 = (1 > 2 * n) ? 1 : 2 * n;
    lwkmin = (tmp1 > 3 * n - 2) ? tmp1 : 3 * n - 2;

    if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
        !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (lwork < lwkmin && !lquery) {
        *info = -10;
    }

    if (*info == 0) {
        dsytrf_aa(uplo, n, A, lda, ipiv, work, -1, info);
        lwkopt_sytrf = (INT)work[0];
        dsytrs_aa(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, -1, info);
        lwkopt_sytrs = (INT)work[0];
        tmp1 = (lwkmin > lwkopt_sytrf) ? lwkmin : lwkopt_sytrf;
        lwkopt = (tmp1 > lwkopt_sytrs) ? tmp1 : lwkopt_sytrs;
        work[0] = (f64)lwkopt;
    }

    if (*info != 0) {
        xerbla("DSYSV_AA ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    dsytrf_aa(uplo, n, A, lda, ipiv, work, lwork, info);
    if (*info == 0) {

        dsytrs_aa(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);

    }

    work[0] = (f64)lwkopt;
}
