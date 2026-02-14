/**
 * @file zhesv_aa.c
 * @brief ZHESV_AA computes the solution to system of linear equations A * X = B for HE matrices using Aasen's algorithm.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZHESV_AA computes the solution to a complex system of linear equations
 *    A * X = B,
 * where A is an N-by-N Hermitian matrix and X and B are N-by-NRHS
 * matrices.
 *
 * Aasen's algorithm is used to factor A as
 *    A = U**H * T * U,  if UPLO = 'U', or
 *    A = L * T * L**H,  if UPLO = 'L',
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and T is Hermitian and tridiagonal. The factored form
 * of A is then used to solve the system of equations A * X = B.
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
 *          Double complex array, dimension (lda, n).
 *          On entry, the Hermitian matrix A. If uplo = 'U', the leading
 *          N-by-N upper triangular part of A contains the upper
 *          triangular part of the matrix A, and the strictly lower
 *          triangular part of A is not referenced. If uplo = 'L', the
 *          leading N-by-N lower triangular part of A contains the lower
 *          triangular part of the matrix A, and the strictly upper
 *          triangular part of A is not referenced.
 *
 *          On exit, if info = 0, the tridiagonal matrix T and the
 *          multipliers used to obtain the factor U or L from the
 *          factorization A = U**H*T*U or A = L*T*L**H as computed by
 *          ZHETRF_AA.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          On exit, it contains the details of the interchanges, i.e.,
 *          the row and column k of A were interchanged with the
 *          row and column ipiv[k].
 *
 * @param[in,out] B
 *          Double complex array, dimension (ldb, nrhs).
 *          On entry, the N-by-NRHS right hand side matrix B.
 *          On exit, if info = 0, the N-by-NRHS solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] work
 *          Double complex array, dimension (max(1, lwork)).
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
void zhesv_aa(
    const char* uplo,
    const int n,
    const int nrhs,
    double complex* const restrict A,
    const int lda,
    int* restrict ipiv,
    double complex* const restrict B,
    const int ldb,
    double complex* restrict work,
    const int lwork,
    int* info)
{
    int lquery;
    int lwkmin, lwkopt, lwkopt_hetrf, lwkopt_hetrs;
    int tmp1;

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
        zhetrf_aa(uplo, n, A, lda, ipiv, work, -1, info);
        lwkopt_hetrf = (int)creal(work[0]);
        zhetrs_aa(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, -1, info);
        lwkopt_hetrs = (int)creal(work[0]);
        tmp1 = (lwkmin > lwkopt_hetrf) ? lwkmin : lwkopt_hetrf;
        lwkopt = (tmp1 > lwkopt_hetrs) ? tmp1 : lwkopt_hetrs;
        work[0] = CMPLX((double)lwkopt, 0.0);
    }

    if (*info != 0) {
        xerbla("ZHESV_AA ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    zhetrf_aa(uplo, n, A, lda, ipiv, work, lwork, info);
    if (*info == 0) {

        zhetrs_aa(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);

    }

    work[0] = CMPLX((double)lwkopt, 0.0);
}
