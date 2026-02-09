/**
 * @file ssysv_aa_2stage.c
 * @brief SSYSV_AA_2STAGE computes the solution to system of linear equations A * X = B for SY matrices using Aasen's 2-stage algorithm.
 */

#include "semicolon_lapack_single.h"

/**
 * SSYSV_AA_2STAGE computes the solution to a real system of
 * linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
 * matrices.
 *
 * Aasen's 2-stage algorithm is used to factor A as
 *    A = U**T * T * U,  if UPLO = 'U', or
 *    A = L * T * L**T,  if UPLO = 'L',
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and T is symmetric and band. The matrix T is
 * then LU-factored with partial pivoting. The factored form of A
 * is then used to solve the system of equations A * X = B.
 *
 * This is the blocked version of the algorithm, calling Level 3 BLAS.
 *
 * @param[in] uplo
 *          = 'U':  Upper triangle of A is stored;
 *          = 'L':  Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in] nrhs
 *          The number of right hand sides, i.e., the number of columns
 *          of the matrix B. nrhs >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the symmetric matrix A.
 *          On exit, L is stored below (or above) the subdiagonal blocks.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] TB
 *          Double precision array, dimension (max(1, ltb)).
 *          On exit, details of the LU factorization of the band matrix.
 *
 * @param[in] ltb
 *          The size of the array TB. ltb >= max(1, 4*n).
 *          If ltb = -1, then a workspace query is assumed.
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          On exit, details of the interchanges.
 *
 * @param[out] ipiv2
 *          Integer array, dimension (n).
 *          On exit, details of the interchanges in T.
 *
 * @param[in,out] B
 *          Double precision array, dimension (ldb, nrhs).
 *          On entry, the right hand side matrix B.
 *          On exit, the solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1, n).
 *
 * @param[out] work
 *          Double precision workspace of size (max(1, lwork)).
 *          On exit, if info = 0, work[0] returns the optimal lwork.
 *
 * @param[in] lwork
 *          The size of work. lwork >= max(1, n).
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *          = 0:  successful exit
 *          < 0:  if info = -i, the i-th argument had an illegal value.
 *          > 0:  if info = i, band LU factorization failed on i-th column
 */
void ssysv_aa_2stage(
    const char* uplo,
    const int n,
    const int nrhs,
    float* const restrict A,
    const int lda,
    float* restrict TB,
    const int ltb,
    int* restrict ipiv,
    int* restrict ipiv2,
    float* const restrict B,
    const int ldb,
    float* restrict work,
    const int lwork,
    int* info)
{
    int upper, tquery, wquery;
    int lwkmin, lwkopt;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    wquery = (lwork == -1);
    tquery = (ltb == -1);
    lwkmin = (1 > n) ? 1 : n;

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (1 > n ? 1 : n)) {
        *info = -5;
    } else if (ltb < (1 > 4 * n ? 1 : 4 * n) && !tquery) {
        *info = -7;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -11;
    } else if (lwork < lwkmin && !wquery) {
        *info = -13;
    }

    if (*info == 0) {
        ssytrf_aa_2stage(uplo, n, A, lda, TB, -1, ipiv, ipiv2, work, -1, info);
        lwkopt = (lwkmin > (int)work[0]) ? lwkmin : (int)work[0];
        work[0] = (float)lwkopt;
    }

    if (*info != 0) {
        xerbla("SSYSV_AA_2STAGE", -(*info));
        return;
    } else if (wquery || tquery) {
        return;
    }

    ssytrf_aa_2stage(uplo, n, A, lda, TB, ltb, ipiv, ipiv2, work, lwork, info);
    if (*info == 0) {

        ssytrs_aa_2stage(uplo, n, nrhs, A, lda, TB, ltb, ipiv, ipiv2, B, ldb, info);

    }

    work[0] = (float)lwkopt;
}
