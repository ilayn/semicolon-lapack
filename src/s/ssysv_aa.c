/**
 * @file ssysv_aa.c
 * @brief SSYSV_AA computes the solution to system of linear equations A * X = B for SY matrices using Aasen's algorithm.
 */

#include "semicolon_lapack_single.h"

/**
 * SSYSV_AA computes the solution to a real system of linear equations
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
 * @param[in]     uplo
 *                      - `'U'`: Upper triangle of A is stored
 *                      - `'L'`: Lower triangle of A is stored
 * @param[in]     n     The number of linear equations, i.e., the order of
 *                      the matrix A. `n>=0`.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B. `nrhs>=0`.
 * @param[in,out] A     Array of dimension `(lda,n)`.
 *                      On entry, the symmetric matrix A. If `uplo='U'`, the
 *                      leading `n`-by-`n` upper triangular part of A
 *                      contains the upper triangular part of the matrix A,
 *                      and the strictly lower triangular part of A is not
 *                      referenced. If `uplo='L'`, the leading `n`-by-`n`
 *                      lower triangular part of A contains the lower
 *                      triangular part of the matrix A, and the strictly
 *                      upper triangular part of A is not referenced.
 *                      On exit, if `info=0`, the tridiagonal matrix T and
 *                      the multipliers used to obtain the factor U or L
 *                      from the factorization A = U**T*T*U or A = L*T*L**T
 *                      as computed by `ssytrf_aa`.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[out]    ipiv  Array of dimension `n`.
 *                      On exit, it contains the details of the
 *                      interchanges, i.e., the row and column `k` of A were
 *                      interchanged with the row and column `ipiv[k]`.
 * @param[in,out] B     Array of dimension `(ldb,nrhs)`.
 *                      On entry, the `n`-by-`nrhs` right hand side matrix B.
 *                      On exit, if `info=0`, the `n`-by-`nrhs` solution
 *                      matrix X.
 * @param[in]     ldb   The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    work  Array of dimension `max(1,lwork)`.
 *                      On exit, if `info=0`, `work[0]` returns the optimal
 *                      `lwork`.
 * @param[in]     lwork The length of `work`. `lwork>=max(1,2*n,3*n-2)`, and
 *                      for the best performance, `lwork>=n*nb`, where `nb`
 *                      is the optimal block size for `ssytrf_aa`.
 *                      If `lwork=-1`, then a workspace query is assumed; the
 *                      routine only calculates the optimal size of the
 *                      `work` array, returns this value as the first entry
 *                      of the `work` array, and no error message related to
 *                      `lwork` is issued.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an
 *                           illegal value
 *                         - `info>0`: if `info=i`, `D(i,i)` is exactly zero.
 *                           The factorization has been completed, but the
 *                           block diagonal matrix D is exactly singular, so
 *                           the solution could not be computed.
 */
void ssysv_aa(
    const char* uplo,
    const INT n,
    const INT nrhs,
    f32* restrict A,
    const INT lda,
    INT* restrict ipiv,
    f32* restrict B,
    const INT ldb,
    f32* restrict work,
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
        ssytrf_aa(uplo, n, A, lda, ipiv, work, -1, info);
        lwkopt_sytrf = (INT)work[0];
        ssytrs_aa(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, -1, info);
        lwkopt_sytrs = (INT)work[0];
        tmp1 = (lwkmin > lwkopt_sytrf) ? lwkmin : lwkopt_sytrf;
        lwkopt = (tmp1 > lwkopt_sytrs) ? tmp1 : lwkopt_sytrs;
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SSYSV_AA ", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    ssytrf_aa(uplo, n, A, lda, ipiv, work, lwork, info);
    if (*info == 0) {

        ssytrs_aa(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, lwork, info);

    }

    work[0] = (f32)lwkopt;
}
