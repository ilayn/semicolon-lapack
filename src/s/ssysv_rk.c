/**
 * @file ssysv_rk.c
 * @brief SSYSV_RK computes the solution to system of linear equations A * X = B for SY matrices using SSYTRF_RK/SSYTRS_3.
 */

#include "semicolon_lapack_single.h"

/**
 * SSYSV_RK computes the solution to a real system of linear
 * equations A * X = B, where A is an N-by-N symmetric matrix
 * and X and B are N-by-NRHS matrices.
 *
 * The bounded Bunch-Kaufman (rook) diagonal pivoting method is used
 * to factor A as
 *    A = P*U*D*(U**T)*(P**T),  if UPLO = 'U', or
 *    A = P*L*D*(L**T)*(P**T),  if UPLO = 'L',
 * where U (or L) is unit upper (or lower) triangular matrix,
 * U**T (or L**T) is the transpose of U (or L), P is a permutation
 * matrix, P**T is the transpose of P, and D is symmetric and block
 * diagonal with 1-by-1 and 2-by-2 diagonal blocks.
 *
 * `ssytrf_rk` is called to compute the factorization of a real
 * symmetric matrix. The factored form of A is then used to solve
 * the system of equations A * X = B by calling BLAS3 routine `ssytrs_3`.
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
 *                      On exit, if `info=0`, diagonal of the block diagonal
 *                      matrix D and factors U or L as computed by
 *                      `ssytrf_rk`:
 *
 *                      - Only diagonal elements of the symmetric block
 *                        diagonal matrix D on the diagonal of A, i.e.
 *                        `D(k,k)=A(k,k)`; (superdiagonal (or subdiagonal)
 *                        elements of D are stored on exit in array `E`), and
 *                      - If `uplo='U'`: factor U in the superdiagonal part
 *                        of A. If `uplo='L'`: factor L in the subdiagonal
 *                        part of A.
 *
 *                      For more info see the description of `ssytrf_rk`.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[out]    E     Array of dimension `n`.
 *                      On exit, contains the output computed by the
 *                      factorization routine `ssytrf_rk`, i.e. the
 *                      superdiagonal (or subdiagonal) elements of the
 *                      symmetric block diagonal matrix D with 1-by-1 or
 *                      2-by-2 diagonal blocks, where if `uplo='U'`:
 *                      `E[i]=D(i-1,i)`, `i=1:n-1`, `E[0]` is set to 0; if
 *                      `uplo='L'`: `E[i]=D(i+1,i)`, `i=0:n-2`, `E[n-1]` is
 *                      set to 0.
 *
 *                      For a 1-by-1 diagonal block `D(k)`, the element
 *                      `E[k]` is set to 0 in both `uplo='U'` or `uplo='L'`
 *                      cases.
 *
 *                      For more info see the description of `ssytrf_rk`.
 * @param[out]    ipiv  Array of dimension `n`.
 *                      Details of the interchanges and the block structure
 *                      of D, as determined by `ssytrf_rk`.
 *
 *                      For more info see the description of `ssytrf_rk`.
 * @param[in,out] B     Array of dimension `(ldb,nrhs)`.
 *                      On entry, the `n`-by-`nrhs` right hand side matrix B.
 *                      On exit, if `info=0`, the `n`-by-`nrhs` solution
 *                      matrix X.
 * @param[in]     ldb   The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    work  Array of dimension `max(1,lwork)`.
 *                      Work array used in the factorization stage.
 *                      On exit, if `info=0`, `work[0]` returns the optimal
 *                      `lwork`.
 * @param[in]     lwork The length of `work`. `lwork>=1`. For best
 *                      performance of the factorization stage
 *                      `lwork>=n*nb`, where `nb` is the optimal block size
 *                      for `ssytrf_rk`.
 *                      If `lwork=-1`, then a workspace query is assumed;
 *                      the routine only calculates the optimal size of the
 *                      `work` array for the factorization stage, returns
 *                      this value as the first entry of the `work` array,
 *                      and no error message related to `lwork` is issued.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-k`, the k-th argument had an
 *                           illegal value
 *                         - `info>0`: if `info=k`, the matrix A is singular,
 *                           because column k in the upper (`uplo='U'`) or
 *                           lower (`uplo='L'`) triangular part of A contains
 *                           all zeros. Therefore `D(k,k)` is exactly zero,
 *                           and superdiagonal elements of column k of U (or
 *                           subdiagonal elements of column k of L) are all
 *                           zeros. The factorization has been completed,
 *                           but the block diagonal matrix D is exactly
 *                           singular, and division by zero will occur if it
 *                           is used to solve a system of equations.
 */
void ssysv_rk(
    const char* uplo,
    const INT n,
    const INT nrhs,
    f32* restrict A,
    const INT lda,
    f32* restrict E,
    INT* restrict ipiv,
    f32* restrict B,
    const INT ldb,
    f32* restrict work,
    const INT lwork,
    INT* info)
{
    INT lquery;
    INT lwkopt;

    *info = 0;
    lquery = (lwork == -1);
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
        *info = -9;
    } else if (lwork < 1 && !lquery) {
        *info = -11;
    }

    if (*info == 0) {
        if (n == 0) {
            lwkopt = 1;
        } else {
            ssytrf_rk(uplo, n, A, lda, E, ipiv, work, -1, info);
            lwkopt = (INT)work[0];
        }
        work[0] = (f32)lwkopt;
    }

    if (*info != 0) {
        xerbla("SSYSV_RK", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    ssytrf_rk(uplo, n, A, lda, E, ipiv, work, lwork, info);

    if (*info == 0) {

        ssytrs_3(uplo, n, nrhs, A, lda, E, ipiv, B, ldb, info);

    }

    work[0] = (f32)lwkopt;
}
