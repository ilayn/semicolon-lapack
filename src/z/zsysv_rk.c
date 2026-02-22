/**
 * @file zsysv_rk.c
 * @brief ZSYSV_RK computes the solution to system of linear equations A * X = B for SY matrices using ZSYTRF_RK/ZSYTRS_3.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>

/**
 * ZSYSV_RK computes the solution to a complex system of linear
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
 * ZSYTRF_RK is called to compute the factorization of a complex
 * symmetric matrix. The factored form of A is then used to solve
 * the system of equations A * X = B by calling BLAS3 routine ZSYTRS_3.
 *
 * @param[in] uplo
 *          Specifies whether the upper or lower triangular part of the
 *          symmetric matrix A is stored:
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
 *          On entry, the symmetric matrix A.
 *          On exit, if info = 0, diagonal of the block diagonal
 *          matrix D and factors U or L as computed by ZSYTRF_RK.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, n).
 *
 * @param[out] E
 *          Double complex array, dimension (n).
 *          On exit, contains the superdiagonal (or subdiagonal)
 *          elements of the symmetric block diagonal matrix D.
 *
 * @param[out] ipiv
 *          Integer array, dimension (n).
 *          Details of the interchanges and the block structure of D.
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
 *          The length of work. lwork >= 1.
 *          If lwork = -1, then a workspace query is assumed.
 *
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0: if info = k, the matrix A is singular.
 */
void zsysv_rk(
    const char* uplo,
    const INT n,
    const INT nrhs,
    c128* restrict A,
    const INT lda,
    c128* restrict E,
    INT* restrict ipiv,
    c128* restrict B,
    const INT ldb,
    c128* restrict work,
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
            zsytrf_rk(uplo, n, A, lda, E, ipiv, work, -1, info);
            lwkopt = (INT)creal(work[0]);
        }
        work[0] = (c128)lwkopt;
    }

    if (*info != 0) {
        xerbla("ZSYSV_RK", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    zsytrf_rk(uplo, n, A, lda, E, ipiv, work, lwork, info);

    if (*info == 0) {

        zsytrs_3(uplo, n, nrhs, A, lda, E, ipiv, B, ldb, info);

    }

    work[0] = (c128)lwkopt;
}
