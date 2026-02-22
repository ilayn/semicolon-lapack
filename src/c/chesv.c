/**
 * @file chesv.c
 * @brief CHESV computes the solution to a complex system of linear equations
 *        A * X = B, where A is an N-by-N Hermitian matrix and X and B are
 *        N-by-NRHS matrices.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CHESV computes the solution to a complex system of linear equations
 *    A * X = B,
 * where A is an N-by-N Hermitian matrix and X and B are N-by-NRHS
 * matrices.
 *
 * The diagonal pivoting method is used to factor A as
 *    A = U * D * U**H,  if UPLO = 'U', or
 *    A = L * D * L**H,  if UPLO = 'L',
 * where U (or L) is a product of permutation and unit upper (lower)
 * triangular matrices, and D is Hermitian and block diagonal with
 * 1-by-1 and 2-by-2 diagonal blocks.  The factored form of A is then
 * used to solve the system of equations A * X = B.
 *
 * @param[in]     uplo  = 'U':  Upper triangle of A is stored;
 *                        = 'L':  Lower triangle of A is stored.
 * @param[in]     n     The number of linear equations, i.e., the order
 *                      of the matrix A.  n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number
 *                      of columns of the matrix B.  nrhs >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the Hermitian matrix A.  If uplo = 'U',
 *                      the leading N-by-N upper triangular part of A
 *                      contains the upper triangular part of the matrix A,
 *                      and the strictly lower triangular part of A is not
 *                      referenced.  If uplo = 'L', the leading N-by-N lower
 *                      triangular part of A contains the lower triangular
 *                      part of the matrix A, and the strictly upper
 *                      triangular part of A is not referenced.
 *                      On exit, if info = 0, the block diagonal matrix D
 *                      and the multipliers used to obtain the factor U or L
 *                      from the factorization A = U*D*U**H or A = L*D*L**H
 *                      as computed by CHETRF.
 * @param[in]     lda   The leading dimension of the array A.  lda >= max(1,n).
 * @param[out]    ipiv  Integer array, dimension (n).
 *                      Details of the interchanges and the block structure
 *                      of D, as determined by CHETRF.  If ipiv[k] > 0, then
 *                      rows and columns k and ipiv[k] were interchanged, and
 *                      D(k,k) is a 1-by-1 diagonal block.  If uplo = 'U' and
 *                      ipiv[k] = ipiv[k-1] < 0, then rows and columns k-1
 *                      and -ipiv[k] were interchanged and D(k-1:k,k-1:k) is
 *                      a 2-by-2 diagonal block.  If uplo = 'L' and
 *                      ipiv[k] = ipiv[k+1] < 0, then rows and columns k+1
 *                      and -ipiv[k] were interchanged and D(k:k+1,k:k+1) is
 *                      a 2-by-2 diagonal block.
 * @param[in,out] B     Single complex array, dimension (ldb, nrhs).
 *                      On entry, the N-by-NRHS right hand side matrix B.
 *                      On exit, if info = 0, the N-by-NRHS solution matrix X.
 * @param[in]     ldb   The leading dimension of the array B.  ldb >= max(1,n).
 * @param[out]    work  Single complex array, dimension (max(1,lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal
 *                      lwork.
 * @param[in]     lwork The length of work.  lwork >= 1, and for best
 *                      performance lwork >= max(1,N*NB), where NB is the
 *                      optimal blocksize for CHETRF.
 *                      For lwork < N, TRS will be done with Level BLAS 2.
 *                      For lwork >= N, TRS will be done with Level BLAS 3.
 *                      If lwork = -1, then a workspace query is assumed;
 *                      the routine only calculates the optimal size of the
 *                      work array, returns this value as the first entry
 *                      of the work array, and no error message related to
 *                      lwork is issued by XERBLA.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) is exactly zero.  The
 *                           factorization has been completed, but the block
 *                           diagonal matrix D is exactly singular, so the
 *                           solution could not be computed.
 */
void chesv(const char* uplo, const INT n, const INT nrhs,
           c64* restrict A, const INT lda,
           INT* restrict ipiv,
           c64* restrict B, const INT ldb,
           c64* restrict work, const INT lwork,
           INT* info)
{
    INT lwkopt;
    INT lquery = (lwork == -1);

    /* Test the input parameters. */
    *info = 0;
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
    } else if (lwork < 1 && !lquery) {
        *info = -10;
    }

    if (*info == 0) {
        if (n == 0) {
            lwkopt = 1;
        } else {
            chetrf(uplo, n, A, lda, ipiv, work, -1, info);
            lwkopt = (INT)crealf(work[0]);
        }
        work[0] = (c64)lwkopt;
    }

    if (*info != 0) {
        xerbla("CHESV", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    /* Compute the factorization A = U*D*U**H or A = L*D*L**H. */
    chetrf(uplo, n, A, lda, ipiv, work, lwork, info);
    if (*info == 0) {

        /* Solve the system A*X = B, overwriting B with X. */
        if (lwork < n) {

            /* Solve with TRS ( Use Level BLAS 2) */
            chetrs(uplo, n, nrhs, A, lda, ipiv, B, ldb, info);

        } else {

            /* Solve with TRS2 ( Use Level BLAS 3) */
            chetrs2(uplo, n, nrhs, A, lda, ipiv, B, ldb, work, info);

        }

    }

    work[0] = (c64)lwkopt;
}
