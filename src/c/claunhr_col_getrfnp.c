/**
 * @file claunhr_col_getrfnp.c
 * @brief CLAUNHR_COL_GETRFNP computes the modified LU factorization without
 *        pivoting of a complex general M-by-N matrix A.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CLAUNHR_COL_GETRFNP computes the modified LU factorization without
 * pivoting of a complex general M-by-N matrix A. The factorization has
 * the form:
 *
 *     A - S = L * U,
 *
 * where:
 *    S is a m-by-n diagonal sign matrix with the diagonal D, so that
 *    D(i) = S(i,i), 1 <= i <= min(M,N). The diagonal D is constructed
 *    as D(i)=-SIGN(A(i,i)), where A(i,i) is the value after performing
 *    i-1 steps of Gaussian elimination. This means that the diagonal
 *    element at each step of "modified" Gaussian elimination is
 *    at least one in absolute value (so that division-by-zero not
 *    not possible during the division by the diagonal element);
 *
 *    L is a M-by-N lower triangular matrix with unit diagonal elements
 *    (lower trapezoidal if M > N);
 *
 *    and U is a M-by-N upper triangular matrix
 *    (upper trapezoidal if M < N).
 *
 * This routine is an auxiliary routine used in the Householder
 * reconstruction routine CUNHR_COL. In CUNHR_COL, this routine is
 * applied to an M-by-N matrix A with orthonormal columns, where each
 * element is bounded by one in absolute value. With the choice of
 * the matrix S above, one can show that the diagonal element at each
 * step of Gaussian elimination is the largest (in absolute value) in
 * the column on or below the diagonal, so that no pivoting is required
 * for numerical stability [1].
 *
 * For more details on the Householder reconstruction algorithm,
 * including the modified LU factorization, see [1].
 *
 * This is the blocked right-looking version of the algorithm,
 * calling Level 3 BLAS to update the submatrix. To factorize a block,
 * this routine calls the recursive routine CLAUNHR_COL_GETRFNP2.
 *
 * [1] "Reconstructing Householder vectors from tall-skinny QR",
 *     G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, H.D. Nguyen,
 *     E. Solomonik, J. Parallel Distrib. Comput.,
 *     vol. 85, pp. 3-31, 2015.
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in,out] A     Single complex array, dimension (lda, n).
 *                      On entry, the M-by-N matrix to be factored.
 *                      On exit, the factors L and U from the factorization
 *                      A-S=L*U; the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    D     Single complex array, dimension min(m, n).
 *                      The diagonal elements of the diagonal M-by-N sign matrix S,
 *                      D(i) = S(i,i), where 0 <= i < min(m, n). The elements can be
 *                      only ( +1.0, 0.0 ) or (-1.0, 0.0 ).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value
 */
void claunhr_col_getrfnp(
    const INT m,
    const INT n,
    c64* restrict A,
    const INT lda,
    c64* restrict D,
    INT* info)
{
    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CLAUNHR_COL_GETRFNP", -(*info));
        return;
    }

    INT minmn = m < n ? m : n;

    if (minmn == 0) {
        return;
    }

    claunhr_col_getrfnp2(m, n, A, lda, D, info);
}
