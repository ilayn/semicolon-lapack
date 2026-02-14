/**
 * @file zlaunhr_col_getrfnp2.c
 * @brief ZLAUNHR_COL_GETRFNP2 computes the modified LU factorization without
 *        pivoting of a complex general M-by-N matrix A.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>

/**
 * ZLAUNHR_COL_GETRFNP2 computes the modified LU factorization without
 * pivoting of a complex general M-by-N matrix A. The factorization has
 * the form:
 *
 *     A - S = L * U,
 *
 * where:
 *    S is a m-by-n diagonal sign matrix with the diagonal D, so that
 *    D(i) = S(i,i), 0 <= i < min(M,N). The diagonal D is constructed
 *    as D(i)=-SIGN(A(i,i)), where A(i,i) is the value after performing
 *    i steps of Gaussian elimination. This means that the diagonal
 *    element at each step of "modified" Gaussian elimination is at
 *    least one in absolute value (so that division-by-zero not
 *    possible during the division by the diagonal element);
 *
 *    L is a M-by-N lower triangular matrix with unit diagonal elements
 *    (lower trapezoidal if M > N);
 *
 *    and U is a M-by-N upper triangular matrix
 *    (upper trapezoidal if M < N).
 *
 * This routine is an auxiliary routine used in the Householder
 * reconstruction routine ZUNHR_COL. In ZUNHR_COL, this routine is
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
 * This is the recursive version of the LU factorization algorithm.
 * Denote A - S by B. The algorithm divides the matrix B into four
 * submatrices:
 *
 *        [  B11 | B12  ]  where B11 is n1 by n1,
 *    B = [ -----|----- ]        B21 is (m-n1) by n1,
 *        [  B21 | B22  ]        B12 is n1 by n2,
 *                               B22 is (m-n1) by n2,
 *                               with n1 = min(m,n)/2, n2 = n-n1.
 *
 *
 * The subroutine calls itself to factor B11, solves for B21,
 * solves for B12, updates B22, then calls itself to factor B22.
 *
 * For more details on the recursive LU algorithm, see [2].
 *
 * ZLAUNHR_COL_GETRFNP2 is called to factorize a block by the blocked
 * routine ZLAUNHR_COL_GETRFNP, which uses blocked code calling
 * Level 3 BLAS to update the submatrix. However, ZLAUNHR_COL_GETRFNP2
 * is self-sufficient and can be used without ZLAUNHR_COL_GETRFNP.
 *
 * [1] "Reconstructing Householder vectors from tall-skinny QR",
 *     G. Ballard, J. Demmel, L. Grigori, M. Jacquelin, H.D. Nguyen,
 *     E. Solomonik, J. Parallel Distrib. Comput.,
 *     vol. 85, pp. 3-31, 2015.
 *
 * [2] "Recursion leads to automatic variable blocking for dense linear
 *     algebra algorithms", F. Gustavson, IBM J. of Res. and Dev.,
 *     vol. 41, no. 6, pp. 737-755, 1997.
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in,out] A     Complex array, dimension (lda, n).
 *                      On entry, the M-by-N matrix to be factored.
 *                      On exit, the factors L and U from the factorization
 *                      A-S=L*U; the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, m).
 * @param[out]    D     Complex array, dimension min(m, n).
 *                      The diagonal elements of the diagonal M-by-N sign matrix S,
 *                      D(i) = S(i,i), where 0 <= i < min(M,N). The elements can be
 *                      only ( +1.0, 0.0 ) or (-1.0, 0.0 ).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void zlaunhr_col_getrfnp2(const int m, const int n,
                           c128* restrict A, const int lda,
                           c128* restrict D,
                           int* info)
{
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);

    f64 sfmin;
    int i, iinfo, n1, n2;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZLAUNHR_COL_GETRFNP2", -(*info));
        return;
    }

    if (m < n ? m == 0 : n == 0) {
        return;
    }

    if (m == 1) {

        D[0] = CMPLX(-copysign(ONE, creal(A[0])), 0.0);

        A[0] = A[0] - D[0];

    } else if (n == 1) {

        D[0] = CMPLX(-copysign(ONE, creal(A[0])), 0.0);

        A[0] = A[0] - D[0];

        sfmin = dlamch("S");

        if (cabs1(A[0]) >= sfmin) {
            c128 scale = CONE / A[0];
            cblas_zscal(m - 1, &scale, &A[1], 1);
        } else {
            for (i = 1; i < m; i++) {
                A[i] = A[i] / A[0];
            }
        }

    } else {

        n1 = (m < n ? m : n) / 2;
        n2 = n - n1;

        zlaunhr_col_getrfnp2(n1, n1, A, lda, D, &iinfo);

        cblas_ztrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    m - n1, n1, &CONE, A, lda,
                    &A[n1], lda);

        cblas_ztrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n1, n2, &CONE, A, lda,
                    &A[n1 * lda], lda);

        c128 neg_cone = CMPLX(-1.0, 0.0);
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - n1, n2, n1, &neg_cone, &A[n1], lda,
                    &A[n1 * lda], lda, &CONE, &A[n1 + n1 * lda], lda);

        zlaunhr_col_getrfnp2(m - n1, n2, &A[n1 + n1 * lda], lda,
                              &D[n1], &iinfo);

    }
}
