/**
 * @file dlaorhr_col_getrfnp2.c
 * @brief DLAORHR_COL_GETRFNP2 computes the modified LU factorization without pivoting of a real general M-by-N matrix A.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DLAORHR_COL_GETRFNP2 computes the modified LU factorization without
 * pivoting of a real general M-by-N matrix A. The factorization has
 * the form:
 *
 *     A - S = L * U,
 *
 * where:
 *    S is a m-by-n diagonal sign matrix with the diagonal D, so that
 *    D(i) = S(i,i), 1 <= i <= min(M,N). The diagonal D is constructed
 *    as D(i)=-SIGN(A(i,i)), where A(i,i) is the value after performing
 *    i-1 steps of Gaussian elimination. This means that the diagonal
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
 * reconstruction routine DORHR_COL.
 *
 * This is the recursive version of the LU factorization algorithm.
 *
 * @param[in] m
 *          The number of rows of the matrix A. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          Double precision array, dimension (lda, n).
 *          On entry, the M-by-N matrix to be factored.
 *          On exit, the factors L and U from the factorization
 *          A-S=L*U; the unit diagonal elements of L are not stored.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1, m).
 *
 * @param[out] D
 *          Double precision array, dimension min(m, n).
 *          The diagonal elements of the diagonal M-by-N sign matrix S,
 *          D(i) = S(i,i), where 1 <= i <= min(M,N). The elements can
 *          be only plus or minus one.
 *
 * @param[out] info
 *                         - = 0:  successful exit
 *                         - < 0:  if info = -i, the i-th argument had an illegal value
 */
void dlaorhr_col_getrfnp2(
    const INT m,
    const INT n,
    f64* restrict A,
    const INT lda,
    f64* restrict D,
    INT* info)
{
    f64 sfmin;
    INT i, n1, n2;
    INT iinfo;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("DLAORHR_COL_GETRFNP2", -(*info));
        return;
    }

    if ((m < n ? m : n) == 0)
        return;

    if (m == 1) {

        D[0] = (A[0 + 0 * lda] >= 0.0) ? -1.0 : 1.0;

        A[0 + 0 * lda] = A[0 + 0 * lda] - D[0];

    } else if (n == 1) {

        D[0] = (A[0 + 0 * lda] >= 0.0) ? -1.0 : 1.0;

        A[0 + 0 * lda] = A[0 + 0 * lda] - D[0];

        sfmin = dlamch("S");

        if (fabs(A[0 + 0 * lda]) >= sfmin) {
            cblas_dscal(m - 1, 1.0 / A[0 + 0 * lda], &A[1 + 0 * lda], 1);
        } else {
            for (i = 1; i < m; i++) {
                A[i + 0 * lda] = A[i + 0 * lda] / A[0 + 0 * lda];
            }
        }

    } else {

        n1 = (m < n ? m : n) / 2;
        n2 = n - n1;

        dlaorhr_col_getrfnp2(n1, n1, A, lda, D, &iinfo);

        cblas_dtrsm(CblasColMajor, CblasRight, CblasUpper, CblasNoTrans, CblasNonUnit,
                    m - n1, n1, 1.0, A, lda, &A[n1 + 0 * lda], lda);

        cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                    n1, n2, 1.0, A, lda, &A[0 + n1 * lda], lda);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    m - n1, n2, n1, -1.0, &A[n1 + 0 * lda], lda,
                    &A[0 + n1 * lda], lda, 1.0, &A[n1 + n1 * lda], lda);

        dlaorhr_col_getrfnp2(m - n1, n2, &A[n1 + n1 * lda], lda, &D[n1], &iinfo);

    }
}
