/**
 * @file slaorhr_col_getrfnp.c
 * @brief SLAORHR_COL_GETRFNP computes the modified LU factorization without pivoting of a real general M-by-N matrix A.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SLAORHR_COL_GETRFNP computes the modified LU factorization without
 * pivoting of a real general M-by-N matrix A. The factorization has
 * the form:
 *
 *     A - S = L * U,
 *
 * where:
 *    S is a m-by-n diagonal sign matrix with the diagonal D, so that
 *    D(i) = S(i,i), 1 <= i <= min(M,N). The diagonal D is constructed
 *    as D(i)=-SIGN(A(i,i)), where A(i,i) is the value after performing
 *    i-1 steps of Gaussian elimination.
 *
 *    L is a M-by-N lower triangular matrix with unit diagonal elements
 *    (lower trapezoidal if M > N);
 *
 *    and U is a M-by-N upper triangular matrix
 *    (upper trapezoidal if M < N).
 *
 * This routine is an auxiliary routine used in the Householder
 * reconstruction routine SORHR_COL.
 *
 * This is the blocked right-looking version of the algorithm,
 * calling Level 3 BLAS to update the submatrix.
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
void slaorhr_col_getrfnp(
    const INT m,
    const INT n,
    f32* restrict A,
    const INT lda,
    f32* restrict D,
    INT* info)
{
    INT iinfo, j, jb, nb;
    INT minmn;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (1 > m ? 1 : m)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("SLAORHR_COL_GETRFNP", -(*info));
        return;
    }

    minmn = (m < n) ? m : n;
    if (minmn == 0)
        return;

    nb = 32;

    if (nb <= 1 || nb >= minmn) {

        slaorhr_col_getrfnp2(m, n, A, lda, D, info);
    } else {

        for (j = 0; j < minmn; j += nb) {
            jb = ((minmn - j) < nb) ? (minmn - j) : nb;

            slaorhr_col_getrfnp2(m - j, jb, &A[j + j * lda], lda, &D[j], &iinfo);

            if (j + jb < n) {

                cblas_strsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                            jb, n - j - jb, 1.0f, &A[j + j * lda], lda, &A[j + (j + jb) * lda], lda);

                if (j + jb < m) {

                    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m - j - jb, n - j - jb, jb, -1.0f, &A[(j + jb) + j * lda], lda,
                                &A[j + (j + jb) * lda], lda, 1.0f, &A[(j + jb) + (j + jb) * lda], lda);
                }
            }
        }
    }
}
