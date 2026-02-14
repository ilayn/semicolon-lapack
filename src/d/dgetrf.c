/**
 * @file dgetrf.c
 * @brief Blocked LU factorization using Level 3 BLAS.
 */

#include <cblas.h>
#include "../include/lapack_tuning.h"
#include "semicolon_lapack_double.h"

/**
 * Computes an LU factorization of a general M-by-N matrix A using partial
 * pivoting with row interchanges.
 *
 * The factorization has the form:
 *
 *     A = P * L * U
 *
 * where P is a permutation matrix, L is lower triangular with unit diagonal
 * elements, and U is upper triangular.
 *
 * This is the right-looking Level 3 BLAS version of the algorithm.
 *
 * @param[in]     m     The number of rows of the matrix A (m >= 0).
 * @param[in]     n     The number of columns of the matrix A (n >= 0).
 * @param[in,out] A     On entry, the M-by-N matrix to be factored.
 *                      On exit, the factors L and U from the factorization;
 *                      the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array A (lda >= max(1,m)).
 * @param[out]    ipiv  The pivot indices; row i was interchanged with row
 *                      ipiv[i]. Array of dimension min(m,n), 0-based.
 *
 * @param[out]   info
 *                           Exit status.
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but U is exactly singular, and division
 *                           by zero will occur if it is used to solve a system of equations.
 */
void dgetrf(
    const int m,
    const int n,
    f64 * const restrict A,
    const int lda,
    int * const restrict ipiv,
    int *info)
{
    const f64 ONE = 1.0;
    const f64 NEG_ONE = -1.0;

    int i, iinfo, j, jb, nb;
    int minmn = m < n ? m : n;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("DGETRF", -(*info));
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    nb = lapack_get_nb("GETRF");

    if (nb <= 1 || nb >= minmn) {
        dgetrf2(m, n, A, lda, ipiv, info);
    } else {
        for (j = 0; j < minmn; j += nb) {
            jb = minmn - j < nb ? minmn - j : nb;

            dgetrf2(m - j, jb, A + j + j * lda, lda, &ipiv[j], &iinfo);

            if (*info == 0 && iinfo > 0) {
                *info = iinfo + j;
            }
            for (i = j; i < (m < j + jb ? m : j + jb); i++) {
                ipiv[i] = j + ipiv[i];
            }

            if (j > 0) {
                dlaswp(j, A, lda, j, j + jb - 1, ipiv, 1);
            }

            if (j + jb < n) {
                dlaswp(n - j - jb, A + (j + jb) * lda, lda, j, j + jb - 1, ipiv, 1);

                cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit,
                            jb, n - j - jb, ONE, A + j + j * lda, lda,
                            A + j + (j + jb) * lda, lda);

                if (j + jb < m) {
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                m - j - jb, n - j - jb, jb,
                                NEG_ONE, A + (j + jb) + j * lda, lda,
                                A + j + (j + jb) * lda, lda,
                                ONE, A + (j + jb) + (j + jb) * lda, lda);
                }
            }
        }
    }
}
