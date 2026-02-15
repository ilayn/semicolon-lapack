/**
 * @file zgetf2.c
 * @brief ZGETF2 computes the LU factorization of a general m-by-n matrix
 *        using partial pivoting with row interchanges (unblocked algorithm).
 *
 * Deviation from reference LAPACK: pivot search, row swap, and column
 * scaling use explicit loops instead of BLAS calls (izamax, zswap, zscal)
 * to avoid function call overhead on small per-column operations.
 * Inspired by faer (https://codeberg.org/sarah-quinones/faer).
 */

#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGETF2 computes an LU factorization of a general M-by-N matrix A
 * using partial pivoting with row interchanges.
 *
 * The factorization has the form
 *    A = P * L * U
 * where P is a permutation matrix, L is lower triangular with unit
 * diagonal elements (lower trapezoidal if m > n), and U is upper
 * triangular (upper trapezoidal if m < n).
 *
 * This is the right-looking Level 2 BLAS version of the algorithm.
 *
 * @param[in]     m     The number of rows of the matrix A. m >= 0.
 * @param[in]     n     The number of columns of the matrix A. n >= 0.
 * @param[in,out] A     Double complex array, dimension (lda, n).
 *                      On entry, the m by n matrix to be factored.
 *                      On exit, the factors L and U from the factorization
 *                      A = P*L*U; the unit diagonal elements of L are not stored.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1,m).
 * @param[out]    ipiv  Integer array, dimension (min(m,n)).
 *                      The pivot indices; for 0 <= i < min(m,n), row i of the
 *                      matrix was interchanged with row ipiv[i]. 0-based.
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value
 *                      > 0: if info = i, U(i-1,i-1) is exactly zero. The factorization
 *                           has been completed, but the factor U is exactly
 *                           singular, and division by zero will occur if it is used
 *                           to solve a system of equations.
 */
void zgetf2(
    const int m,
    const int n,
    c128* restrict A,
    const int lda,
    int* restrict ipiv,
    int* info)
{
    const c128 ZERO = CMPLX(0.0, 0.0);
    const f64 sfmin = dlamch("S");

    int i, j, jp;
    int minmn = m < n ? m : n;
    c128 pivot, inv;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("ZGETF2", -(*info));
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    for (j = 0; j < minmn; j++) {

        jp = j + cblas_izamax(m - j, &A[j + j * lda], 1);
        ipiv[j] = jp;

        if (A[jp + j * lda] != ZERO) {

            if (jp != j) {
                cblas_zswap(n, &A[j], lda, &A[jp], lda);
            }

            if (j < m - 1) {
                pivot = A[j + j * lda];
                if (cabs(pivot) >= sfmin) {
                    inv = CMPLX(1.0, 0.0) / pivot;
                    cblas_zscal(m - j - 1, &inv, &A[j + 1 + j * lda], 1);
                } else {
                    for (i = j + 1; i < m; i++) {
                        A[i + j * lda] /= pivot;
                    }
                }
            }

        } else if (*info == 0) {
            *info = j + 1;
        }

        if (j < minmn - 1) {
            const c128 NEG_ONE = CMPLX(-1.0, 0.0);
            cblas_zgeru(CblasColMajor, m - j - 1, n - j - 1, &NEG_ONE,
                        &A[j + 1 + j * lda], 1,
                        &A[j + (j + 1) * lda], lda,
                        &A[j + 1 + (j + 1) * lda], lda);
        }
    }
}
