/**
 * @file cgetf2.c
 * @brief CGETF2 computes the LU factorization of a general m-by-n matrix
 *        using partial pivoting with row interchanges (unblocked algorithm).
 *
 * Deviation from reference LAPACK: pivot search, row swap, and column
 * scaling use explicit loops instead of BLAS calls (icamax, cswap, cscal)
 * to avoid function call overhead on small per-column operations.
 * Inspired by faer (https://codeberg.org/sarah-quinones/faer).
 */

#include <complex.h>
#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CGETF2 computes an LU factorization of a general M-by-N matrix A
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
 * @param[in,out] A     Single complex array, dimension (lda, n).
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
void cgetf2(
    const INT m,
    const INT n,
    c64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    INT* info)
{
    const c64 ZERO = CMPLXF(0.0f, 0.0f);
    const f32 sfmin = slamch("S");

    INT i, j, jp;
    INT minmn = m < n ? m : n;
    c64 pivot, inv;

    *info = 0;
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < (m > 1 ? m : 1)) {
        *info = -4;
    }
    if (*info != 0) {
        xerbla("CGETF2", -(*info));
        return;
    }

    if (m == 0 || n == 0) {
        return;
    }

    for (j = 0; j < minmn; j++) {

        jp = j + cblas_icamax(m - j, &A[j + j * lda], 1);
        ipiv[j] = jp;

        if (A[jp + j * lda] != ZERO) {

            if (jp != j) {
                cblas_cswap(n, &A[j], lda, &A[jp], lda);
            }

            if (j < m - 1) {
                pivot = A[j + j * lda];
                if (cabsf(pivot) >= sfmin) {
                    inv = CMPLXF(1.0f, 0.0f) / pivot;
                    cblas_cscal(m - j - 1, &inv, &A[j + 1 + j * lda], 1);
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
            const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);
            cblas_cgeru(CblasColMajor, m - j - 1, n - j - 1, &NEG_ONE,
                        &A[j + 1 + j * lda], 1,
                        &A[j + (j + 1) * lda], lda,
                        &A[j + 1 + (j + 1) * lda], lda);
        }
    }
}
