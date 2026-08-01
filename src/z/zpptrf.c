/**
 * @file zpptrf.c
 * @brief ZPPTRF computes the Cholesky factorization of a packed Hermitian positive definite matrix.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZPPTRF computes the Cholesky factorization of a complex Hermitian
 * positive definite matrix A stored in packed format.
 *
 * The factorization has the form
 * @rst
 * .. code-block:: text
 *
 *     A = U**H * U,  if uplo = 'U', or
 *     A = L  * L**H,  if uplo = 'L',
 * @endrst
 * where U is an upper triangular matrix and L is lower triangular.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in,out] AP     Array of dimension `n*(n+1)/2`.
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array. The
 *                       j-th column of A is stored in the array AP as follows:
 *                       if `uplo='U'`, `AP[i + j*(j+1)/2] = A(i,j)` for `0<=i<=j`;
 *                       if `uplo='L'`, `AP[i + j*(2*n-j-1)/2] = A(i,j)` for
 *                       `j<=i<=n-1`.
 *                       See below for further details.
 *                       On exit, if `info=0`, the triangular factor U or L from
 *                       the Cholesky factorization A = U**H*U or A = L*L**H,
 *                       in the same storage format as A.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, the leading principal minor of order
 *                           i is not positive, and the factorization could not be
 *                           completed.
 *
 * @par Further Details:
 * @rst
 * The packed storage scheme is illustrated by the following example
 * when ``n=4``, ``uplo='U'``:
 *
 * Two-dimensional storage of the Hermitian matrix A:
 *
 * .. code-block:: text
 *
 *     a00 a01 a02 a03
 *         a11 a12 a13
 *             a22 a23     (aij = conjg(aji))
 *                 a33
 *
 * Packed storage of the upper triangle of A:
 *
 * .. code-block:: text
 *
 *     AP = [ a00, a01, a11, a02, a12, a22, a03, a13, a23, a33 ]
 * @endrst
 */
void zpptrf(
    const char* uplo,
    const INT n,
    c128* restrict AP,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    }
    if (*info != 0) {
        xerbla("ZPPTRF", -(*info));
        return;
    }

    if (n == 0) {
        return;
    }

    if (upper) {
        // Compute the Cholesky factorization A = U**H * U.
        INT jj = -1;
        for (INT j = 0; j < n; j++) {
            INT jc = jj + 1;
            jj = jj + (j + 1);

            // Compute elements 1:J-1 of column J.
            if (j > 0) {
                cblas_ztpsv(CblasColMajor, CblasUpper, CblasConjTrans, CblasNonUnit,
                            j, AP, &AP[jc], 1);
            }

            // Compute U(J,J) and test for non-positive-definiteness.
            f64 ajj = creal(AP[jj]);
            if (j > 0) {
                c128 dotc;
                cblas_zdotc_sub(j, &AP[jc], 1, &AP[jc], 1, &dotc);
                ajj -= creal(dotc);
            }
            if (ajj <= ZERO) {
                AP[jj] = ajj;
                *info = j + 1;
                return;
            }
            AP[jj] = sqrt(ajj);
        }
    } else {
        // Compute the Cholesky factorization A = L * L**H.
        INT jj = 0;
        for (INT j = 0; j < n; j++) {

            // Compute L(J,J) and test for non-positive-definiteness.
            f64 ajj = creal(AP[jj]);
            if (ajj <= ZERO) {
                AP[jj] = ajj;
                *info = j + 1;
                return;
            }
            ajj = sqrt(ajj);
            AP[jj] = ajj;

            // Compute elements J+1:N of column J and update
            // the trailing submatrix.
            if (j < n - 1) {
                cblas_zdscal(n - j - 1, ONE / ajj, &AP[jj + 1], 1);
                cblas_zhpr(CblasColMajor, CblasLower, n - j - 1, -ONE,
                           &AP[jj + 1], 1, &AP[jj + n - j]);
                jj = jj + n - j;
            }
        }
    }
}
