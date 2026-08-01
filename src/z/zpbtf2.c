/**
 * @file zpbtf2.c
 * @brief ZPBTF2 computes the Cholesky factorization of a Hermitian positive definite band matrix (unblocked).
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBTF2 computes the Cholesky factorization of a complex Hermitian
 * positive definite band matrix A.
 *
 * The factorization has the form
 * @rst
 * .. code-block:: text
 *
 *     A = U**H * U ,  if uplo = 'U', or
 *     A = L  * L**H,  if uplo = 'L',
 * @endrst
 * where U is an upper triangular matrix, U**H is the conjugate transpose
 * of U, and L is lower triangular.
 *
 * This is the unblocked version of the algorithm, calling Level 2 BLAS.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangular
 *                       - `'L'`: Lower triangular
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in,out] AB     Array of dimension (`ldab`, `n`).
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       band matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of A is stored in the j-th
 *                       column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = A(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = A(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 *                       On exit, if `info=0`, the triangular factor U or L
 *                       from the Cholesky factorization A = U**H*U or
 *                       A = L*L**H of the band matrix A, in the same storage
 *                       format as A.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-k`, the k-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=k`, the leading principal minor of order
 *                           k is not positive, and the factorization could not be
 *                           completed.
 *
 * @par Further Details:
 * @rst
 * The band storage scheme is illustrated by the following example, when
 * ``n=6``, ``kd=2``, and ``uplo='U'``:
 *
 * .. code-block:: text
 *
 *     On entry:                       On exit:
 *
 *          *    *   a02  a13  a24  a35      *    *   u02  u13  u24  u35
 *          *   a01  a12  a23  a34  a45      *   u01  u12  u23  u34  u45
 *         a00  a11  a22  a33  a44  a55     u00  u11  u22  u33  u44  u55
 *
 * Similarly, if ``uplo='L'`` the format of A is as follows:
 *
 * .. code-block:: text
 *
 *     On entry:                       On exit:
 *
 *         a00  a11  a22  a33  a44  a55     l00  l11  l22  l33  l44  l55
 *         a10  a21  a32  a43  a54   *      l10  l21  l32  l43  l54   *
 *         a20  a31  a42  a53   *    *      l20  l31  l42  l53   *    *
 *
 * Array elements marked * are not used by the routine.
 * @endrst
 */
void zpbtf2(
    const char* uplo,
    const INT n,
    const INT kd,
    c128* restrict AB,
    const INT ldab,
    INT* info)
{
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    INT upper;
    INT j, kld, kn;
    f64 ajj;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (ldab < kd + 1) {
        *info = -5;
    }
    if (*info != 0) {
        xerbla("ZPBTF2", -(*info));
        return;
    }

    if (n == 0)
        return;

    kld = (1 > ldab - 1) ? 1 : (ldab - 1);

    if (upper) {
        // Compute the Cholesky factorization A = U**H * U
        for (j = 0; j < n; j++) {
            // Compute U(j,j) and test for non-positive-definiteness
            ajj = creal(AB[kd + j * ldab]);
            if (ajj <= ZERO) {
                AB[kd + j * ldab] = CMPLX(ajj, 0.0);
                *info = j + 1;
                return;
            }
            ajj = sqrt(ajj);
            AB[kd + j * ldab] = CMPLX(ajj, 0.0);

            // Compute elements j+1:j+kn of row j and update trailing submatrix
            kn = (kd < n - j - 1) ? kd : (n - j - 1);
            if (kn > 0) {
                cblas_zdscal(kn, ONE / ajj, &AB[kd - 1 + (j + 1) * ldab], kld);
                zlacgv(kn, &AB[kd - 1 + (j + 1) * ldab], kld);
                cblas_zher(CblasColMajor, CblasUpper, kn, -ONE,
                           &AB[kd - 1 + (j + 1) * ldab], kld,
                           &AB[kd + (j + 1) * ldab], kld);
                zlacgv(kn, &AB[kd - 1 + (j + 1) * ldab], kld);
            }
        }
    } else {
        // Compute the Cholesky factorization A = L * L**H
        for (j = 0; j < n; j++) {
            // Compute L(j,j) and test for non-positive-definiteness
            ajj = creal(AB[0 + j * ldab]);
            if (ajj <= ZERO) {
                AB[0 + j * ldab] = CMPLX(ajj, 0.0);
                *info = j + 1;
                return;
            }
            ajj = sqrt(ajj);
            AB[0 + j * ldab] = CMPLX(ajj, 0.0);

            // Compute elements j+1:j+kn of column j and update trailing submatrix
            kn = (kd < n - j - 1) ? kd : (n - j - 1);
            if (kn > 0) {
                cblas_zdscal(kn, ONE / ajj, &AB[1 + j * ldab], 1);
                cblas_zher(CblasColMajor, CblasLower, kn, -ONE,
                           &AB[1 + j * ldab], 1,
                           &AB[0 + (j + 1) * ldab], kld);
            }
        }
    }
}
