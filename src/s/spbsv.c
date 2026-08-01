/**
 * @file spbsv.c
 * @brief SPBSV computes the solution to a symmetric positive definite banded system A * X = B.
 */

#include "semicolon_lapack_single.h"

/**
 * SPBSV computes the solution to a real system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where A is an `n`-by-`n` symmetric positive definite band matrix and X
 * and `B` are `n`-by-`nrhs` matrices.
 *
 * The Cholesky decomposition is used to factor A as
 * @rst
 * .. code-block:: text
 *
 *     A = U**T * U,  if uplo = 'U', or
 *     A = L * L**T,  if uplo = 'L',
 * @endrst
 * where U is an upper triangular band matrix, and L is a lower
 * triangular band matrix, with the same number of superdiagonals or
 * subdiagonals as A. The factored form of A is then used to solve the
 * system of equations A * X = B.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The number of linear equations, i.e., the order of
 *                       the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in,out] AB     Array of dimension (`ldab`, `n`).
 *                       On entry, the upper or lower triangle of the symmetric
 *                       band matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of A is stored in the j-th
 *                       column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = A(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = A(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 *                       See below for further details.
 *                       On exit, if `info=0`, the triangular factor U or L
 *                       from the Cholesky factorization A = U**T*U or
 *                       A = L*L**T of the band matrix A, in the same storage
 *                       format as A.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[in,out] B      Array of dimension (`ldb`, `nrhs`).
 *                       On entry, the `n`-by-`nrhs` right hand side matrix `B`.
 *                       On exit, if `info=0`, the `n`-by-`nrhs` solution matrix X.
 * @param[in]     ldb    The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, the leading principal minor of order
 *                           i of A is not positive, so the factorization could not
 *                           be completed, and the solution has not been computed.
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
void spbsv(
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    f32* restrict AB,
    const INT ldab,
    f32* restrict B,
    const INT ldb,
    INT* info)
{
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldab < kd + 1) {
        *info = -6;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    }
    if (*info != 0) {
        xerbla("SPBSV ", -(*info));
        return;
    }

    // Compute the Cholesky factorization A = U**T*U or A = L*L**T
    spbtrf(uplo, n, kd, AB, ldab, info);
    if (*info == 0) {
        // Solve the system A*X = B
        spbtrs(uplo, n, kd, nrhs, AB, ldab, B, ldb, info);
    }
}
