#include "semicolon_lapack_complex_double.h"
#include <complex.h>
/**
 * @file zgbsv.c
 * @brief Solves a general banded system of linear equations A * X = B.
 */

/**
 * ZGBSV computes the solution to a complex system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where A is a band matrix of order `n` with `kl` subdiagonals and `ku`
 * superdiagonals, and X and `B` are `n`-by-`nrhs` matrices.
 *
 * The LU decomposition with partial pivoting and row interchanges is
 * used to factor A as A = L * U, where L is a product of permutation
 * and unit lower triangular matrices with `kl` subdiagonals, and U is
 * upper triangular with `kl+ku` superdiagonals. The factored form of A
 * is then used to solve the system of equations A * X = B.
 *
 * @param[in]     n       The number of linear equations, i.e., the order of the
 *                        matrix A. `n>=0`.
 * @param[in]     kl      The number of subdiagonals within the band of A. `kl>=0`.
 * @param[in]     ku      The number of superdiagonals within the band of A. `ku>=0`.
 * @param[in]     nrhs    The number of right hand sides, i.e., the number of
 *                        columns of the matrix `B`. `nrhs>=0`.
 * @param[in,out] AB      Array of dimension (`ldab`, `n`).
 *                        On entry, the matrix A in band storage, in rows `kl` to
 *                        `2*kl+ku`; rows 0 to `kl-1` of the array need not be set.
 *                        The j-th column of A is stored in the j-th column of
 *                        the array `AB` as follows:
 *                        `AB[kl+ku+i-j + j*ldab] = A(i,j)` for `max(0,j-ku)<=i<=min(n-1,j+kl)`.
 *                        On exit, details of the factorization: U is stored as an
 *                        upper triangular band matrix with `kl+ku` superdiagonals in
 *                        rows 0 to `kl+ku`, and the multipliers used during the
 *                        factorization are stored in rows `kl+ku+1` to `2*kl+ku`.
 *                        See below for further details.
 * @param[in]     ldab    The leading dimension of the array `AB`. `ldab>=2*kl+ku+1`.
 * @param[out]    ipiv    Array of dimension `n`.
 *                        The pivot indices that define the permutation matrix P;
 *                        row i of the matrix was interchanged with row `ipiv[i]`.
 * @param[in,out] B       Array of dimension (`ldb`, `nrhs`).
 *                        On entry, the `n`-by-`nrhs` right hand side matrix `B`.
 *                        On exit, if `info=0`, the `n`-by-`nrhs` solution matrix X.
 * @param[in]     ldb     The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    info
 *                          - `info=0`: successful exit
 *                          - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                            value
 *                          - `info>0`: if `info=i`, U(i,i) is exactly zero. The
 *                            factorization has been completed, but the factor U is
 *                            exactly singular, and the solution has not been computed.
 *
 * @par Further Details:
 * @rst
 * The band storage scheme is illustrated by the following example, when
 * m = n = 6, kl = 2, ku = 1:
 *
 * .. code-block:: text
 *
 *     On entry:                       On exit:
 *
 *         *    *    *    +    +    +       *    *    *   u03  u14  u25
 *         *    *    +    +    +    +       *    *   u02  u13  u24  u35
 *         *   a01  a12  a23  a34  a45      *   u01  u12  u23  u34  u45
 *        a00  a11  a22  a33  a44  a55     u00  u11  u22  u33  u44  u55
 *        a10  a21  a32  a43  a54   *      m10  m21  m32  m43  m54   *
 *        a20  a31  a42  a53   *    *      m20  m31  m42  m53   *    *
 *
 * Array elements marked * are not used by the routine; elements marked
 * + need not be set on entry, but are required by the routine to store
 * elements of U because of fill-in resulting from the row interchanges.
 * @endrst
 */
void zgbsv(
    const INT n,
    const INT kl,
    const INT ku,
    const INT nrhs,
    c128* restrict AB,
    const INT ldab,
    INT* restrict ipiv,
    c128* restrict B,
    const INT ldb,
    INT* info)
{
    /* Test the input parameters */
    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (kl < 0) {
        *info = -2;
    } else if (ku < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldab < 2 * kl + ku + 1) {
        *info = -6;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -9;
    }

    if (*info != 0) {
        xerbla("ZGBSV ", -(*info));
        return;
    }

    /* Compute the LU factorization of the band matrix A */
    zgbtrf(n, n, kl, ku, AB, ldab, ipiv, info);

    if (*info == 0) {
        /* Solve the system A*X = B, overwriting B with X */
        zgbtrs("N", n, kl, ku, nrhs, AB, ldab, ipiv, B, ldb, info);
    }
}
