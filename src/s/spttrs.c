/**
 * @file spttrs.c
 * @brief SPTTRS solves a tridiagonal system of the form A*X = B using the
 *        L*D*L**T factorization of A computed by SPTTRF.
 */

#include "semicolon_lapack_single.h"

/**
 * SPTTRS solves a tridiagonal system of the form
 *    A * X = B
 * using the L*D*L**T factorization of A computed by SPTTRF.  D is a
 * diagonal matrix specified in the vector D, L is a unit bidiagonal
 * matrix whose subdiagonal is specified in the vector E, and X and B
 * are N by NRHS matrices.
 *
 * @param[in]     n     The order of the tridiagonal matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number
 *                      of columns of the matrix B. nrhs >= 0.
 * @param[in]     D     Double precision array, dimension (n).
 *                      The n diagonal elements of the diagonal matrix D
 *                      from the L*D*L**T factorization of A.
 * @param[in]     E     Double precision array, dimension (n-1).
 *                      The (n-1) subdiagonal elements of the unit bidiagonal
 *                      factor L from the L*D*L**T factorization of A.
 *                      E can also be regarded as the superdiagonal of the
 *                      unit bidiagonal factor U from the factorization
 *                      A = U**T*D*U.
 * @param[in,out] B     Double precision array, dimension (ldb, nrhs).
 *                      On entry, the right hand side vectors B for the
 *                      system of linear equations.
 *                      On exit, the solution vectors, X.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void spttrs(
    const INT n,
    const INT nrhs,
    const f32* restrict D,
    const f32* restrict E,
    f32* restrict B,
    const INT ldb,
    INT* info)
{
    INT max_n_1 = (1 > n) ? 1 : n;

    *info = 0;
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (ldb < max_n_1) {
        *info = -6;
    }
    if (*info != 0) {
        xerbla("SPTTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0)
        return;

    /*
     * ILAENV(1, 'SPTTRS', ...) returns NB=1 (no special case in ilaenv.f).
     * Therefore, we call sptts2 directly without blocking.
     */
    sptts2(n, nrhs, D, E, B, ldb);
}
