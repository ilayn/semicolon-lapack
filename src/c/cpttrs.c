/**
 * @file cpttrs.c
 * @brief CPTTRS solves a tridiagonal system of the form A*X = B using the
 *        factorization A = U**H*D*U or A = L*D*L**H computed by CPTTRF.
 */

#include "internal_build_defs.h"
#include "semicolon_lapack_complex_single.h"
#include <complex.h>

/**
 * CPTTRS solves a tridiagonal system of the form
 *    A * X = B
 * using the factorization A = U**H *D* U or A = L*D*L**H computed by CPTTRF.
 * D is a diagonal matrix specified in the vector D, U (or L) is a unit
 * bidiagonal matrix whose superdiagonal (subdiagonal) is specified in
 * the vector E, and X and B are N by NRHS matrices.
 *
 * @param[in]     uplo  Specifies the form of the factorization and whether the
 *                      vector E is the superdiagonal of the upper bidiagonal factor
 *                      U or the subdiagonal of the lower bidiagonal factor L.
 *                      = 'U':  A = U**H *D*U, E is the superdiagonal of U
 *                      = 'L':  A = L*D*L**H, E is the subdiagonal of L
 * @param[in]     n     The order of the tridiagonal matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B. nrhs >= 0.
 * @param[in]     D     Single precision array, dimension (n).
 *                      The n diagonal elements of the diagonal matrix D from the
 *                      factorization A = U**H *D*U or A = L*D*L**H.
 * @param[in]     E     Complex*16 array, dimension (n-1).
 *                      If UPLO = 'U', the (n-1) superdiagonal elements of the unit
 *                      bidiagonal factor U from the factorization A = U**H*D*U.
 *                      If UPLO = 'L', the (n-1) subdiagonal elements of the unit
 *                      bidiagonal factor L from the factorization A = L*D*L**H.
 * @param[in,out] B     Complex*16 array, dimension (ldb, nrhs).
 *                      On entry, the right hand side vectors B for the system of
 *                      linear equations.
 *                      On exit, the solution vectors, X.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void cpttrs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f32* restrict D,
    const c64* restrict E,
    c64* restrict B,
    const INT ldb,
    INT* info)
{
    INT max_n_1 = (1 > n) ? 1 : n;

    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < max_n_1) {
        *info = -7;
    }
    if (*info != 0) {
        xerbla("CPTTRS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0)
        return;

    /*
     * ILAENV(1, 'CPTTRS', ...) returns NB=1 (no special case in ilaenv.f).
     * Therefore, we call cptts2 directly without blocking.
     */
    INT iuplo = upper ? 1 : 0;
    cptts2(iuplo, n, nrhs, D, E, B, ldb);
}
