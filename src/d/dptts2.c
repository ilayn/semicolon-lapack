/**
 * @file dptts2.c
 * @brief DPTTS2 solves a tridiagonal system of the form AX=B using the
 *        L*D*L**T factorization computed by DPTTRF.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DPTTS2 solves a tridiagonal system of the form
 *    A * X = B
 * using the L*D*L**T factorization of A computed by DPTTRF.  D is a
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
 */
void dptts2(
    const INT n,
    const INT nrhs,
    const f64* restrict D,
    const f64* restrict E,
    f64* restrict B,
    const INT ldb)
{
    INT i, j;

    if (n <= 1) {
        if (n == 1) {
            cblas_dscal(nrhs, 1.0 / D[0], B, ldb);
        }
        return;
    }

    for (j = 0; j < nrhs; j++) {

        /* Solve L * x = b. */

        for (i = 1; i < n; i++) {
            B[i + j * ldb] = B[i + j * ldb] - B[(i - 1) + j * ldb] * E[i - 1];
        }

        /* Solve D * L**T * x = b. */

        B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
        for (i = n - 2; i >= 0; i--) {
            B[i + j * ldb] = B[i + j * ldb] / D[i] - B[(i + 1) + j * ldb] * E[i];
        }
    }
}
