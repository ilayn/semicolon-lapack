/**
 * @file zptts2.c
 * @brief ZPTTS2 solves a tridiagonal system of the form AX=B using the
 *        L*D*L**H factorization computed by ZPTTRF.
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPTTS2 solves a tridiagonal system of the form
 *    A * X = B
 * using the factorization A = U**H*D*U or A = L*D*L**H computed by ZPTTRF.
 * D is a diagonal matrix specified in the vector D, U (or L) is a unit
 * bidiagonal matrix whose superdiagonal (subdiagonal) is specified in
 * the vector E, and X and B are N by NRHS matrices.
 *
 * @param[in]     iuplo Specifies the form of the factorization and whether the
 *                      vector E is the superdiagonal of the upper bidiagonal factor
 *                      U or the subdiagonal of the lower bidiagonal factor L.
 *                      = 1:  A = U**H*D*U, E is the superdiagonal of U
 *                      = 0:  A = L*D*L**H, E is the subdiagonal of L
 * @param[in]     n     The order of the tridiagonal matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number
 *                      of columns of the matrix B. nrhs >= 0.
 * @param[in]     D     Double precision array, dimension (n).
 *                      The n diagonal elements of the diagonal matrix D from the
 *                      factorization A = U**H*D*U or A = L*D*L**H.
 * @param[in]     E     Complex*16 array, dimension (n-1).
 *                      If iuplo = 1, the (n-1) superdiagonal elements of the unit
 *                      bidiagonal factor U from the factorization A = U**H*D*U.
 *                      If iuplo = 0, the (n-1) subdiagonal elements of the unit
 *                      bidiagonal factor L from the factorization A = L*D*L**H.
 * @param[in,out] B     Complex*16 array, dimension (ldb, nrhs).
 *                      On entry, the right hand side vectors B for the system of
 *                      linear equations.
 *                      On exit, the solution vectors, X.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1,n).
 */
void zptts2(
    const int iuplo,
    const int n,
    const int nrhs,
    const double* const restrict D,
    const double complex* const restrict E,
    double complex* const restrict B,
    const int ldb)
{
    int i, j;

    /* Quick return if possible */

    if (n <= 1) {
        if (n == 1) {
            double scale = 1.0 / D[0];
            cblas_zdscal(nrhs, scale, B, ldb);
        }
        return;
    }

    if (iuplo == 1) {

        /* Solve A * X = B using the factorization A = U**H *D*U,
           overwriting each right hand side vector with its solution. */

        if (nrhs <= 2) {
            j = 0;
            L10:

            /* Solve U**H * x = b. */

            for (i = 1; i < n; i++) {
                B[i + j * ldb] = B[i + j * ldb] - B[(i - 1) + j * ldb] * conj(E[i - 1]);
            }

            /* Solve D * U * x = b. */

            for (i = 0; i < n; i++) {
                B[i + j * ldb] = B[i + j * ldb] / D[i];
            }
            for (i = n - 2; i >= 0; i--) {
                B[i + j * ldb] = B[i + j * ldb] - B[(i + 1) + j * ldb] * E[i];
            }
            if (j < nrhs - 1) {
                j = j + 1;
                goto L10;
            }
        } else {
            for (j = 0; j < nrhs; j++) {

                /* Solve U**H * x = b. */

                for (i = 1; i < n; i++) {
                    B[i + j * ldb] = B[i + j * ldb] - B[(i - 1) + j * ldb] * conj(E[i - 1]);
                }

                /* Solve D * U * x = b. */

                B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
                for (i = n - 2; i >= 0; i--) {
                    B[i + j * ldb] = B[i + j * ldb] / D[i] - B[(i + 1) + j * ldb] * E[i];
                }
            }
        }
    } else {

        /* Solve A * X = B using the factorization A = L*D*L**H,
           overwriting each right hand side vector with its solution. */

        if (nrhs <= 2) {
            j = 0;
            L80:

            /* Solve L * x = b. */

            for (i = 1; i < n; i++) {
                B[i + j * ldb] = B[i + j * ldb] - B[(i - 1) + j * ldb] * E[i - 1];
            }

            /* Solve D * L**H * x = b. */

            for (i = 0; i < n; i++) {
                B[i + j * ldb] = B[i + j * ldb] / D[i];
            }
            for (i = n - 2; i >= 0; i--) {
                B[i + j * ldb] = B[i + j * ldb] - B[(i + 1) + j * ldb] * conj(E[i]);
            }
            if (j < nrhs - 1) {
                j = j + 1;
                goto L80;
            }
        } else {
            for (j = 0; j < nrhs; j++) {

                /* Solve L * x = b. */

                for (i = 1; i < n; i++) {
                    B[i + j * ldb] = B[i + j * ldb] - B[(i - 1) + j * ldb] * E[i - 1];
                }

                /* Solve D * L**H * x = b. */

                B[(n - 1) + j * ldb] = B[(n - 1) + j * ldb] / D[n - 1];
                for (i = n - 2; i >= 0; i--) {
                    B[i + j * ldb] = B[i + j * ldb] / D[i] - B[(i + 1) + j * ldb] * conj(E[i]);
                }
            }
        }
    }
}
