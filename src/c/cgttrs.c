#include "semicolon_lapack_complex_single.h"
#include <complex.h>
/**
 * @file cgttrs.c
 * @brief CGTTRS solves a system of linear equations with a tridiagonal matrix
 *        using the LU factorization computed by cgttrf.
 */

/**
 * CGTTRS solves one of the systems of equations
 *    A*X = B,  A**T*X = B,  or  A**H*X = B,
 * with a tridiagonal matrix A using the LU factorization computed
 * by CGTTRF.
 *
 * @param[in] trans   Specifies the form of the system of equations.
 *                    = 'N': A * X = B  (No transpose)
 *                    = 'T': A**T * X = B  (Transpose)
 *                    = 'C': A**H * X = B  (Conjugate transpose)
 * @param[in] n       The order of the matrix A. n >= 0.
 * @param[in] nrhs    The number of right hand sides, i.e., the number of columns
 *                    of the matrix B. nrhs >= 0.
 * @param[in] DL      The (n-1) multipliers that define the matrix L from the
 *                    LU factorization of A. Array of dimension (n-1).
 * @param[in] D       The n diagonal elements of the upper triangular matrix U from
 *                    the LU factorization of A. Array of dimension (n).
 * @param[in] DU      The (n-1) elements of the first super-diagonal of U.
 *                    Array of dimension (n-1).
 * @param[in] DU2     The (n-2) elements of the second super-diagonal of U.
 *                    Array of dimension (n-2).
 * @param[in] ipiv    The pivot indices; for 0 <= i < n, row i of the matrix was
 *                    interchanged with row ipiv[i]. ipiv[i] will always be either
 *                    i or i+1; ipiv[i] = i indicates a row interchange was not
 *                    required. Array of dimension (n).
 * @param[in,out] B   On entry, the matrix of right hand side vectors B.
 *                    On exit, B is overwritten by the solution vectors X.
 *                    Array of dimension (ldb, nrhs).
 * @param[in] ldb     The leading dimension of the array B. ldb >= max(1, n).
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void cgttrs(
    const char* trans,
    const int n,
    const int nrhs,
    const c64* restrict DL,
    const c64* restrict D,
    const c64* restrict DU,
    const c64* restrict DU2,
    const int* restrict ipiv,
    c64* restrict B,
    const int ldb,
    int* info)
{
    int notran;
    int itrans;
    int ldb_min;

    *info = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (!notran && !(trans[0] == 'T' || trans[0] == 't') && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else {
        ldb_min = (n > 1) ? n : 1;
        if (ldb < ldb_min) {
            *info = -10;
        }
    }

    if (*info != 0) {
        xerbla("CGTTRS", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return;
    }

    /* Decode trans */
    if (notran) {
        itrans = 0;
    } else if (trans[0] == 'T' || trans[0] == 't') {
        itrans = 1;
    } else {
        itrans = 2;
    }

    /* For tridiagonal systems, LAPACK's ilaenv returns NB=1 (unblocked).
       Simply call cgtts2 directly for all right-hand sides. */
    cgtts2(itrans, n, nrhs, DL, D, DU, DU2, ipiv, B, ldb);
}
