#include "internal_build_defs.h"
#include "semicolon_lapack_double.h"
/**
 * @file dgttrs.c
 * @brief DGTTRS solves a system of linear equations with a tridiagonal matrix
 *        using the LU factorization computed by dgttrf.
 */

/**
 * DGTTRS solves one of the systems of equations
 *    A*X = B  or  A**T*X = B,
 * with a tridiagonal matrix A using the LU factorization computed
 * by DGTTRF.
 *
 * @param[in] trans   Specifies the form of the system of equations.
 *                    = 'N': A * X = B  (No transpose)
 *                    = 'T': A**T * X = B  (Transpose)
 *                    = 'C': A**T * X = B  (Conjugate transpose = Transpose)
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
void dgttrs(
    const char* trans,
    const INT n,
    const INT nrhs,
    const f64* restrict DL,
    const f64* restrict D,
    const f64* restrict DU,
    const f64* restrict DU2,
    const INT* restrict ipiv,
    f64* restrict B,
    const INT ldb,
    INT* info)
{
    INT notran;
    INT itrans;
    INT ldb_min;

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
        xerbla("DGTTRS", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        return;
    }

    /* Decode trans */
    if (notran) {
        itrans = 0;
    } else {
        itrans = 1;
    }

    /* For tridiagonal systems, LAPACK's ilaenv returns NB=1 (unblocked).
       Simply call dgtts2 directly for all right-hand sides. */
    dgtts2(itrans, n, nrhs, DL, D, DU, DU2, ipiv, B, ldb);
}
