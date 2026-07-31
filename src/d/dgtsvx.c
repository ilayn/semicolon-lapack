/**
 * @file dgtsvx.c
 * @brief DGTSVX computes the solution to system of linear equations A * X = B
 *        for GT matrices with condition estimation and error bounds.
 */

#include <string.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DGTSVX uses the LU factorization to compute the solution to a real
 * system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B  or  A**T * X = B
 * @endrst
 * where A is a tridiagonal matrix of order `n` and X and `B` are `n`-by-`nrhs`
 * matrices.
 *
 * Error bounds on the solution and a condition estimate are also provided.
 *
 * @rst
 * The following steps are performed:
 *
 * 1. If ``fact='N'``, the LU decomposition is used to factor the matrix A
 *    as A = L * U, where L is a product of permutation and unit lower
 *    bidiagonal matrices and U is upper triangular with nonzeros in
 *    only the main diagonal and first two superdiagonals.
 *
 * 2. If some ``U(i,i)=0``, so that U is exactly singular, then the routine
 *    returns with ``info=i``. Otherwise, the factored form of A is used
 *    to estimate the condition number of the matrix A. If the reciprocal
 *    of the condition number is less than machine precision, ``info=n+1``
 *    is returned as a warning, but the routine still goes on to solve for
 *    X and compute error bounds as described below.
 *
 * 3. The system of equations is solved for X using the factored form of A.
 *
 * 4. Iterative refinement is applied to improve the computed solution
 *    matrix and calculate error bounds and backward error estimates for it.
 * @endrst
 *
 * @param[in]     fact
 *                       - `'F'`: `DLF`, `DF`, `DUF`, `DU2`, and `ipiv` contain the
 *                         factored form of A.
 *                       - `'N'`: The matrix will be copied and factored.
 * @param[in]     trans  Specifies the form of the system of equations:
 *                       - `'N'`: A * X = B (No transpose)
 *                       - `'T'`: A**T * X = B (Transpose)
 *                       - `'C'`: A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in]     DL     Array of dimension (`n-1`).
 *                       The (n-1) subdiagonal elements of A.
 * @param[in]     D      Array of dimension (`n`).
 *                       The diagonal elements of A.
 * @param[in]     DU     Array of dimension (`n-1`).
 *                       The (n-1) superdiagonal elements of A.
 * @param[in,out] DLF    Array of dimension (`n-1`).
 *                       If `fact='F'`, the (n-1) multipliers from LU factorization.
 *                       If `fact='N'`, output.
 * @param[in,out] DF     Array of dimension (`n`).
 *                       If `fact='F'`, the n diagonal elements of U.
 *                       If `fact='N'`, output.
 * @param[in,out] DUF    Array of dimension (`n-1`).
 *                       If `fact='F'`, the (n-1) elements of first superdiagonal of U.
 *                       If `fact='N'`, output.
 * @param[in,out] DU2    Array of dimension (`n-2`).
 *                       If `fact='F'`, the (n-2) elements of second superdiagonal of U.
 *                       If `fact='N'`, output.
 * @param[in,out] ipiv   Array of dimension (`n`).
 *                       If `fact='F'`, the pivot indices from factorization.
 *                       If `fact='N'`, output.
 * @param[in]     B      Array of dimension (`ldb`, `nrhs`).
 *                       The `n`-by-`nrhs` right hand side matrix.
 * @param[in]     ldb    The leading dimension of `B`. `ldb>=max(1,n)`.
 * @param[out]    X      Array of dimension (`ldx`, `nrhs`).
 *                       The `n`-by-`nrhs` solution matrix.
 * @param[in]     ldx    The leading dimension of `X`. `ldx>=max(1,n)`.
 * @param[out]    rcond  The reciprocal condition number estimate.
 * @param[out]    ferr   Array of dimension (`nrhs`).
 *                       Forward error bounds for each solution vector.
 * @param[out]    berr   Array of dimension (`nrhs`).
 *                       Backward error for each solution vector.
 * @param[out]    work   Workspace array of dimension (`3*n`).
 * @param[out]    iwork  Integer workspace array of dimension (`n`).
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i` (i <= `n`), U(i,i) is exactly zero
 *                         - `info=n+1`: U is nonsingular, but `rcond` < machine
 *                           precision
 */
void dgtsvx(
    const char* fact,
    const char* trans,
    const INT n,
    const INT nrhs,
    const f64* restrict DL,
    const f64* restrict D,
    const f64* restrict DU,
    f64* restrict DLF,
    f64* restrict DF,
    f64* restrict DUF,
    f64* restrict DU2,
    INT* restrict ipiv,
    const f64* restrict B,
    const INT ldb,
    f64* restrict X,
    const INT ldx,
    f64* rcond,
    f64* restrict ferr,
    f64* restrict berr,
    f64* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f64 ZERO = 0.0;

    INT nofact, notran;
    char norm;
    f64 anorm;
    INT ldb_min, ldx_min;
    INT j;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (!nofact && !(fact[0] == 'F' || fact[0] == 'f')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't') && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else {
        ldb_min = (n > 1) ? n : 1;
        ldx_min = (n > 1) ? n : 1;
        if (ldb < ldb_min) {
            *info = -14;
        } else if (ldx < ldx_min) {
            *info = -16;
        }
    }

    if (*info != 0) {
        xerbla("DGTSVX", -(*info));
        return;
    }

    if (nofact) {
        /* Compute the LU factorization of A */
        /* Copy D to DF */
        cblas_dcopy(n, D, 1, DF, 1);
        if (n > 1) {
            /* Copy DL to DLF and DU to DUF */
            cblas_dcopy(n - 1, DL, 1, DLF, 1);
            cblas_dcopy(n - 1, DU, 1, DUF, 1);
        }
        dgttrf(n, DLF, DF, DUF, DU2, ipiv, info);

        /* Return if info is non-zero */
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    /* Compute the norm of the matrix A */
    if (notran) {
        norm = '1';
    } else {
        norm = 'I';
    }
    anorm = dlangt(&norm, n, DL, D, DU);

    /* Compute the reciprocal of the condition number of A */
    dgtcon(&norm, n, DLF, DF, DUF, DU2, ipiv, anorm, rcond, work, iwork, info);

    /* Compute the solution vectors X */
    /* Copy B to X */
    for (j = 0; j < nrhs; j++) {
        cblas_dcopy(n, &B[j * ldb], 1, &X[j * ldx], 1);
    }
    dgttrs(trans, n, nrhs, DLF, DF, DUF, DU2, ipiv, X, ldx, info);

    /* Use iterative refinement to improve the computed solutions */
    dgtrfs(trans, n, nrhs, DL, D, DU, DLF, DF, DUF, DU2, ipiv,
           B, ldb, X, ldx, ferr, berr, work, iwork, info);

    /* Set info = n+1 if the matrix is singular to working precision */
    if (*rcond < dlamch("E")) {
        *info = n + 1;
    }
}
