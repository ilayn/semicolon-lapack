/**
 * @file sgtsvx.c
 * @brief SGTSVX computes the solution to system of linear equations A * X = B
 *        for GT matrices with condition estimation and error bounds.
 */

#include <string.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGTSVX uses the LU factorization to compute the solution to a real
 * system of linear equations A * X = B or A**T * X = B,
 * where A is a tridiagonal matrix of order N and X and B are N-by-NRHS
 * matrices.
 *
 * Error bounds on the solution and a condition estimate are also provided.
 *
 * @param[in]     fact   Specifies whether the factored form of A has been supplied.
 *                       = 'F': DLF, DF, DUF, DU2, and IPIV contain the factored form.
 *                       = 'N': The matrix will be copied and factored.
 * @param[in]     trans  Specifies the form of the system of equations:
 *                       = 'N': A * X = B     (No transpose)
 *                       = 'T': A**T * X = B  (Transpose)
 *                       = 'C': A**H * X = B  (Conjugate transpose = Transpose)
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     DL     The (n-1) subdiagonal elements of A. Array of dimension (n-1).
 * @param[in]     D      The diagonal elements of A. Array of dimension (n).
 * @param[in]     DU     The (n-1) superdiagonal elements of A. Array of dimension (n-1).
 * @param[in,out] DLF    If fact = "F", the (n-1) multipliers from LU factorization.
 *                       If fact = "N", output. Array of dimension (n-1).
 * @param[in,out] DF     If fact = "F", the n diagonal elements of U.
 *                       If fact = "N", output. Array of dimension (n).
 * @param[in,out] DUF    If fact = "F", the (n-1) elements of first superdiagonal of U.
 *                       If fact = "N", output. Array of dimension (n-1).
 * @param[in,out] DU2    If fact = "F", the (n-2) elements of second superdiagonal of U.
 *                       If fact = "N", output. Array of dimension (n-2).
 * @param[in,out] ipiv   If fact = "F", the pivot indices from factorization.
 *                       If fact = "N", output. Array of dimension (n).
 * @param[in]     B      The N-by-NRHS right hand side matrix. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[out]    X      The N-by-NRHS solution matrix. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1, n).
 * @param[out]    rcond  The reciprocal condition number estimate.
 * @param[out]    ferr   Forward error bounds for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr   Backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i (i <= n), U(i,i) is exactly zero
 *                         - = n+1: U is nonsingular, but rcond < machine precision
 */
void sgtsvx(
    const char* fact,
    const char* trans,
    const int n,
    const int nrhs,
    const f32 * const restrict DL,
    const f32 * const restrict D,
    const f32 * const restrict DU,
    f32 * const restrict DLF,
    f32 * const restrict DF,
    f32 * const restrict DUF,
    f32 * const restrict DU2,
    int * const restrict ipiv,
    const f32 * const restrict B,
    const int ldb,
    f32 * const restrict X,
    const int ldx,
    f32 *rcond,
    f32 * const restrict ferr,
    f32 * const restrict berr,
    f32 * const restrict work,
    int * const restrict iwork,
    int *info)
{
    const f32 ZERO = 0.0f;

    int nofact, notran;
    char norm;
    f32 anorm;
    int ldb_min, ldx_min;
    int j;

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
        xerbla("SGTSVX", -(*info));
        return;
    }

    if (nofact) {
        /* Compute the LU factorization of A */
        /* Copy D to DF */
        cblas_scopy(n, D, 1, DF, 1);
        if (n > 1) {
            /* Copy DL to DLF and DU to DUF */
            cblas_scopy(n - 1, DL, 1, DLF, 1);
            cblas_scopy(n - 1, DU, 1, DUF, 1);
        }
        sgttrf(n, DLF, DF, DUF, DU2, ipiv, info);

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
    anorm = slangt(&norm, n, DL, D, DU);

    /* Compute the reciprocal of the condition number of A */
    sgtcon(&norm, n, DLF, DF, DUF, DU2, ipiv, anorm, rcond, work, iwork, info);

    /* Compute the solution vectors X */
    /* Copy B to X */
    for (j = 0; j < nrhs; j++) {
        cblas_scopy(n, B + j * ldb, 1, X + j * ldx, 1);
    }
    sgttrs(trans, n, nrhs, DLF, DF, DUF, DU2, ipiv, X, ldx, info);

    /* Use iterative refinement to improve the computed solutions */
    sgtrfs(trans, n, nrhs, DL, D, DU, DLF, DF, DUF, DU2, ipiv,
           B, ldb, X, ldx, ferr, berr, work, iwork, info);

    /* Set info = n+1 if the matrix is singular to working precision */
    if (*rcond < slamch("E")) {
        *info = n + 1;
    }
}
