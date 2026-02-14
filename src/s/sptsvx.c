/**
 * @file sptsvx.c
 * @brief SPTSVX uses the factorization A = L*D*L**T to compute the solution
 *        to a real system of linear equations A*X = B, where A is an N-by-N
 *        symmetric positive definite tridiagonal matrix.
 */

#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SPTSVX uses the factorization A = L*D*L**T to compute the solution
 * to a real system of linear equations A*X = B, where A is an N-by-N
 * symmetric positive definite tridiagonal matrix and X and B are
 * N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * @param[in]     fact   Specifies whether or not the factored form of A has
 *                       been supplied on entry.
 *                       = 'F': On entry, DF and EF contain the factored form
 *                              of A. D, E, DF, and EF will not be modified.
 *                       = 'N': The matrix A will be copied to DF and EF and
 *                              factored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides, i.e., the number
 *                       of columns of the matrices B and X. nrhs >= 0.
 * @param[in]     D      Double precision array, dimension (n).
 *                       The n diagonal elements of the tridiagonal matrix A.
 * @param[in]     E      Double precision array, dimension (n-1).
 *                       The (n-1) subdiagonal elements of the tridiagonal
 *                       matrix A.
 * @param[in,out] DF     Double precision array, dimension (n).
 *                       If fact = 'F', then DF is an input argument and on
 *                       entry contains the n diagonal elements of the diagonal
 *                       matrix D from the L*D*L**T factorization of A.
 *                       If fact = 'N', then DF is an output argument and on
 *                       exit contains the n diagonal elements of the diagonal
 *                       matrix D from the L*D*L**T factorization of A.
 * @param[in,out] EF     Double precision array, dimension (n-1).
 *                       If fact = 'F', then EF is an input argument and on
 *                       entry contains the (n-1) subdiagonal elements of the
 *                       unit bidiagonal factor L from the L*D*L**T factorization
 *                       of A.
 *                       If fact = 'N', then EF is an output argument and on exit
 *                       contains the (n-1) subdiagonal elements of the unit
 *                       bidiagonal factor L from the L*D*L**T factorization of A.
 * @param[in]     B      Double precision array, dimension (ldb, nrhs).
 *                       The N-by-NRHS right hand side matrix B.
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[out]    X      Double precision array, dimension (ldx, nrhs).
 *                       If info = 0 or info = n+1, the N-by-NRHS solution matrix X.
 * @param[in]     ldx    The leading dimension of the array X. ldx >= max(1,n).
 * @param[out]    rcond  The reciprocal condition number of the matrix A.
 *                       If rcond is less than the machine precision (in
 *                       particular, if rcond = 0), the matrix is singular to
 *                       working precision. This condition is indicated by a
 *                       return code of info > 0.
 * @param[out]    ferr   Double precision array, dimension (nrhs).
 *                       The forward error bound for each solution vector X(j).
 * @param[out]    berr   Double precision array, dimension (nrhs).
 *                       The componentwise relative backward error of each
 *                       solution vector X(j).
 * @param[out]    work   Double precision array, dimension (2*n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, and i is
 *                         - <= n: the leading principal minor of order i of A
 *                           is not positive, so the factorization could
 *                           not be completed, and the solution has not
 *                           been computed. rcond = 0 is returned.
 *                         - = n+1: U is nonsingular, but rcond is less than
 *                           machine precision, meaning that the matrix
 *                           is singular to working precision.
 */
void sptsvx(
    const char* fact,
    const int n,
    const int nrhs,
    const f32* const restrict D,
    const f32* const restrict E,
    f32* const restrict DF,
    f32* const restrict EF,
    const f32* const restrict B,
    const int ldb,
    f32* const restrict X,
    const int ldx,
    f32* rcond,
    f32* const restrict ferr,
    f32* const restrict berr,
    f32* const restrict work,
    int* info)
{
    const f32 ZERO = 0.0f;
    int nofact;
    f32 anorm;
    int max_n_1 = (1 > n) ? 1 : n;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    if (!nofact && !(fact[0] == 'F' || fact[0] == 'f')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < max_n_1) {
        *info = -9;
    } else if (ldx < max_n_1) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("SPTSVX", -(*info));
        return;
    }

    if (nofact) {

        /* Compute the L*D*L**T (or U**T*D*U) factorization of A. */

        cblas_scopy(n, D, 1, DF, 1);
        if (n > 1) {
            cblas_scopy(n - 1, E, 1, EF, 1);
        }
        spttrf(n, DF, EF, info);

        /* Return if INFO is non-zero. */

        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    /* Compute the norm of the matrix A. */

    anorm = slanst("1", n, D, E);

    /* Compute the reciprocal of the condition number of A. */

    sptcon(n, DF, EF, anorm, rcond, work, info);

    /* Compute the solution vectors X. */

    slacpy("Full", n, nrhs, B, ldb, X, ldx);
    spttrs(n, nrhs, DF, EF, X, ldx, info);

    /*
     * Use iterative refinement to improve the computed solutions and
     * compute error bounds and backward error estimates for them.
     */

    sptrfs(n, nrhs, D, E, DF, EF, B, ldb, X, ldx, ferr, berr, work, info);

    /* Set INFO = N+1 if the matrix is singular to working precision. */

    if (*rcond < slamch("Epsilon")) {
        *info = n + 1;
    }
}
