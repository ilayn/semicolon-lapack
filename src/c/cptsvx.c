/**
 * @file cptsvx.c
 * @brief CPTSVX uses the factorization A = L*D*L**H to compute the solution
 *        to a complex system of linear equations A*X = B, where A is an N-by-N
 *        Hermitian positive definite tridiagonal matrix.
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPTSVX uses the factorization A = L*D*L**H to compute the solution
 * to a complex system of linear equations A*X = B, where A is an N-by-N
 * Hermitian positive definite tridiagonal matrix and X and B are
 * N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * @rst
 * The following steps are performed:
 *
 * 1. If ``fact='N'``, the matrix A is factored as A = L*D*L**H, where L
 *    is a unit lower bidiagonal matrix and D is diagonal. The
 *    factorization can also be regarded as having the form
 *    A = U**H*D*U.
 *
 * 2. If the leading principal minor of order i is not positive,
 *    then the routine returns with ``info=i``. Otherwise, the factored
 *    form of A is used to estimate the condition number of the matrix
 *    A. If the reciprocal of the condition number is less than machine
 *    precision, ``info=n+1`` is returned as a warning, but the routine
 *    still goes on to solve for X and compute error bounds as
 *    described below.
 *
 * 3. The system of equations is solved for X using the factored form
 *    of A.
 *
 * 4. Iterative refinement is applied to improve the computed solution
 *    matrix and calculate error bounds and backward error estimates
 *    for it.
 * @endrst
 *
 * @param[in]     fact
 *                       - `'F'`: On entry, DF and EF contain the factored
 *                         form of A. D, E, DF, and EF will not be modified.
 *                       - `'N'`: The matrix A will be copied to DF and EF and
 *                         factored.
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides, i.e., the number
 *                       of columns of the matrices B and X. `nrhs>=0`.
 * @param[in]     D      Array of dimension `n`.
 *                       The n diagonal elements of the tridiagonal matrix A.
 * @param[in]     E      Complex array of dimension `n-1`.
 *                       The (n-1) subdiagonal elements of the tridiagonal
 *                       matrix A.
 * @param[in,out] DF     Array of dimension `n`.
 *                       If `fact='F'`, then DF is an input argument and on
 *                       entry contains the n diagonal elements of the diagonal
 *                       matrix D from the L*D*L**H factorization of A.
 *                       If `fact='N'`, then DF is an output argument and on
 *                       exit contains the n diagonal elements of the diagonal
 *                       matrix D from the L*D*L**H factorization of A.
 * @param[in,out] EF     Complex array of dimension `n-1`.
 *                       If `fact='F'`, then EF is an input argument and on
 *                       entry contains the (n-1) subdiagonal elements of the
 *                       unit bidiagonal factor L from the L*D*L**H factorization
 *                       of A.
 *                       If `fact='N'`, then EF is an output argument and on exit
 *                       contains the (n-1) subdiagonal elements of the unit
 *                       bidiagonal factor L from the L*D*L**H factorization of A.
 * @param[in]     B      Complex array of dimension `(ldb,nrhs)`.
 *                       The n-by-nrhs right hand side matrix B.
 * @param[in]     ldb    The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    X      Complex array of dimension `(ldx,nrhs)`.
 *                       If `info=0` or `info=n+1`, the n-by-nrhs solution matrix X.
 * @param[in]     ldx    The leading dimension of the array X. `ldx>=max(1,n)`.
 * @param[out]    rcond  The reciprocal condition number of the matrix A.
 *                       If `rcond` is less than the machine precision (in
 *                       particular, if `rcond=0`), the matrix is singular to
 *                       working precision. This condition is indicated by a
 *                       return code of `info>0`.
 * @param[out]    ferr   Array of dimension `nrhs`.
 *                       The forward error bound for each solution vector X(j).
 * @param[out]    berr   Array of dimension `nrhs`.
 *                       The componentwise relative backward error of each
 *                       solution vector X(j).
 * @param[out]    work   Complex array of dimension `n`.
 * @param[out]    rwork  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, and `i<=n`, the leading principal
 *                           minor of order i of A is not positive, so the
 *                           factorization could not be completed, and the solution
 *                           has not been computed. `rcond=0` is returned.
 *                         - `info=n+1`: U is nonsingular, but `rcond` is less than
 *                           machine precision, meaning that the matrix is singular
 *                           to working precision.
 */
void cptsvx(
    const char* fact,
    const INT n,
    const INT nrhs,
    const f32* restrict D,
    const c64* restrict E,
    f32* restrict DF,
    c64* restrict EF,
    const c64* restrict B,
    const INT ldb,
    c64* restrict X,
    const INT ldx,
    f32* rcond,
    f32* restrict ferr,
    f32* restrict berr,
    c64* restrict work,
    f32* restrict rwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    INT nofact;
    f32 anorm;
    INT max_n_1 = (1 > n) ? 1 : n;

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
        xerbla("CPTSVX", -(*info));
        return;
    }

    if (nofact) {

        /* Compute the L*D*L**H (or U**H*D*U) factorization of A. */

        cblas_scopy(n, D, 1, DF, 1);
        if (n > 1) {
            cblas_ccopy(n - 1, E, 1, EF, 1);
        }
        cpttrf(n, DF, EF, info);

        /* Return if INFO is non-zero. */

        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    /* Compute the norm of the matrix A. */

    anorm = clanht("1", n, D, E);

    /* Compute the reciprocal of the condition number of A. */

    cptcon(n, DF, EF, anorm, rcond, rwork, info);

    /* Compute the solution vectors X. */

    clacpy("Full", n, nrhs, B, ldb, X, ldx);
    cpttrs("Lower", n, nrhs, DF, EF, X, ldx, info);

    /*
     * Use iterative refinement to improve the computed solutions and
     * compute error bounds and backward error estimates for them.
     */

    cptrfs("Lower", n, nrhs, D, E, DF, EF, B, ldb, X, ldx, ferr, berr, work, rwork, info);

    /* Set INFO = N+1 if the matrix is singular to working precision. */

    if (*rcond < slamch("Epsilon")) {
        *info = n + 1;
    }
}
