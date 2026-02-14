/**
 * @file zhpsvx.c
 * @brief ZHPSVX computes the solution to a complex system of linear equations A * X = B for Hermitian packed matrices (expert driver).
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <cblas.h>

/**
 * ZHPSVX uses the diagonal pivoting factorization A = U*D*U**H or
 * A = L*D*L**H to compute the solution to a complex system of linear
 * equations A * X = B, where A is an N-by-N Hermitian matrix stored
 * in packed format and X and B are N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * The following steps are performed:
 *
 * 1. If FACT = 'N', the diagonal pivoting method is used to factor A as
 *       A = U * D * U**H,  if UPLO = 'U', or
 *       A = L * D * L**H,  if UPLO = 'L',
 *    where U (or L) is a product of permutation and unit upper (lower)
 *    triangular matrices and D is Hermitian and block diagonal with
 *    1-by-1 and 2-by-2 diagonal blocks.
 *
 * 2. If some D(i,i)=0, so that D is exactly singular, then the routine
 *    returns with INFO = i. Otherwise, the factored form of A is used
 *    to estimate the condition number of the matrix A.  If the
 *    reciprocal of the condition number is less than machine precision,
 *    INFO = N+1 is returned as a warning, but the routine still goes on
 *    to solve for X and compute error bounds as described below.
 *
 * 3. The system of equations is solved for X using the factored form
 *    of A.
 *
 * 4. Iterative refinement is applied to improve the computed solution
 *    matrix and calculate error bounds and backward error estimates
 *    for it.
 *
 * @param[in]     fact   Specifies whether the factored form of A has been supplied.
 *                       = 'F': AFP and IPIV contain the factored form of A.
 *                       = 'N': The matrix A will be copied to AFP and factored.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The number of linear equations. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The upper or lower triangle of the Hermitian matrix A,
 *                       packed columnwise. Array of dimension (n*(n+1)/2).
 * @param[in,out] AFP    If fact = 'F', contains the factored form from ZHPTRF.
 *                       If fact = 'N', on exit contains the factorization.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in,out] ipiv   If fact = 'F', contains the pivot details from ZHPTRF.
 *                       If fact = 'N', on exit contains the pivot details.
 *                       Array of dimension (n).
 * @param[in]     B      The N-by-NRHS right hand side matrix B.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    X      The N-by-NRHS solution matrix X.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    rcond  The reciprocal condition number estimate.
 * @param[out]    ferr   Forward error bounds. Array of dimension (nrhs).
 * @param[out]    berr   Backward error bounds. Array of dimension (nrhs).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, and i is
 *                           <= n: D(i,i) is exactly zero. RCOND = 0 is returned.
 *                           = n+1: D is nonsingular, but RCOND is less than machine
 *                                  precision.
 */
void zhpsvx(
    const char* fact,
    const char* uplo,
    const int n,
    const int nrhs,
    const c128* const restrict AP,
    c128* const restrict AFP,
    int* const restrict ipiv,
    const c128* const restrict B,
    const int ldb,
    c128* const restrict X,
    const int ldx,
    f64* rcond,
    f64* const restrict ferr,
    f64* const restrict berr,
    c128* const restrict work,
    f64* const restrict rwork,
    int* info)
{
    const f64 ZERO = 0.0;

    int nofact;
    f64 anorm;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    if (!nofact && !(fact[0] == 'F' || fact[0] == 'f')) {
        *info = -1;
    } else if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
               !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -9;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("ZHPSVX", -(*info));
        return;
    }

    if (nofact) {

        // Compute the factorization A = U*D*U**H or A = L*D*L**H.

        cblas_zcopy(n * (n + 1) / 2, AP, 1, AFP, 1);
        zhptrf(uplo, n, AFP, ipiv, info);

        // Return if INFO is non-zero.

        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A.

    anorm = zlanhp("I", uplo, n, AP, rwork);

    // Compute the reciprocal of the condition number of A.

    zhpcon(uplo, n, AFP, ipiv, anorm, rcond, work, info);

    // Compute the solution vectors X.

    zlacpy("F", n, nrhs, B, ldb, X, ldx);
    zhptrs(uplo, n, nrhs, AFP, ipiv, X, ldx, info);

    // Use iterative refinement to improve the computed solutions and
    // compute error bounds and backward error estimates for them.

    zhprfs(uplo, n, nrhs, AP, AFP, ipiv, B, ldb, X, ldx, ferr,
           berr, work, rwork, info);

    // Set INFO = N+1 if the matrix is singular to working precision.

    if (*rcond < dlamch("E")) {
        *info = n + 1;
    }
}
