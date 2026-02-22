/**
 * @file dspsvx.c
 * @brief DSPSVX computes the solution to a real system of linear equations A * X = B with error bounds.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSPSVX uses the diagonal pivoting factorization A = U*D*U**T or
 * A = L*D*L**T to compute the solution to a real system of linear
 * equations A * X = B, where A is an N-by-N symmetric matrix stored
 * in packed format and X and B are N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * @param[in]     fact   = 'F': AFP and IPIV contain the factored form of A
 *                        = 'N': The matrix A will be copied to AFP and factored
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The original packed symmetric matrix A. Array of dimension (n*(n+1)/2).
 * @param[in,out] AFP    If fact = 'F', contains the factored form from DSPTRF.
 *                       If fact = 'N', on exit contains the factored form.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in,out] ipiv   If fact = 'F', contains the pivot indices from DSPTRF.
 *                       If fact = 'N', on exit contains the pivot indices.
 *                       Array of dimension (n).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    X      The solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    rcond  The reciprocal condition number of A.
 * @param[out]    ferr   The forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr   The backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) is exactly zero, or
 *                           if info = n+1, the matrix is singular to working precision
 */
void dspsvx(
    const char* fact,
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* restrict AP,
    f64* restrict AFP,
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

    INT nofact;
    f64 anorm;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    if (!nofact && !(fact[0] == 'F' || fact[0] == 'f')) {
        *info = -1;
    } else if (!(uplo[0] == 'U' || uplo[0] == 'u') && !(uplo[0] == 'L' || uplo[0] == 'l')) {
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
        xerbla("DSPSVX", -(*info));
        return;
    }

    if (nofact) {
        // Compute the factorization A = U*D*U**T or A = L*D*L**T
        cblas_dcopy(n * (n + 1) / 2, AP, 1, AFP, 1);
        dsptrf(uplo, n, AFP, ipiv, info);

        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A
    anorm = dlansp("I", uplo, n, AP, work);

    // Compute the reciprocal of the condition number of A
    dspcon(uplo, n, AFP, ipiv, anorm, rcond, work, iwork, info);

    // Compute the solution vectors X
    dlacpy("F", n, nrhs, B, ldb, X, ldx);
    dsptrs(uplo, n, nrhs, AFP, ipiv, X, ldx, info);

    // Use iterative refinement to improve the computed solutions and
    // compute error bounds and backward error estimates for them
    dsprfs(uplo, n, nrhs, AP, AFP, ipiv, B, ldb, X, ldx, ferr, berr, work, iwork, info);

    // Set INFO = N+1 if the matrix is singular to working precision
    if (*rcond < dlamch("E"))
        *info = n + 1;
}
