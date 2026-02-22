/**
 * @file cspsvx.c
 * @brief CSPSVX computes the solution to a complex system of linear equations A * X = B with error bounds.
 */

#include "internal_build_defs.h"
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CSPSVX uses the diagonal pivoting factorization A = U*D*U**T or
 * A = L*D*L**T to compute the solution to a complex system of linear
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
 * @param[in,out] AFP    If fact = 'F', contains the factored form from CSPTRF.
 *                       If fact = 'N', on exit contains the factored form.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in,out] ipiv   If fact = 'F', contains the pivot indices from CSPTRF.
 *                       If fact = 'N', on exit contains the pivot indices.
 *                       Array of dimension (n).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    X      The solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    rcond  The reciprocal condition number of A.
 * @param[out]    ferr   The forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr   The backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Single precision workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, D(i,i) is exactly zero, or
 *                           if info = n+1, the matrix is singular to working precision
 */
void cspsvx(
    const char* fact,
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c64* restrict AP,
    c64* restrict AFP,
    INT* restrict ipiv,
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
        xerbla("CSPSVX", -(*info));
        return;
    }

    if (nofact) {
        // Compute the factorization A = U*D*U**T or A = L*D*L**T
        cblas_ccopy(n * (n + 1) / 2, AP, 1, AFP, 1);
        csptrf(uplo, n, AFP, ipiv, info);

        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A
    anorm = clansp("I", uplo, n, AP, rwork);

    // Compute the reciprocal of the condition number of A
    cspcon(uplo, n, AFP, ipiv, anorm, rcond, work, info);

    // Compute the solution vectors X
    clacpy("F", n, nrhs, B, ldb, X, ldx);
    csptrs(uplo, n, nrhs, AFP, ipiv, X, ldx, info);

    // Use iterative refinement to improve the computed solutions and
    // compute error bounds and backward error estimates for them
    csprfs(uplo, n, nrhs, AP, AFP, ipiv, B, ldb, X, ldx, ferr, berr, work, rwork, info);

    // Set INFO = N+1 if the matrix is singular to working precision
    if (*rcond < slamch("E"))
        *info = n + 1;
}
