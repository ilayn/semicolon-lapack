/**
 * @file cspsvx.c
 * @brief CSPSVX computes the solution to a complex system of linear equations A * X = B with error bounds.
 */

#include "semicolon_cblas.h"
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
 * @rst
 * The following steps are performed:
 *
 * 1. If ``fact='N'``, the diagonal pivoting method is used to factor A as
 *
 *    .. code-block:: text
 *
 *        A = U * D * U**T,  if uplo = 'U', or
 *        A = L * D * L**T,  if uplo = 'L',
 *
 *    where U (or L) is a product of permutation and unit upper (lower)
 *    triangular matrices and D is symmetric and block diagonal with
 *    1-by-1 and 2-by-2 diagonal blocks.
 *
 * 2. If some D(i,i)=0, so that D is exactly singular, then the routine
 *    returns with ``info=i``. Otherwise, the factored form of A is used
 *    to estimate the condition number of the matrix A. If the
 *    reciprocal of the condition number is less than machine precision,
 *    ``info=n+1`` is returned as a warning, but the routine still goes on
 *    to solve for X and compute error bounds as described below.
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
 *                       - `'F'`: On entry, AFP and IPIV contain the factored
 *                         form of A. AP, AFP and IPIV will not be modified.
 *                       - `'N'`: The matrix A will be copied to AFP and
 *                         factored.
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The number of linear equations, i.e., the order of
 *                       the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in]     AP     Complex array of dimension `n*(n+1)/2`.
 *                       The upper or lower triangle of the symmetric matrix
 *                       A, packed columnwise in a linear array. The j-th
 *                       column of A is stored in the array AP as follows:
 *                       if `uplo='U'`, `AP[i + j*(j+1)/2] = A(i,j)` for
 *                       `0<=i<=j`; if `uplo='L'`,
 *                       `AP[i + j*(2*n-j-1)/2] = A(i,j)` for `j<=i<=n-1`.
 *                       See below for further details.
 * @param[in,out] AFP    Complex array of dimension `n*(n+1)/2`.
 *                       If `fact='F'`, then AFP is an input argument and on
 *                       entry contains the block diagonal matrix D and the
 *                       multipliers used to obtain the factor U or L from the
 *                       factorization A = U*D*U**T or A = L*D*L**T as
 *                       computed by `csptrf`, stored as a packed triangular
 *                       matrix in the same storage format as A.
 *                       If `fact='N'`, then AFP is an output argument and on
 *                       exit contains the block diagonal matrix D and the
 *                       multipliers used to obtain the factor U or L.
 * @param[in,out] ipiv   Array of dimension `n`. Pivot indices (0-based).
 *                       If `fact='F'`, then ipiv is an input argument and on
 *                       entry contains details of the interchanges and the
 *                       block structure of D, as determined by `csptrf`.
 *                       If `ipiv[k]>=0`, then rows and columns `k` and
 *                       `ipiv[k]` were interchanged and `D(k,k)` is a 1-by-1
 *                       diagonal block. If `uplo='U'` and
 *                       `ipiv[k]=ipiv[k-1]<0`, then rows and columns `k-1`
 *                       and `-ipiv[k]-1` were interchanged and
 *                       `D(k-1:k,k-1:k)` is a 2-by-2 diagonal block. If
 *                       `uplo='L'` and `ipiv[k]=ipiv[k+1]<0`, then rows and
 *                       columns `k+1` and `-ipiv[k]-1` were interchanged and
 *                       `D(k:k+1,k:k+1)` is a 2-by-2 diagonal block.
 *                       If `fact='N'`, then ipiv is an output argument and on
 *                       exit contains details of the interchanges and the
 *                       block structure of D, as determined by `csptrf`.
 * @param[in]     B      Complex array of dimension `(ldb,nrhs)`.
 *                       The n-by-nrhs right hand side matrix B.
 * @param[in]     ldb    The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    X      Complex array of dimension `(ldx,nrhs)`.
 *                       If `info=0` or `info=n+1`, the n-by-nrhs solution
 *                       matrix X.
 * @param[in]     ldx    The leading dimension of the array X. `ldx>=max(1,n)`.
 * @param[out]    rcond  The estimate of the reciprocal condition number of
 *                       the matrix A. If `rcond` is less than the machine
 *                       precision (in particular, if `rcond=0`), the matrix
 *                       is singular to working precision. This condition is
 *                       indicated by a return code of `info>0`.
 * @param[out]    ferr   Array of dimension `nrhs`.
 *                       The estimated forward error bound for each solution
 *                       vector X(j).
 * @param[out]    berr   Array of dimension `nrhs`.
 *                       The componentwise relative backward error of each
 *                       solution vector X(j).
 * @param[out]    work   Complex array of dimension `2*n`.
 * @param[out]    rwork  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 *                         - `info>0`: if `info=i`, and `i<=n`, `D(i,i)` is exactly
 *                           zero. The factorization has been completed but D
 *                           is exactly singular, so the solution and error
 *                           bounds could not be computed. `rcond=0` is
 *                           returned.
 *                         - `info=n+1`: D is nonsingular, but `rcond` is less
 *                           than machine precision, meaning that the matrix
 *                           is singular to working precision.
 *
 * @par Further Details:
 * @rst
 * The packed storage scheme is illustrated by the following example
 * when ``n=4``, ``uplo='U'``:
 *
 * Two-dimensional storage of the symmetric matrix A:
 *
 * .. code-block:: text
 *
 *     a00 a01 a02 a03
 *         a11 a12 a13
 *             a22 a23     (aij = aji)
 *                 a33
 *
 * Packed storage of the upper triangle of A:
 *
 * .. code-block:: text
 *
 *     AP = [ a00, a01, a11, a02, a12, a22, a03, a13, a23, a33 ]
 * @endrst
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
