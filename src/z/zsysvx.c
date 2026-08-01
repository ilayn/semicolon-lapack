/**
 * @file zsysvx.c
 * @brief ZSYSVX computes the solution to a complex system of linear equations
 *        A * X = B for symmetric matrices, with condition estimation and
 *        iterative refinement.
 */

#include <complex.h>
#include <float.h>
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZSYSVX uses the diagonal pivoting factorization to compute the
 * solution to a complex system of linear equations A * X = B,
 * where A is an N-by-N symmetric matrix and X and B are N-by-NRHS
 * matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * @rst
 * The following steps are performed:
 *
 * 1. If ``fact='N'``, the diagonal pivoting method is used to factor A.
 *    The form of the factorization is
 *
 *    .. code-block:: text
 *
 *        A = U * D * U**T,  if uplo = 'U', or
 *        A = L * D * L**T,  if uplo = 'L',
 *
 *    where U (or L) is a product of permutation and unit upper (lower)
 *    triangular matrices, and D is symmetric and block diagonal with
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
 *                       - `'F'`: On entry, AF and IPIV contain the factored
 *                         form of A. AF and IPIV will not be modified.
 *                       - `'N'`: The matrix A will be copied to AF and
 *                         factored.
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n     The number of linear equations, i.e., the order of
 *                      the matrix A. `n>=0`.
 * @param[in]     nrhs  The number of right hand sides. `nrhs>=0`.
 * @param[in]     A     Complex array of dimension `(lda,n)`.
 *                      The symmetric matrix A.
 * @param[in]     lda   The leading dimension of the array A. `lda>=max(1,n)`.
 * @param[in,out] AF    Complex array of dimension `(ldaf,n)`.
 *                      If `fact='F'`, then AF is an input argument and on
 *                      entry contains the block diagonal matrix D and the
 *                      multipliers used to obtain the factor U or L from the
 *                      factorization A = U*D*U**T or A = L*D*L**T as computed
 *                      by `zsytrf`.
 *                      If `fact='N'`, then AF is an output argument and on
 *                      exit returns the block diagonal matrix D and the
 *                      multipliers used to obtain the factor U or L.
 * @param[in]     ldaf  The leading dimension of the array AF. `ldaf>=max(1,n)`.
 * @param[in,out] ipiv  Array of dimension `n`. Pivot indices (0-based).
 *                      If `fact='F'`, then ipiv is an input argument and on
 *                      entry contains details of the interchanges and the
 *                      block structure of D, as determined by `zsytrf`.
 *                      If `ipiv[k]>=0`, then rows and columns `k` and
 *                      `ipiv[k]` were interchanged and `D(k,k)` is a 1-by-1
 *                      diagonal block. If `uplo='U'` and
 *                      `ipiv[k]=ipiv[k-1]<0`, then rows and columns `k-1`
 *                      and `-ipiv[k]-1` were interchanged and
 *                      `D(k-1:k,k-1:k)` is a 2-by-2 diagonal block. If
 *                      `uplo='L'` and `ipiv[k]=ipiv[k+1]<0`, then rows and
 *                      columns `k+1` and `-ipiv[k]-1` were interchanged and
 *                      `D(k:k+1,k:k+1)` is a 2-by-2 diagonal block.
 *                      If `fact='N'`, then ipiv is an output argument and on
 *                      exit contains details of the interchanges and the
 *                      block structure of D, as determined by `zsytrf`.
 * @param[in]     B     Complex array of dimension `(ldb,nrhs)`.
 *                       The n-by-nrhs right hand side matrix B.
 * @param[in]     ldb   The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[out]    X     Complex array of dimension `(ldx,nrhs)`.
 *                      If `info=0` or `info=n+1`, the n-by-nrhs solution
 *                      matrix X.
 * @param[in]     ldx   The leading dimension of the array X. `ldx>=max(1,n)`.
 * @param[out]    rcond The estimate of the reciprocal condition number of
 *                      the matrix A. If `rcond` is less than the machine
 *                      precision (in particular, if `rcond=0`), the matrix
 *                      is singular to working precision. This condition is
 *                      indicated by a return code of `info>0`.
 * @param[out]    ferr  Array of dimension `nrhs`.
 *                      The estimated forward error bound for each solution
 *                      vector X(j).
 * @param[out]    berr  Array of dimension `nrhs`.
 *                      The componentwise relative backward error of each
 *                      solution vector X(j).
 * @param[out]    work  Complex array of dimension `max(1,lwork)`.
 *                      On exit, if `info=0`, `work[0]` returns the optimal `lwork`.
 * @param[in]     lwork The length of `work`. `lwork>=max(1,2*n)`, and for
 *                      best performance, when `fact='N'`,
 *                      `lwork>=max(1,2*n,n*nb)`, where `nb` is the optimal
 *                      block size for `zsytrf`.
 *                      If `lwork=-1`, then a workspace query is assumed; the
 *                      routine only calculates the optimal size of the
 *                      `work` array, returns this value as the first entry
 *                      of the `work` array, and no error message related to
 *                      `lwork` is issued.
 * @param[out]    rwork Array of dimension `n`.
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
 */
void zsysvx(
    const char* fact,
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c128* restrict A,
    const INT lda,
    c128* restrict AF,
    const INT ldaf,
    INT* restrict ipiv,
    const c128* restrict B,
    const INT ldb,
    c128* restrict X,
    const INT ldx,
    f64* rcond,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    const INT lwork,
    f64* restrict rwork,
    INT* info)
{
    const f64 ZERO = 0.0;

    INT nofact, lquery;
    INT lwkmin, lwkopt, nb;
    f64 anorm;

    /* Test the input parameters. */
    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    lquery = (lwork == -1);
    lwkmin = (1 > 2 * n) ? 1 : 2 * n;

    if (!nofact && !(fact[0] == 'F' || fact[0] == 'f')) {
        *info = -1;
    } else if (!(uplo[0] == 'U' || uplo[0] == 'u') &&
               !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldaf < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -11;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -13;
    } else if (lwork < lwkmin && !lquery) {
        *info = -18;
    }

    if (*info == 0) {
        lwkopt = lwkmin;
        if (nofact) {
            nb = lapack_get_nb("SYTRF");
            INT nbnb = n * nb;
            lwkopt = (lwkopt > nbnb) ? lwkopt : nbnb;
        }
        work[0] = (c128)lwkopt;
    }

    if (*info != 0) {
        xerbla("ZSYSVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (nofact) {
        /* Compute the factorization A = U*D*U**T or A = L*D*L**T. */
        zlacpy(uplo, n, n, A, lda, AF, ldaf);
        zsytrf(uplo, n, AF, ldaf, ipiv, work, lwork, info);

        /* Return if INFO is non-zero. */
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    /* Compute the norm of the matrix A. */
    anorm = zlansy("I", uplo, n, A, lda, rwork);

    /* Compute the reciprocal of the condition number of A. */
    zsycon(uplo, n, AF, ldaf, ipiv, anorm, rcond, work, info);

    /* Compute the solution vectors X. */
    zlacpy("F", n, nrhs, B, ldb, X, ldx);
    zsytrs(uplo, n, nrhs, AF, ldaf, ipiv, X, ldx, info);

    /* Use iterative refinement to improve the computed solutions and
     * compute error bounds and backward error estimates for them. */
    zsyrfs(uplo, n, nrhs, A, lda, AF, ldaf, ipiv, B, ldb, X, ldx,
           ferr, berr, work, rwork, info);

    /* Set INFO = N+1 if the matrix is singular to working precision. */
    if (*rcond < DBL_EPSILON) {
        *info = n + 1;
    }

    work[0] = (c128)lwkopt;
}
