/**
 * @file zhesvx.c
 * @brief ZHESVX computes the solution to a complex system of linear equations
 *        A * X = B for Hermitian matrices, with condition estimation and
 *        iterative refinement.
 */

#include <complex.h>
#include <float.h>
#include "semicolon_lapack_complex_double.h"
#include "lapack_tuning.h"

/**
 * ZHESVX uses the diagonal pivoting factorization to compute the
 * solution to a complex system of linear equations A * X = B,
 * where A is an N-by-N Hermitian matrix and X and B are N-by-NRHS
 * matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * The following steps are performed:
 *
 * 1. If FACT = 'N', the diagonal pivoting method is used to factor A.
 *    The form of the factorization is
 *       A = U * D * U**H,  if UPLO = 'U', or
 *       A = L * D * L**H,  if UPLO = 'L',
 *    where U (or L) is a product of permutation and unit upper (lower)
 *    triangular matrices, and D is Hermitian and block diagonal with
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
 * @param[in]     fact  Specifies whether the factored form of A has been
 *                      supplied on entry.
 *                      = 'F': On entry, AF and IPIV contain the factored form.
 *                      = 'N': The matrix A will be copied to AF and factored.
 * @param[in]     uplo  = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The number of linear equations. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     A     The Hermitian matrix A.
 *                      Double complex array, dimension (lda, n).
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in,out] AF    If fact = 'F', contains the factored form from ZHETRF.
 *                      If fact = 'N', on exit contains the factored form.
 *                      Double complex array, dimension (ldaf, n).
 * @param[in]     ldaf  The leading dimension of the array AF. ldaf >= max(1, n).
 * @param[in,out] ipiv  If fact = 'F', contains the pivot indices from ZHETRF.
 *                      If fact = 'N', on exit contains the pivot indices.
 *                      Integer array, dimension (n).
 * @param[in]     B     The right hand side matrix B.
 *                      Double complex array, dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[out]    X     The solution matrix X.
 *                      Double complex array, dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1, n).
 * @param[out]    rcond The reciprocal condition number estimate of A.
 * @param[out]    ferr  Forward error bound for each solution vector.
 *                      Double precision array, dimension (nrhs).
 * @param[out]    berr  Backward error for each solution vector.
 *                      Double precision array, dimension (nrhs).
 * @param[out]    work  Double complex array, dimension (max(1, lwork)).
 *                      On exit, if info = 0, work[0] returns the optimal lwork.
 * @param[in]     lwork The length of work. lwork >= max(1, 2*n), and for best
 *                      performance lwork >= max(1, 2*n, n*nb).
 *                      If lwork = -1, a workspace query is assumed.
 * @param[out]    rwork Double precision array, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, and i is
 *                         - <= N: D(i,i) is exactly zero. The factorization has
 *                           been completed but D is exactly singular, so
 *                           the solution and error bounds could not be
 *                           computed. RCOND = 0 is returned.
 *                         - = N+1: D is nonsingular, but RCOND is less than
 *                           machine precision, meaning that the matrix
 *                           is singular to working precision.
 */
void zhesvx(
    const char* fact,
    const char* uplo,
    const int n,
    const int nrhs,
    const c128* restrict A,
    const int lda,
    c128* restrict AF,
    const int ldaf,
    int* restrict ipiv,
    const c128* restrict B,
    const int ldb,
    c128* restrict X,
    const int ldx,
    f64* rcond,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    const int lwork,
    f64* restrict rwork,
    int* info)
{
    const f64 ZERO = 0.0;

    int nofact, lquery;
    int lwkmin, lwkopt, nb;
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
            nb = lapack_get_nb("HETRF");
            int nbnb = n * nb;
            lwkopt = (lwkopt > nbnb) ? lwkopt : nbnb;
        }
        work[0] = CMPLX((f64)lwkopt, 0.0);
    }

    if (*info != 0) {
        xerbla("ZHESVX", -(*info));
        return;
    } else if (lquery) {
        return;
    }

    if (nofact) {
        /* Compute the factorization A = U*D*U**H or A = L*D*L**H. */
        zlacpy(uplo, n, n, A, lda, AF, ldaf);
        zhetrf(uplo, n, AF, ldaf, ipiv, work, lwork, info);

        /* Return if INFO is non-zero. */
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    /* Compute the norm of the matrix A. */
    anorm = zlanhe("I", uplo, n, A, lda, rwork);

    /* Compute the reciprocal of the condition number of A. */
    zhecon(uplo, n, AF, ldaf, ipiv, anorm, rcond, work, info);

    /* Compute the solution vectors X. */
    zlacpy("F", n, nrhs, B, ldb, X, ldx);
    zhetrs(uplo, n, nrhs, AF, ldaf, ipiv, X, ldx, info);

    /* Use iterative refinement to improve the computed solutions and
     * compute error bounds and backward error estimates for them. */
    zherfs(uplo, n, nrhs, A, lda, AF, ldaf, ipiv, B, ldb, X, ldx,
           ferr, berr, work, rwork, info);

    /* Set INFO = N+1 if the matrix is singular to working precision. */
    if (*rcond < dlamch("Epsilon")) {
        *info = n + 1;
    }

    work[0] = CMPLX((f64)lwkopt, 0.0);
}
