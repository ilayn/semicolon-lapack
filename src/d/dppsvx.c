/**
 * @file dppsvx.c
 * @brief DPPSVX computes the solution to a real system of linear equations A * X = B with symmetric positive definite matrix in packed storage (expert driver).
 */

#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DPPSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
 * compute the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric positive definite matrix stored in
 * packed format and X and B are N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * @param[in]     fact   Specifies whether the factored form of A is supplied.
 *                       = 'F': AFP contains the factored form of A.
 *                       = 'N': The matrix A will be copied to AFP and factored.
 *                       = 'E': The matrix A will be equilibrated if necessary,
 *                              then copied to AFP and factored.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The number of linear equations. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] AP     On entry, the symmetric matrix A in packed format.
 *                       On exit, if fact = 'E' and equed = 'Y', A is overwritten
 *                       by diag(S)*A*diag(S). Array of dimension (n*(n+1)/2).
 * @param[in,out] AFP    If fact = 'F', AFP is an input with the factored form.
 *                       Otherwise, AFP is an output with the Cholesky factor.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in,out] equed  Specifies the form of equilibration.
 *                       = 'N': No equilibration
 *                       = 'Y': Equilibration was done
 *                       equed is an input if fact = 'F'; otherwise, it is an output.
 * @param[in,out] S      Scale factors for A. Array of dimension (n).
 * @param[in,out] B      On entry, the N-by-NRHS right hand side matrix B.
 *                       On exit, if equed = 'Y', B is overwritten by diag(S)*B.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    X      The solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    rcond  The reciprocal condition number estimate.
 * @param[out]    ferr   Forward error bounds. Array of dimension (nrhs).
 * @param[out]    berr   Backward error bounds. Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i and i <= n, the leading principal minor
 *                           of order i is not positive; if info = n+1, the matrix
 *                           is singular to working precision.
 */
void dppsvx(
    const char* fact,
    const char* uplo,
    const int n,
    const int nrhs,
    double* const restrict AP,
    double* const restrict AFP,
    char* equed,
    double* const restrict S,
    double* const restrict B,
    const int ldb,
    double* const restrict X,
    const int ldx,
    double* rcond,
    double* const restrict ferr,
    double* const restrict berr,
    double* const restrict work,
    int* const restrict iwork,
    int* info)
{
    // dppsvx.f lines 330-331: Parameters
    const double ZERO = 0.0;
    const double ONE = 1.0;

    // dppsvx.f lines 334-336: Local Scalars
    int equil, nofact, rcequ;
    int i, infequ, j;
    double amax, anorm, bignum, scond, smax, smin, smlnum;

    // dppsvx.f lines 353-363: Initialize
    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    equil = (fact[0] == 'E' || fact[0] == 'e');
    if (nofact || equil) {
        *equed = 'N';
        rcequ = 0;
    } else {
        rcequ = (*equed == 'Y' || *equed == 'y');
        smlnum = dlamch("S");  // dppsvx.f line 361: Safe minimum
        bignum = ONE / smlnum;  // dppsvx.f line 362
    }

    // dppsvx.f lines 367-406: Test the input parameters
    if (!nofact && !equil && !(fact[0] == 'F' || fact[0] == 'f')) {
        *info = -1;
    } else if (!((uplo[0] == 'U' || uplo[0] == 'u') ||
                 (uplo[0] == 'L' || uplo[0] == 'l'))) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if ((fact[0] == 'F' || fact[0] == 'f') &&
               !(rcequ || (*equed == 'N' || *equed == 'n'))) {
        *info = -7;
    } else {
        if (rcequ) {
            // dppsvx.f lines 384-397
            smin = bignum;
            smax = ZERO;
            for (j = 0; j < n; j++) {  // dppsvx.f line 387: DO 10 J = 1, N
                smin = (smin < S[j]) ? smin : S[j];  // dppsvx.f line 388
                smax = (smax > S[j]) ? smax : S[j];  // dppsvx.f line 389
            }
            if (smin <= ZERO) {
                *info = -8;
            } else if (n > 0) {
                // dppsvx.f line 394
                scond = ((smin > smlnum) ? smin : smlnum) /
                        ((smax < bignum) ? smax : bignum);
            } else {
                scond = ONE;  // dppsvx.f line 396
            }
        }
        if (*info == 0) {
            // dppsvx.f lines 399-405
            if (ldb < (1 > n ? 1 : n)) {
                *info = -10;
            } else if (ldx < (1 > n ? 1 : n)) {
                *info = -12;
            }
        }
    }

    if (*info != 0) {
        xerbla("DPPSVX", -(*info));
        return;
    }

    // dppsvx.f lines 413-425: Equilibrate the matrix if requested
    if (equil) {
        // dppsvx.f line 417: Compute row and column scalings
        dppequ(uplo, n, AP, S, &scond, &amax, &infequ);
        if (infequ == 0) {
            // dppsvx.f lines 422-423: Equilibrate the matrix
            dlaqsp(uplo, n, AP, S, scond, amax, equed);
            rcequ = (*equed == 'Y' || *equed == 'y');
        }
    }

    // dppsvx.f lines 429-435: Scale the right-hand side
    if (rcequ) {
        for (j = 0; j < nrhs; j++) {  // dppsvx.f line 430: DO 30 J = 1, NRHS
            for (i = 0; i < n; i++) {  // dppsvx.f line 431: DO 20 I = 1, N
                B[i + j * ldb] = S[i] * B[i + j * ldb];  // dppsvx.f line 432
            }
        }
    }

    // dppsvx.f lines 437-450: Compute the Cholesky factorization
    if (nofact || equil) {
        // dppsvx.f line 441: CALL DCOPY( N*( N+1 ) / 2, AP, 1, AFP, 1 )
        cblas_dcopy(n * (n + 1) / 2, AP, 1, AFP, 1);
        // dppsvx.f line 442
        dpptrf(uplo, n, AFP, info);

        // dppsvx.f lines 446-449: Return if INFO is non-zero
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // dppsvx.f line 454: Compute the norm of the matrix A
    anorm = dlansp("I", uplo, n, AP, work);

    // dppsvx.f line 458: Compute the reciprocal of the condition number of A
    dppcon(uplo, n, AFP, anorm, rcond, work, iwork, info);

    // dppsvx.f lines 462-463: Compute the solution matrix X
    dlacpy("F", n, nrhs, B, ldb, X, ldx);
    dpptrs(uplo, n, nrhs, AFP, X, ldx, info);

    // dppsvx.f lines 468-470: Use iterative refinement
    dpprfs(uplo, n, nrhs, AP, AFP, B, ldb, X, ldx, ferr, berr, work, iwork, info);

    // dppsvx.f lines 475-484: Transform the solution matrix X
    if (rcequ) {
        for (j = 0; j < nrhs; j++) {  // dppsvx.f line 476: DO 50 J = 1, NRHS
            for (i = 0; i < n; i++) {  // dppsvx.f line 477: DO 40 I = 1, N
                X[i + j * ldx] = S[i] * X[i + j * ldx];  // dppsvx.f line 478
            }
        }
        for (j = 0; j < nrhs; j++) {  // dppsvx.f line 481: DO 60 J = 1, NRHS
            ferr[j] = ferr[j] / scond;  // dppsvx.f line 482
        }
    }

    // dppsvx.f lines 488-489: Set INFO = N+1 if matrix is singular to working precision
    if (*rcond < dlamch("E")) {  // Epsilon
        *info = n + 1;
    }
}
