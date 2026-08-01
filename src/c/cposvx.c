/**
 * @file cposvx.c
 * @brief CPOSVX computes the solution to a complex system of linear equations
 *        A * X = B for Hermitian positive definite matrices, with equilibration,
 *        condition estimation, and iterative refinement.
 */

#include <complex.h>
#include <float.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPOSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to
 * compute the solution to a complex system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where A is an `n`-by-`n` Hermitian positive definite matrix and X and `B`
 * are `n`-by-`nrhs` matrices.
 *
 * Error bounds on the solution and a condition estimate are also provided.
 *
 * @rst
 * The following steps are performed:
 *
 * 1. If ``fact='E'``, real scaling factors are computed to equilibrate
 *    the system::
 *
 *        diag(S)*A*diag(S) * inv(diag(S))*X = diag(S)*B
 *
 *    Whether or not the system will be equilibrated depends on the
 *    scaling of the matrix ``A``, but if equilibration is used, ``A`` is
 *    overwritten by ``diag(S)*A*diag(S)`` and ``B`` by ``diag(S)*B``.
 *
 * 2. If ``fact='N'`` or ``'E'``, the Cholesky decomposition is used to
 *    factor the matrix ``A`` (after equilibration if ``fact='E'``) as::
 *
 *        A = U**H * U,  if uplo = 'U', or
 *        A = L * L**H,  if uplo = 'L',
 *
 *    where U is an upper triangular matrix and L is a lower triangular
 *    matrix.
 *
 * 3. If the leading principal minor of order i is not positive, then the
 *    routine returns with ``info=i``. Otherwise, the factored form of ``A``
 *    is used to estimate the condition number of the matrix ``A``. If the
 *    reciprocal of the condition number is less than machine precision,
 *    ``info=n+1`` is returned as a warning, but the routine still goes on
 *    to solve for X and compute error bounds as described below.
 *
 * 4. The system of equations is solved for X using the factored form of ``A``.
 *
 * 5. Iterative refinement is applied to improve the computed solution
 *    matrix and calculate error bounds and backward error estimates for it.
 *
 * 6. If equilibration was used, the matrix X is premultiplied by
 *    ``diag(S)`` so that it solves the original system before equilibration.
 * @endrst
 *
 * @param[in]     fact
 *                       - `'F'`: `AF` contains the factored form of `A`. If
 *                         `equed='Y'`, `A` has been equilibrated with
 *                         scaling factors given by `S`; `A` and `AF` will
 *                         not be modified.
 *                       - `'N'`: The matrix `A` will be copied to `AF` and
 *                         factored.
 *                       - `'E'`: The matrix `A` will be equilibrated if
 *                         necessary, then copied to `AF` and factored.
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The number of linear equations. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in,out] A      Array of dimension (`lda`, `n`).
 *                       On entry, the Hermitian matrix A, except if `fact='F'`
 *                       and `equed='Y'`, then A must contain the equilibrated
 *                       matrix diag(S)*A*diag(S). A is not modified if
 *                       `fact='F'` or `'N'`, or if `fact='E'` and `equed='N'`
 *                       on exit.
 *                       On exit, if `fact='E'` and `equed='Y'`, A is
 *                       overwritten by diag(S)*A*diag(S).
 * @param[in]     lda    The leading dimension of the array `A`. `lda>=max(1,n)`.
 * @param[in,out] AF     Array of dimension (`ldaf`, `n`).
 *                       If `fact='F'`, an input argument containing the
 *                       triangular factor U or L from the Cholesky
 *                       factorization, in the same storage format as A.
 *                       If `fact='N'` or `'E'`, an output argument returning
 *                       the triangular factor of the (possibly equilibrated)
 *                       matrix A.
 * @param[in]     ldaf   The leading dimension of the array `AF`. `ldaf>=max(1,n)`.
 * @param[in,out] equed
 *                       - `'N'`: No equilibration (always true if `fact='N'`)
 *                       - `'Y'`: Equilibration was done, i.e., A has been
 *                         replaced by diag(S) * A * diag(S)
 *
 *                       `equed` is an input argument if `fact='F'`; otherwise
 *                       it is an output argument.
 * @param[in,out] S      Array of dimension (`n`).
 *                       The scale factors for A; not accessed if `equed='N'`.
 *                       `S` is an input argument if `fact='F'`; otherwise it
 *                       is an output argument. If `fact='F'` and `equed='Y'`,
 *                       each element of `S` must be positive.
 * @param[in,out] B      Array of dimension (`ldb`, `nrhs`).
 *                       On entry, the `n`-by-`nrhs` right hand side matrix `B`.
 *                       On exit, if `equed='N'`, B is not modified; if
 *                       `equed='Y'`, B is overwritten by diag(S) * B.
 * @param[in]     ldb    The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[out]    X      Array of dimension (`ldx`, `nrhs`).
 *                       If `info=0` or `info=n+1`, the `n`-by-`nrhs` solution
 *                       matrix X to the original system of equations. Note
 *                       that if `equed='Y'`, A and B are modified on exit,
 *                       and the solution to the equilibrated system is
 *                       inv(diag(S))*X.
 * @param[in]     ldx    The leading dimension of the array `X`. `ldx>=max(1,n)`.
 * @param[out]    rcond  The estimate of the reciprocal condition number of
 *                       the matrix A after equilibration (if done).
 * @param[out]    ferr   Array of dimension (`nrhs`).
 *                       The estimated forward error bound for each solution
 *                       vector.
 * @param[out]    berr   Array of dimension (`nrhs`).
 *                       The componentwise relative backward error of each
 *                       solution vector.
 * @param[out]    work   Complex array of dimension `2*n`.
 * @param[out]    rwork  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-k`, the k-th argument had an
 *                           illegal value
 *                         - `info>0`: if `info=k`, and `k<=n`, the leading
 *                           principal minor of order k of A is not positive,
 *                           so the factorization could not be completed, and
 *                           the solution has not been computed. `rcond=0` is
 *                           returned.
 *                         - `info=n+1`: U is nonsingular, but `rcond` is less
 *                           than machine precision, meaning that the matrix
 *                           is singular to working precision. Nevertheless,
 *                           the solution and error bounds are computed
 *                           because there are a number of situations where
 *                           the computed solution can be more accurate than
 *                           the value of rcond would suggest.
 */
void cposvx(
    const char* fact,
    const char* uplo,
    const INT n,
    const INT nrhs,
    c64* restrict A,
    const INT lda,
    c64* restrict AF,
    const INT ldaf,
    char* equed,
    f32* restrict S,
    c64* restrict B,
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
    const f32 ONE = 1.0f;

    *info = 0;
    INT nofact = (fact[0] == 'N' || fact[0] == 'n');
    INT equil = (fact[0] == 'E' || fact[0] == 'e');
    INT rcequ = 0;
    f32 smlnum = 0.0f, bignum = 0.0f, scond = ONE, amax;

    if (nofact || equil) {
        *equed = 'N';
        rcequ = 0;
    } else {
        rcequ = (*equed == 'Y' || *equed == 'y');
        smlnum = slamch("S");
        bignum = ONE / smlnum;
    }

    // Test the input parameters
    if (!nofact && !equil && !(fact[0] == 'F' || fact[0] == 'f')) {
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
    } else if ((fact[0] == 'F' || fact[0] == 'f') &&
               !(rcequ || *equed == 'N' || *equed == 'n')) {
        *info = -9;
    } else {
        if (rcequ) {
            f32 smin_val = bignum;
            f32 smax_val = ZERO;
            for (INT j = 0; j < n; j++) {
                if (S[j] < smin_val) smin_val = S[j];
                if (S[j] > smax_val) smax_val = S[j];
            }
            if (smin_val <= ZERO) {
                *info = -10;
            } else if (n > 0) {
                f32 smin_clamped = smin_val > smlnum ? smin_val : smlnum;
                f32 smax_clamped = smax_val < bignum ? smax_val : bignum;
                scond = smin_clamped / smax_clamped;
            } else {
                scond = ONE;
            }
        }
        if (*info == 0) {
            if (ldb < (n > 1 ? n : 1)) {
                *info = -12;
            } else if (ldx < (n > 1 ? n : 1)) {
                *info = -14;
            }
        }
    }

    if (*info != 0) {
        xerbla("CPOSVX", -(*info));
        return;
    }

    if (equil) {
        // Compute row and column scalings to equilibrate the matrix A.
        INT infequ;
        cpoequ(n, A, lda, S, &scond, &amax, &infequ);
        if (infequ == 0) {
            // Equilibrate the matrix.
            claqhe(uplo, n, A, lda, S, scond, amax, equed);
            rcequ = (*equed == 'Y' || *equed == 'y');
        }
    }

    // Scale the right hand side.
    if (rcequ) {
        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                B[i + j * ldb] = S[i] * B[i + j * ldb];
            }
        }
    }

    if (nofact || equil) {
        // Compute the Cholesky factorization A = U**H*U or A = L*L**H.
        clacpy(uplo, n, n, A, lda, AF, ldaf);
        cpotrf(uplo, n, AF, ldaf, info);

        // Return if INFO is non-zero.
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A.
    f32 anorm = clanhe("1", uplo, n, A, lda, rwork);

    // Compute the reciprocal of the condition number of A.
    cpocon(uplo, n, AF, ldaf, anorm, rcond, work, rwork, info);

    // Compute the solution matrix X.
    clacpy("F", n, nrhs, B, ldb, X, ldx);
    cpotrs(uplo, n, nrhs, AF, ldaf, X, ldx, info);

    // Use iterative refinement to improve the computed solution and
    // compute error bounds and backward error estimates for it.
    cporfs(uplo, n, nrhs, A, lda, AF, ldaf, B, ldb, X, ldx,
           ferr, berr, work, rwork, info);

    // Transform the solution matrix X to a solution of the original system.
    if (rcequ) {
        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                X[i + j * ldx] = S[i] * X[i + j * ldx];
            }
        }
        for (INT j = 0; j < nrhs; j++) {
            ferr[j] = ferr[j] / scond;
        }
    }

    // Set INFO = N+1 if the matrix is singular to working precision.
    if (*rcond < slamch("E")) {
        *info = n + 1;
    }
}
