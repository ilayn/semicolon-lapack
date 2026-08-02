/**
 * @file cppsvx.c
 * @brief CPPSVX computes the solution to a complex system of linear equations A * X = B with Hermitian positive definite matrix in packed storage (expert driver).
 */

#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPPSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to
 * compute the solution to a complex system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where A is an `n`-by-`n` Hermitian positive definite matrix stored in
 * packed format and X and `B` are `n`-by-`nrhs` matrices.
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
 *                       - `'F'`: `AFP` contains the factored form of `A`. If
 *                         `equed='Y'`, `A` has been equilibrated with
 *                         scaling factors given by `S`; `AP` and `AFP` will
 *                         not be modified.
 *                       - `'N'`: The matrix `A` will be copied to `AFP` and
 *                         factored.
 *                       - `'E'`: The matrix `A` will be equilibrated if
 *                         necessary, then copied to `AFP` and factored.
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The number of linear equations, i.e., the order of
 *                       the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in,out] AP     Array of dimension `n*(n+1)/2`.
 *                       On entry, the upper or lower triangle of the Hermitian
 *                       matrix A, packed columnwise in a linear array, except
 *                       if `fact='F'` and `equed='Y'`, then A must contain the
 *                       equilibrated matrix diag(S)*A*diag(S). The j-th column
 *                       of A is stored in the array AP as follows:
 *                       if `uplo='U'`, `AP[i + j*(j+1)/2] = A(i,j)` for `0<=i<=j`;
 *                       if `uplo='L'`, `AP[i + j*(2*n-j-1)/2] = A(i,j)` for
 *                       `j<=i<=n-1`.
 *                       See below for further details. A is not modified if
 *                       `fact='F'` or `'N'`, or if `fact='E'` and `equed='N'`
 *                       on exit.
 *                       On exit, if `fact='E'` and `equed='Y'`, A is
 *                       overwritten by diag(S)*A*diag(S).
 * @param[in,out] AFP    Array of dimension `n*(n+1)/2`.
 *                       If `fact='F'`, an input argument containing the
 *                       triangular factor U or L from the Cholesky
 *                       factorization, in the same storage format as A.
 *                       If `fact='N'` or `'E'`, an output argument returning
 *                       the triangular factor of the (possibly equilibrated)
 *                       matrix A.
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
 *                         - `info<0`: if `info=-i`, the i-th argument had an
 *                           illegal value
 *                         - `info>0`: if `info=i`, and `i<=n`, the leading
 *                           principal minor of order i of A is not positive,
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
 *
 * @par Further Details:
 * @rst
 * The packed storage scheme is illustrated by the following example
 * when ``n=4``, ``uplo='U'``:
 *
 * Two-dimensional storage of the Hermitian matrix A:
 *
 * .. code-block:: text
 *
 *     a00 a01 a02 a03
 *         a11 a12 a13
 *             a22 a23     (aij = conjg(aji))
 *                 a33
 *
 * Packed storage of the upper triangle of A:
 *
 * .. code-block:: text
 *
 *     AP = [ a00, a01, a11, a02, a12, a22, a03, a13, a23, a33 ]
 * @endrst
 */
void cppsvx(
    const char* fact,
    const char* uplo,
    const INT n,
    const INT nrhs,
    c64* restrict AP,
    c64* restrict AFP,
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

    INT equil, nofact, rcequ;
    INT i, infequ, j;
    f32 amax, anorm, bignum, scond, smax, smin, smlnum;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    equil = (fact[0] == 'E' || fact[0] == 'e');
    if (nofact || equil) {
        *equed = 'N';
        rcequ = 0;
    } else {
        rcequ = (*equed == 'Y' || *equed == 'y');
        smlnum = slamch("S");
        bignum = ONE / smlnum;
    }

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
            smin = bignum;
            smax = ZERO;
            for (j = 0; j < n; j++) {
                smin = (smin < S[j]) ? smin : S[j];
                smax = (smax > S[j]) ? smax : S[j];
            }
            if (smin <= ZERO) {
                *info = -8;
            } else if (n > 0) {
                scond = ((smin > smlnum) ? smin : smlnum) /
                        ((smax < bignum) ? smax : bignum);
            } else {
                scond = ONE;
            }
        }
        if (*info == 0) {
            if (ldb < (1 > n ? 1 : n)) {
                *info = -10;
            } else if (ldx < (1 > n ? 1 : n)) {
                *info = -12;
            }
        }
    }

    if (*info != 0) {
        xerbla("CPPSVX", -(*info));
        return;
    }

    if (equil) {
        cppequ(uplo, n, AP, S, &scond, &amax, &infequ);
        if (infequ == 0) {
            claqhp(uplo, n, AP, S, scond, amax, equed);
            rcequ = (*equed == 'Y' || *equed == 'y');
        }
    }

    if (rcequ) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = S[i] * B[i + j * ldb];
            }
        }
    }

    if (nofact || equil) {
        cblas_ccopy(n * (n + 1) / 2, AP, 1, AFP, 1);
        cpptrf(uplo, n, AFP, info);

        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    anorm = clanhp("I", uplo, n, AP, rwork);

    cppcon(uplo, n, AFP, anorm, rcond, work, rwork, info);

    clacpy("F", n, nrhs, B, ldb, X, ldx);
    cpptrs(uplo, n, nrhs, AFP, X, ldx, info);

    cpprfs(uplo, n, nrhs, AP, AFP, B, ldb, X, ldx, ferr, berr, work, rwork, info);

    if (rcequ) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                X[i + j * ldx] = S[i] * X[i + j * ldx];
            }
        }
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ferr[j] / scond;
        }
    }

    if (*rcond < slamch("E")) {
        *info = n + 1;
    }
}
