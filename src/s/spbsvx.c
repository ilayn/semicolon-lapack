/**
 * @file spbsvx.c
 * @brief SPBSVX computes the solution to a symmetric positive definite banded system with error bounds.
 */

#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SPBSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
 * compute the solution to a real system of linear equations
 * @rst
 * .. code-block:: text
 *
 *     A * X = B
 * @endrst
 * where A is an `n`-by-`n` symmetric positive definite band matrix and X
 * and `B` are `n`-by-`nrhs` matrices.
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
 *        A = U**T * U,  if uplo = 'U', or
 *        A = L * L**T,  if uplo = 'L',
 *
 *    where U is an upper triangular band matrix, and L is a lower
 *    triangular band matrix.
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
 *                       - `'F'`: `AFB` contains the factored form of `A`. If
 *                         `equed='Y'`, `A` has been equilibrated with
 *                         scaling factors given by `S`; `AB` and `AFB` will
 *                         not be modified.
 *                       - `'N'`: The matrix `A` will be copied to `AFB` and
 *                         factored.
 *                       - `'E'`: The matrix `A` will be equilibrated if
 *                         necessary, then copied to `AFB` and factored.
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The number of linear equations, i.e., the order of
 *                       the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in]     nrhs   The number of right-hand sides. `nrhs>=0`.
 * @param[in,out] AB     Array of dimension (`ldab`, `n`).
 *                       On entry, the upper or lower triangle of the symmetric
 *                       band matrix A, stored in the first `kd+1` rows of the
 *                       array, except if `fact='F'` and `equed='Y'`, then A
 *                       must contain the equilibrated matrix diag(S)*A*diag(S).
 *                       The j-th column of A is stored in the j-th column of
 *                       the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = A(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = A(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 *                       See below for further details.
 *                       On exit, if `fact='E'` and `equed='Y'`, A is
 *                       overwritten by diag(S)*A*diag(S).
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[in,out] AFB    Array of dimension (`ldafb`, `n`).
 *                       If `fact='F'`, an input argument containing the
 *                       triangular factor U or L from the Cholesky
 *                       factorization of the band matrix A, in the same
 *                       storage format as A (see `AB`). If `equed='Y'`, then
 *                       `AFB` is the factored form of the equilibrated matrix A.
 *                       If `fact='N'` or `'E'`, an output argument returning
 *                       the triangular factor U or L from the Cholesky
 *                       factorization of the (possibly equilibrated) matrix A.
 * @param[in]     ldafb  The leading dimension of the array `AFB`. `ldafb>=kd+1`.
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
 * @param[out]    work   Array of dimension `3*n`.
 * @param[out]    iwork  Array of dimension `n`.
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
 * The band storage scheme is illustrated by the following example, when
 * ``n=6``, ``kd=2``, and ``uplo='U'``:
 *
 * Two-dimensional storage of the symmetric matrix A:
 *
 * .. code-block:: text
 *
 *     a00  a01  a02
 *          a11  a12  a13
 *               a22  a23  a24
 *                    a33  a34  a35
 *                         a44  a45
 *     (aij=conjg(aji))         a55
 *
 * Band storage of the upper triangle of A:
 *
 * .. code-block:: text
 *
 *          *    *   a02  a13  a24  a35
 *          *   a01  a12  a23  a34  a45
 *         a00  a11  a22  a33  a44  a55
 *
 * Similarly, if ``uplo='L'`` the format of A is as follows:
 *
 * .. code-block:: text
 *
 *         a00  a11  a22  a33  a44  a55
 *         a10  a21  a32  a43  a54   *
 *         a20  a31  a42  a53   *    *
 *
 * Array elements marked * are not used by the routine.
 * @endrst
 */
void spbsvx(
    const char* fact,
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    f32* restrict AB,
    const INT ldab,
    f32* restrict AFB,
    const INT ldafb,
    char* equed,
    f32* restrict S,
    f32* restrict B,
    const INT ldb,
    f32* restrict X,
    const INT ldx,
    f32* rcond,
    f32* restrict ferr,
    f32* restrict berr,
    f32* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    INT equil, nofact, rcequ, upper;
    INT i, infequ, j, j1, j2;
    f32 amax, anorm, bignum = 0.0f, scond, smax, smin, smlnum = 0.0f;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    equil = (fact[0] == 'E' || fact[0] == 'e');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

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
    } else if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (kd < 0) {
        *info = -4;
    } else if (nrhs < 0) {
        *info = -5;
    } else if (ldab < kd + 1) {
        *info = -7;
    } else if (ldafb < kd + 1) {
        *info = -9;
    } else if ((fact[0] == 'F' || fact[0] == 'f') &&
               !(rcequ || *equed == 'N' || *equed == 'n')) {
        *info = -10;
    } else {
        if (rcequ) {
            smin = bignum;
            smax = ZERO;
            for (j = 0; j < n; j++) {
                if (smin > S[j]) smin = S[j];
                if (smax < S[j]) smax = S[j];
            }
            if (smin <= ZERO) {
                *info = -11;
            } else if (n > 0) {
                scond = (smin > smlnum ? smin : smlnum) / (smax < bignum ? smax : bignum);
            } else {
                scond = ONE;
            }
        }
        if (*info == 0) {
            if (ldb < (1 > n ? 1 : n)) {
                *info = -13;
            } else if (ldx < (1 > n ? 1 : n)) {
                *info = -15;
            }
        }
    }

    if (*info != 0) {
        xerbla("SPBSVX", -(*info));
        return;
    }

    if (equil) {
        // Compute row and column scalings to equilibrate the matrix A
        spbequ(uplo, n, kd, AB, ldab, S, &scond, &amax, &infequ);
        if (infequ == 0) {
            // Equilibrate the matrix
            slaqsb(uplo, n, kd, AB, ldab, S, scond, amax, equed);
            rcequ = (*equed == 'Y' || *equed == 'y');
        }
    }

    // Scale the right-hand side
    if (rcequ) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = S[i] * B[i + j * ldb];
            }
        }
    }

    if (nofact || equil) {
        // Compute the Cholesky factorization A = U**T*U or A = L*L**T
        if (upper) {
            for (j = 0; j < n; j++) {
                j1 = (0 > j - kd ? 0 : j - kd);
                cblas_scopy(j - j1 + 1, &AB[kd - j + j1 + j * ldab], 1,
                            &AFB[kd - j + j1 + j * ldafb], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                j2 = (n < j + kd + 1 ? n : j + kd + 1);
                cblas_scopy(j2 - j, &AB[0 + j * ldab], 1, &AFB[0 + j * ldafb], 1);
            }
        }

        spbtrf(uplo, n, kd, AFB, ldafb, info);

        // Return if INFO is non-zero
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A
    anorm = slansb("1", uplo, n, kd, AB, ldab, work);

    // Compute the reciprocal of the condition number of A
    spbcon(uplo, n, kd, AFB, ldafb, anorm, rcond, work, iwork, info);

    // Compute the solution matrix X
    slacpy("F", n, nrhs, B, ldb, X, ldx);
    spbtrs(uplo, n, kd, nrhs, AFB, ldafb, X, ldx, info);

    // Use iterative refinement to improve the computed solution and
    // compute error bounds and backward error estimates for it
    spbrfs(uplo, n, kd, nrhs, AB, ldab, AFB, ldafb, B, ldb, X, ldx,
           ferr, berr, work, iwork, info);

    // Transform the solution matrix X to a solution of the original system
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

    // Set INFO = N+1 if the matrix is singular to working precision
    if (*rcond < slamch("E"))
        *info = n + 1;
}
