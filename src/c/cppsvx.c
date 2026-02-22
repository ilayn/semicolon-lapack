/**
 * @file cppsvx.c
 * @brief CPPSVX computes the solution to a complex system of linear equations A * X = B with Hermitian positive definite matrix in packed storage (expert driver).
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPPSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to
 * compute the solution to a complex system of linear equations
 *    A * X = B,
 * where A is an N-by-N Hermitian positive definite matrix stored in
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
 * @param[in,out] AP     On entry, the Hermitian matrix A in packed format.
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
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i and i <= n, the leading principal minor
 *                           of order i is not positive; if info = n+1, the matrix
 *                           is singular to working precision.
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
