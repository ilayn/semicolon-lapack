/**
 * @file zppsvx.c
 * @brief ZPPSVX computes the solution to a complex system of linear equations A * X = B with Hermitian positive definite matrix in packed storage (expert driver).
 */

#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPPSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to
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
void zppsvx(
    const char* fact,
    const char* uplo,
    const int n,
    const int nrhs,
    c128* const restrict AP,
    c128* const restrict AFP,
    char* equed,
    f64* const restrict S,
    c128* const restrict B,
    const int ldb,
    c128* const restrict X,
    const int ldx,
    f64* rcond,
    f64* const restrict ferr,
    f64* const restrict berr,
    c128* const restrict work,
    f64* const restrict rwork,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    int equil, nofact, rcequ;
    int i, infequ, j;
    f64 amax, anorm, bignum, scond, smax, smin, smlnum;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    equil = (fact[0] == 'E' || fact[0] == 'e');
    if (nofact || equil) {
        *equed = 'N';
        rcequ = 0;
    } else {
        rcequ = (*equed == 'Y' || *equed == 'y');
        smlnum = dlamch("S");
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
        xerbla("ZPPSVX", -(*info));
        return;
    }

    if (equil) {
        zppequ(uplo, n, AP, S, &scond, &amax, &infequ);
        if (infequ == 0) {
            zlaqhp(uplo, n, AP, S, scond, amax, equed);
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
        cblas_zcopy(n * (n + 1) / 2, AP, 1, AFP, 1);
        zpptrf(uplo, n, AFP, info);

        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    anorm = zlanhp("I", uplo, n, AP, rwork);

    zppcon(uplo, n, AFP, anorm, rcond, work, rwork, info);

    zlacpy("F", n, nrhs, B, ldb, X, ldx);
    zpptrs(uplo, n, nrhs, AFP, X, ldx, info);

    zpprfs(uplo, n, nrhs, AP, AFP, B, ldb, X, ldx, ferr, berr, work, rwork, info);

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

    if (*rcond < dlamch("E")) {
        *info = n + 1;
    }
}
