/**
 * @file zpbsvx.c
 * @brief ZPBSVX computes the solution to a Hermitian positive definite banded system with error bounds.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBSVX uses the Cholesky factorization A = U**H*U or A = L*L**H to
 * compute the solution to a complex system of linear equations
 *    A * X = B,
 * where A is an N-by-N Hermitian positive definite band matrix and X
 * and B are N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * @param[in]     fact   = 'F': AFB contains the factored form of A
 *                        = 'N': The matrix A will be copied to AFB and factored
 *                        = 'E': The matrix A will be equilibrated, copied, and factored
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] AB     The banded matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in,out] AFB    The factored form of A. Array of dimension (ldafb, n).
 * @param[in]     ldafb  The leading dimension of AFB. ldafb >= kd+1.
 * @param[in,out] equed  = 'N': No equilibration
 *                        = 'Y': Equilibration was done
 * @param[in,out] S      The scale factors for A. Array of dimension (n).
 * @param[in,out] B      On entry, the right hand side matrix B.
 *                       On exit, if EQUED='Y', B is overwritten by diag(S)*B.
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[out]    X      The solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    rcond  The reciprocal condition number.
 * @param[out]    ferr   The forward error bound. Array of dimension (nrhs).
 * @param[out]    berr   The backward error. Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: if info = i, the leading minor of order i is not
 *                           positive definite; if info = n+1, rcond is too small.
 */
void zpbsvx(
    const char* fact,
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    c128* restrict AB,
    const INT ldab,
    c128* restrict AFB,
    const INT ldafb,
    char* equed,
    f64* restrict S,
    c128* restrict B,
    const INT ldb,
    c128* restrict X,
    const INT ldx,
    f64* rcond,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    f64* restrict rwork,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT equil, nofact, rcequ, upper;
    INT i, infequ, j, j1, j2;
    f64 amax, anorm, bignum = 0.0, scond, smax, smin, smlnum = 0.0;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    equil = (fact[0] == 'E' || fact[0] == 'e');
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (nofact || equil) {
        *equed = 'N';
        rcequ = 0;
    } else {
        rcequ = (*equed == 'Y' || *equed == 'y');
        smlnum = dlamch("S");
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
        xerbla("ZPBSVX", -(*info));
        return;
    }

    if (equil) {
        // Compute row and column scalings to equilibrate the matrix A
        zpbequ(uplo, n, kd, AB, ldab, S, &scond, &amax, &infequ);
        if (infequ == 0) {
            // Equilibrate the matrix
            zlaqhb(uplo, n, kd, AB, ldab, S, scond, amax, equed);
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
        // Compute the Cholesky factorization A = U**H*U or A = L*L**H
        if (upper) {
            for (j = 0; j < n; j++) {
                j1 = (0 > j - kd ? 0 : j - kd);
                cblas_zcopy(j - j1 + 1, &AB[kd - j + j1 + j * ldab], 1,
                            &AFB[kd - j + j1 + j * ldafb], 1);
            }
        } else {
            for (j = 0; j < n; j++) {
                j2 = (n < j + kd + 1 ? n : j + kd + 1);
                cblas_zcopy(j2 - j, &AB[0 + j * ldab], 1, &AFB[0 + j * ldafb], 1);
            }
        }

        zpbtrf(uplo, n, kd, AFB, ldafb, info);

        // Return if INFO is non-zero
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A
    anorm = zlanhb("1", uplo, n, kd, AB, ldab, rwork);

    // Compute the reciprocal of the condition number of A
    zpbcon(uplo, n, kd, AFB, ldafb, anorm, rcond, work, rwork, info);

    // Compute the solution matrix X
    zlacpy("F", n, nrhs, B, ldb, X, ldx);
    zpbtrs(uplo, n, kd, nrhs, AFB, ldafb, X, ldx, info);

    // Use iterative refinement to improve the computed solution and
    // compute error bounds and backward error estimates for it
    zpbrfs(uplo, n, kd, nrhs, AB, ldab, AFB, ldafb, B, ldb, X, ldx,
           ferr, berr, work, rwork, info);

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
    if (*rcond < dlamch("E"))
        *info = n + 1;
}
