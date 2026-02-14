/**
 * @file dposvx.c
 * @brief DPOSVX computes the solution to a real system of linear equations
 *        A * X = B for symmetric positive definite matrices, with equilibration,
 *        condition estimation, and iterative refinement.
 */

#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DPOSVX uses the Cholesky factorization A = U**T*U or A = L*L**T to
 * compute the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric positive definite matrix and X and B
 * are N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also
 * provided.
 *
 * @param[in]     fact   Specifies whether the factored form of the matrix A is
 *                       supplied on entry, and if not, whether to equilibrate.
 *                       = 'F': AF contains the factored form of A; S, equed
 *                              specify equilibration done previously.
 *                       = 'N': The matrix A will be factored without equilibration.
 *                       = 'E': The matrix A will be equilibrated then factored.
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The number of linear equations. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in,out] A      The symmetric matrix A, or equilibrated form.
 *                       Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[in,out] AF     The triangular factor U or L. Array of dimension (ldaf, n).
 * @param[in]     ldaf   The leading dimension of AF. ldaf >= max(1, n).
 * @param[in,out] equed  On entry if fact = 'F', specifies whether A has been
 *                       equilibrated ('N' or 'Y'). On exit, set to 'Y' if
 *                       equilibration was done.
 * @param[in,out] S      Scale factors. Array of dimension (n).
 * @param[in,out] B      Right hand side matrix B (scaled if equed = 'Y').
 *                       Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[out]    X      Solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1, n).
 * @param[out]    rcond  Reciprocal condition number estimate.
 * @param[out]    ferr   Forward error bound. Array of dimension (nrhs).
 * @param[out]    berr   Backward error. Array of dimension (nrhs).
 * @param[out]    work   Workspace, dimension (3*n).
 * @param[out]    iwork  Integer workspace, dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 *                         - > 0, <= n: the leading principal minor of order info is not
 *                           positive; rcond = 0 is returned.
 *                         - = n+1: the matrix is singular to working precision.
 */
void dposvx(
    const char* fact,
    const char* uplo,
    const int n,
    const int nrhs,
    f64* restrict A,
    const int lda,
    f64* restrict AF,
    const int ldaf,
    char* equed,
    f64* restrict S,
    f64* restrict B,
    const int ldb,
    f64* restrict X,
    const int ldx,
    f64* rcond,
    f64* restrict ferr,
    f64* restrict berr,
    f64* restrict work,
    int* restrict iwork,
    int* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    *info = 0;
    int nofact = (fact[0] == 'N' || fact[0] == 'n');
    int equil = (fact[0] == 'E' || fact[0] == 'e');
    int rcequ = 0;
    f64 smlnum = 0.0, bignum = 0.0, scond = ONE, amax;

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
            f64 smin_val = bignum;
            f64 smax_val = ZERO;
            for (int j = 0; j < n; j++) {
                if (S[j] < smin_val) smin_val = S[j];
                if (S[j] > smax_val) smax_val = S[j];
            }
            if (smin_val <= ZERO) {
                *info = -10;
            } else if (n > 0) {
                f64 smin_clamped = smin_val > smlnum ? smin_val : smlnum;
                f64 smax_clamped = smax_val < bignum ? smax_val : bignum;
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
        xerbla("DPOSVX", -(*info));
        return;
    }

    if (equil) {
        // Compute row and column scalings to equilibrate the matrix A.
        int infequ;
        dpoequ(n, A, lda, S, &scond, &amax, &infequ);
        if (infequ == 0) {
            // Equilibrate the matrix.
            dlaqsy(uplo, n, A, lda, S, scond, amax, equed);
            rcequ = (*equed == 'Y' || *equed == 'y');
        }
    }

    // Scale the right hand side.
    if (rcequ) {
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                B[i + j * ldb] = S[i] * B[i + j * ldb];
            }
        }
    }

    if (nofact || equil) {
        // Compute the Cholesky factorization A = U**T*U or A = L*L**T.
        dlacpy(uplo, n, n, A, lda, AF, ldaf);
        dpotrf(uplo, n, AF, ldaf, info);

        // Return if INFO is non-zero.
        if (*info > 0) {
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A.
    f64 anorm = dlansy("1", uplo, n, A, lda, work);

    // Compute the reciprocal of the condition number of A.
    dpocon(uplo, n, AF, ldaf, anorm, rcond, work, iwork, info);

    // Compute the solution matrix X.
    dlacpy("F", n, nrhs, B, ldb, X, ldx);
    dpotrs(uplo, n, nrhs, AF, ldaf, X, ldx, info);

    // Use iterative refinement to improve the computed solution and
    // compute error bounds and backward error estimates for it.
    dporfs(uplo, n, nrhs, A, lda, AF, ldaf, B, ldb, X, ldx,
           ferr, berr, work, iwork, info);

    // Transform the solution matrix X to a solution of the original system.
    if (rcequ) {
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                X[i + j * ldx] = S[i] * X[i + j * ldx];
            }
        }
        for (int j = 0; j < nrhs; j++) {
            ferr[j] = ferr[j] / scond;
        }
    }

    // Set INFO = N+1 if the matrix is singular to working precision.
    if (*rcond < dlamch("E")) {
        *info = n + 1;
    }
}
