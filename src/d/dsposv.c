/**
 * @file dsposv.c
 * @brief DSPOSV computes the solution to a real system of linear equations
 *        A * X = B for symmetric positive definite matrices using mixed
 *        precision iterative refinement.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"
#include "semicolon_lapack_single.h"

/**
 * DSPOSV computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N symmetric positive definite matrix and X and B
 * are N-by-NRHS matrices.
 *
 * DSPOSV first attempts to factorize the matrix in SINGLE PRECISION
 * and use this factorization within an iterative refinement procedure
 * to produce a solution with DOUBLE PRECISION normwise backward error
 * quality. If the approach fails the method switches to a DOUBLE PRECISION
 * factorization and solve.
 *
 * The iterative refinement process is stopped if
 *     ITER > ITERMAX
 * or for all the RHS we have:
 *     RNRM < SQRT(N)*XNRM*ANRM*EPS*BWDMAX
 * where
 *     o ITER is the number of the current iteration in the iterative
 *       refinement process
 *     o RNRM is the infinity-norm of the residual
 *     o XNRM is the infinity-norm of the solution
 *     o ANRM is the infinity-operator-norm of the matrix A
 *     o EPS is the machine epsilon returned by DLAMCH('Epsilon')
 * The value ITERMAX and BWDMAX are fixed to 30 and 1.0D+00 respectively.
 *
 * @param[in]     uplo  Specifies whether the upper or lower triangular part
 *                      of the symmetric matrix A is stored.
 *                      = 'U': Upper triangle of A is stored
 *                      = 'L': Lower triangle of A is stored
 * @param[in]     n     The number of linear equations, i.e., the order of
 *                      the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides, i.e., the number of
 *                      columns of the matrix B. nrhs >= 0.
 * @param[in,out] A     Double precision array, dimension (lda, n).
 *                      On entry, the symmetric matrix A. If UPLO = 'U', the
 *                      leading N-by-N upper triangular part of A contains the
 *                      upper triangular part of the matrix A. If UPLO = 'L',
 *                      the leading N-by-N lower triangular part of A contains
 *                      the lower triangular part of the matrix A.
 *                      On exit, if iterative refinement has been successfully
 *                      used (info = 0 and iter >= 0) then A is unchanged.
 *                      If f64 precision factorization has been used
 *                      (info = 0 and iter < 0) then the array A contains
 *                      the factor U or L from the Cholesky factorization
 *                      A = U**T*U or A = L*L**T.
 * @param[in]     lda   The leading dimension of the array A. lda >= max(1, n).
 * @param[in]     B     Double precision array, dimension (ldb, nrhs).
 *                      The N-by-NRHS right hand side matrix B.
 * @param[in]     ldb   The leading dimension of the array B. ldb >= max(1, n).
 * @param[out]    X     Double precision array, dimension (ldx, nrhs).
 *                      If info = 0, the N-by-NRHS solution matrix X.
 * @param[in]     ldx   The leading dimension of the array X. ldx >= max(1, n).
 * @param[out]    work  Double precision array, dimension (n, nrhs).
 *                      This array is used to hold the residual vectors.
 * @param[out]    swork Single precision array, dimension (n*(n+nrhs)).
 *                      This array is used to use the single precision matrix
 *                      and the right-hand sides or solutions in single precision.
 * @param[out]    iter  Iteration count:
 *                      - < 0: iterative refinement has failed, f64 precision
 *                        factorization has been performed
 *                        - -1 : the routine fell back to full precision for
 *                          implementation- or machine-specific reasons
 *                        - -2 : narrowing the precision induced an overflow,
 *                          the routine fell back to full precision
 *                        - -3 : failure of SPOTRF
 *                        - -31: stop the iterative refinement after the 30th
 *                          iterations
 *                      - > 0: iterative refinement has been successfully used.
 *                        Returns the number of iterations
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 *                           - > 0: if info = i, the leading principal minor of order i
 *                           of (DOUBLE PRECISION) A is not positive, so the
 *                           factorization could not be completed, and the
 *                           solution has not been computed.
 */
void dsposv(
    const char* uplo,
    const INT n,
    const INT nrhs,
    f64* restrict A,
    const INT lda,
    const f64* restrict B,
    const INT ldb,
    f64* restrict X,
    const INT ldx,
    f64* restrict work,
    float* restrict swork,
    INT* iter,
    INT* info)
{
    const INT ITERMAX = 30;
    const f64 BWDMAX = 1.0;
    const f64 NEGONE = -1.0;
    const f64 ONE = 1.0;

    INT i, iiter, iinfo;
    f64 anrm, cte, eps, rnrm, xnrm;
    INT converged;

    // Pointers into swork
    float* SA;   // n x n
    float* SX;   // n x nrhs
    INT ptsa, ptsx;

    *info = 0;
    *iter = 0;

    // Test the input parameters
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DSPOSV", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        return;
    }

    // Compute some constants
    anrm = dlansy("I", uplo, n, A, lda, work);
    eps = dlamch("E");
    cte = anrm * eps * sqrt((f64)n) * BWDMAX;

    // Set the indices PTSA, PTSX for referencing SA and SX in SWORK.
    ptsa = 0;
    ptsx = n * n;
    SA = &swork[ptsa];
    SX = &swork[ptsx];

    // Convert B from f64 precision to single precision and store in SX.
    dlag2s(n, nrhs, B, ldb, SX, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    // Convert A from f64 precision to single precision and store in SA.
    dlat2s(uplo, n, A, lda, SA, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    // Compute the Cholesky factorization of SA.
    spotrf(uplo, n, SA, n, &iinfo);
    if (iinfo != 0) {
        *iter = -3;
        goto fallback;
    }

    // Solve the system SA*SX = SB.
    spotrs(uplo, n, nrhs, SA, n, SX, n, &iinfo);

    // Convert SX back to f64 precision
    slag2d(n, nrhs, SX, n, X, ldx, &iinfo);

    // Compute R = B - A*X (R is WORK).
    dlacpy("A", n, nrhs, B, ldb, work, n);
    cblas_dsymm(CblasColMajor,
                CblasLeft,
                upper ? CblasUpper : CblasLower,
                n, nrhs, NEGONE, A, lda, X, ldx, ONE, work, n);

    // Check whether the NRHS normwise backward errors satisfy the stopping criterion
    converged = 1;
    for (i = 0; i < nrhs; i++) {
        INT imax_x = cblas_idamax(n, &X[i * ldx], 1);
        INT imax_r = cblas_idamax(n, &work[i * n], 1);
        xnrm = fabs(X[imax_x + i * ldx]);
        rnrm = fabs(work[imax_r + i * n]);
        if (rnrm > xnrm * cte) {
            converged = 0;
            break;
        }
    }

    if (converged) {
        *iter = 0;
        return;
    }

    // Iterative refinement
    for (iiter = 1; iiter <= ITERMAX; iiter++) {
        // Convert R from f64 to single precision and store in SX
        dlag2s(n, nrhs, work, n, SX, n, &iinfo);
        if (iinfo != 0) {
            *iter = -2;
            goto fallback;
        }

        // Solve the system SA*SX = SR
        spotrs(uplo, n, nrhs, SA, n, SX, n, &iinfo);

        // Convert SX back to f64 precision and update the current iterate
        slag2d(n, nrhs, SX, n, work, n, &iinfo);

        for (i = 0; i < nrhs; i++) {
            cblas_daxpy(n, ONE, &work[i * n], 1, &X[i * ldx], 1);
        }

        // Compute R = B - A*X
        dlacpy("A", n, nrhs, B, ldb, work, n);
        cblas_dsymm(CblasColMajor,
                    CblasLeft,
                    upper ? CblasUpper : CblasLower,
                    n, nrhs, NEGONE, A, lda, X, ldx, ONE, work, n);

        // Check whether the NRHS normwise backward errors satisfy the stopping criterion
        converged = 1;
        for (i = 0; i < nrhs; i++) {
            INT imax_x = cblas_idamax(n, &X[i * ldx], 1);
            INT imax_r = cblas_idamax(n, &work[i * n], 1);
            xnrm = fabs(X[imax_x + i * ldx]);
            rnrm = fabs(work[imax_r + i * n]);
            if (rnrm > xnrm * cte) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            *iter = iiter;
            return;
        }
    }

    // Exceeded maximum iterations
    *iter = -ITERMAX - 1;

fallback:
    // Single-precision iterative refinement failed to converge to a
    // satisfactory solution, so we resort to f64 precision.
    dpotrf(uplo, n, A, lda, info);
    if (*info != 0) {
        return;
    }

    dlacpy("A", n, nrhs, B, ldb, X, ldx);
    dpotrs(uplo, n, nrhs, A, lda, X, ldx, info);
}
