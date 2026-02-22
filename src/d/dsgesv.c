/**
 * @file dsgesv.c
 * @brief Mixed precision iterative refinement solver for general linear systems.
 */

#include <math.h>
#include <string.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"
#include "semicolon_lapack_single.h"


/**
 * DSGESV computes the solution to a real system of linear equations
 *    A * X = B,
 * where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
 *
 * DSGESV first attempts to factorize the matrix in SINGLE PRECISION
 * and use this factorization within an iterative refinement procedure
 * to produce a solution with DOUBLE PRECISION normwise backward error
 * quality. If the approach fails the method switches to a DOUBLE PRECISION
 * factorization and solve.
 *
 * @param[in]     n     The number of linear equations, i.e., the order of A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in,out] A     On entry, the N-by-N coefficient matrix A.
 *                      On exit, if iterative refinement succeeded (iter >= 0), A is unchanged.
 *                      If f64 precision factorization was used (iter < 0), A contains
 *                      the factors L and U from A = P*L*U.
 *                      Array of dimension (lda, n).
 * @param[in]     lda   The leading dimension of A. lda >= max(1, n).
 * @param[out]    ipiv  The pivot indices. Array of dimension n, 0-based.
 * @param[in]     B     The N-by-NRHS right hand side matrix B.
 *                      Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[out]    X     If info = 0, the N-by-NRHS solution matrix X.
 *                      Array of dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of X. ldx >= max(1, n).
 * @param[out]    work  Double precision workspace for residual vectors.
 *                      Array of dimension (n, nrhs).
 * @param[out]    swork Single precision workspace for matrix and solutions.
 *                      Array of dimension n*(n+nrhs).
 * @param[out]    iter  Iteration count:
 *                      - < 0: iterative refinement has failed, f64 precision
 *                        factorization has been performed
 *                        - -1 : the routine fell back to full precision for
 *                          implementation- or machine-specific reasons
 *                        - -2 : narrowing the precision induced an overflow,
 *                          the routine fell back to full precision
 *                        - -3 : failure of SGETRF
 *                        - -31: stop the iterative refinement after the 30th
 *                          iterations
 *                      - > 0: iterative refinement has been successfully used.
 *                        Returns the number of iterations
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, argument i had an illegal value
 *                           - > 0: if info = i, U(i-1,i-1) is exactly zero
 */
void dsgesv(
    const INT n,
    const INT nrhs,
    f64* restrict A,
    const INT lda,
    INT* restrict ipiv,
    const f64* restrict B,
    const INT ldb,
    f64* restrict X,
    const INT ldx,
    f64* restrict work,
    float * restrict swork,
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
    float *SA;   // n x n
    float *SX;   // n x nrhs

    *info = 0;
    *iter = 0;

    // Test the input parameters
    if (n < 0) {
        *info = -1;
    } else if (nrhs < 0) {
        *info = -2;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -4;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DSGESV", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        return;
    }

    // Set up pointers into swork
    SA = swork;
    SX = &swork[n * n];

    // Compute some constants
    anrm = dlange("I", n, n, A, lda, work);
    eps = dlamch("E");
    cte = anrm * eps * sqrt((f64)n) * BWDMAX;

    // Convert B from f64 to single precision and store in SX
    dlag2s(n, nrhs, B, ldb, SX, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    // Convert A from f64 to single precision and store in SA
    dlag2s(n, n, A, lda, SA, n, &iinfo);
    if (iinfo != 0) {
        *iter = -2;
        goto fallback;
    }

    // Compute the LU factorization of SA
    sgetrf(n, n, SA, n, ipiv, &iinfo);
    if (iinfo != 0) {
        *iter = -3;
        goto fallback;
    }

    // Solve the system SA*SX = SB
    sgetrs("N", n, nrhs, SA, n, ipiv, SX, n, &iinfo);

    // Convert SX back to f64 precision
    slag2d(n, nrhs, SX, n, X, ldx, &iinfo);

    // Compute R = B - A*X (R is stored in work)
    dlacpy("A", n, nrhs, B, ldb, work, n);
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, nrhs, n, NEGONE, A, lda, X, ldx, ONE, work, n);

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
        sgetrs("N", n, nrhs, SA, n, ipiv, SX, n, &iinfo);

        // Convert SX back to f64 precision and update the current iterate
        slag2d(n, nrhs, SX, n, work, n, &iinfo);

        for (i = 0; i < nrhs; i++) {
            cblas_daxpy(n, ONE, &work[i * n], 1, &X[i * ldx], 1);
        }

        // Compute R = B - A*X
        dlacpy("A", n, nrhs, B, ldb, work, n);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, nrhs, n, NEGONE, A, lda, X, ldx, ONE, work, n);

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
    dgetrf(n, n, A, lda, ipiv, info);
    if (*info != 0) {
        return;
    }

    dlacpy("A", n, nrhs, B, ldb, X, ldx);
    dgetrs("N", n, nrhs, A, lda, ipiv, X, ldx, info);
}
