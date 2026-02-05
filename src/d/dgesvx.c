/**
 * @file dgesvx.c
 * @brief DGESVX computes the solution to a real system of linear equations
 *        A * X = B, using the LU factorization with equilibration and
 *        iterative refinement.
 */

#include <math.h>
#include <float.h>
#include "semicolon_lapack_double.h"

/**
 * DGESVX uses the LU factorization to compute the solution to a real
 * system of linear equations
 *    A * X = B,
 * where A is an N-by-N matrix and X and B are N-by-NRHS matrices.
 *
 * Error bounds on the solution and a condition estimate are also provided.
 *
 * The following steps are performed:
 *
 * 1. If FACT = "E", real scaling factors are computed to equilibrate
 *    the system:
 *       TRANS = 'N':  diag(R)*A*diag(C)     *inv(diag(C))*X = diag(R)*B
 *       TRANS = 'T': (diag(R)*A*diag(C))**T *inv(diag(R))*X = diag(C)*B
 *       TRANS = 'C': (diag(R)*A*diag(C))**H *inv(diag(R))*X = diag(C)*B
 *
 * 2. If FACT = 'N' or "E", the LU decomposition is used to factor the
 *    matrix A (after equilibration if FACT = "E") as
 *       A = P * L * U
 *
 * 3. If some U(i,i)=0, so that U is exactly singular, then the routine
 *    returns with INFO = i. Otherwise, the factored form of A is used
 *    to estimate the condition number of the matrix A.
 *
 * 4. The system of equations is solved for X using the factored form of A.
 *
 * 5. Iterative refinement is applied to improve the computed solution
 *    matrix and calculate error bounds and backward error estimates for it.
 *
 * 6. If equilibration was used, the matrix X is premultiplied by
 *    diag(C) (if TRANS = "N") or diag(R) (if TRANS = 'T' or "C").
 *
 * @param[in]     fact    'F': AF and IPIV contain the factored form of A.
 *                        'N': The matrix A will be copied to AF and factored.
 *                        'E': The matrix A will be equilibrated if necessary,
 *                             then copied to AF and factored.
 * @param[in]     trans   'N': A * X = B (No transpose)
 *                        'T': A**T * X = B (Transpose)
 *                        'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     n       The number of linear equations (order of A). n >= 0.
 * @param[in]     nrhs    The number of right hand sides. nrhs >= 0.
 * @param[in,out] A       On entry, the N-by-N matrix A.
 *                        On exit, if equilibration was done, A is scaled.
 *                        Array of dimension (lda, n).
 * @param[in]     lda     The leading dimension of A. lda >= max(1, n).
 * @param[in,out] AF      On entry (if fact='F'), contains the LU factors.
 *                        On exit, contains the factors L and U.
 *                        Array of dimension (ldaf, n).
 * @param[in]     ldaf    The leading dimension of AF. ldaf >= max(1, n).
 * @param[in,out] ipiv    Pivot indices from factorization. Array of dimension (n).
 * @param[in,out] equed   On entry (if fact='F'), specifies equilibration done.
 *                        On exit, specifies the form of equilibration:
 *                        'N': No equilibration
 *                        'R': Row equilibration (A := diag(R) * A)
 *                        'C': Column equilibration (A := A * diag(C))
 *                        'B': Both (A := diag(R) * A * diag(C))
 * @param[in,out] R       Row scale factors. Array of dimension (n).
 * @param[in,out] C       Column scale factors. Array of dimension (n).
 * @param[in,out] B       On entry, the N-by-NRHS right hand side matrix B.
 *                        On exit, if equilibration was done, B is scaled.
 *                        Array of dimension (ldb, nrhs).
 * @param[in]     ldb     The leading dimension of B. ldb >= max(1, n).
 * @param[out]    X       The N-by-NRHS solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx     The leading dimension of X. ldx >= max(1, n).
 * @param[out]    rcond   Reciprocal condition number estimate.
 * @param[out]    ferr    Forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr    Backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work    Workspace array of dimension (4*n).
 *                        On exit, work[0] contains the reciprocal pivot growth factor.
 * @param[out]    iwork   Integer workspace array of dimension (n).
 * @param[out]    info    = 0: successful exit
 *                        < 0: if info = -i, the i-th argument had an illegal value
 *                        > 0: if info = i, U(i,i) is exactly zero (1-based).
 *                             if info = n+1, U is nonsingular but RCOND < machine precision.
 */
void dgesvx(
    const char* fact,
    const char* trans,
    const int n,
    const int nrhs,
    double * const restrict A,
    const int lda,
    double * const restrict AF,
    const int ldaf,
    int * const restrict ipiv,
    char *equed,
    double * const restrict R,
    double * const restrict C,
    double * const restrict B,
    const int ldb,
    double * const restrict X,
    const int ldx,
    double *rcond,
    double * const restrict ferr,
    double * const restrict berr,
    double * const restrict work,
    int * const restrict iwork,
    int *info)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;

    int i, j, infequ;
    double amax, anorm, bignum, colcnd, rcmax, rcmin, rowcnd, rpvgrw, smlnum;
    char norm;
    int nofact, equil, notran, rowequ, colequ;

    *info = 0;
    nofact = (fact[0] == 'N' || fact[0] == 'n');
    equil = (fact[0] == 'E' || fact[0] == 'e');
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (nofact || equil) {
        *equed = 'N';
        rowequ = 0;
        colequ = 0;
    } else {
        rowequ = (*equed == 'R' || *equed == 'r' || *equed == 'B' || *equed == 'b');
        colequ = (*equed == 'C' || *equed == 'c' || *equed == 'B' || *equed == 'b');
        smlnum = DBL_MIN;
        bignum = ONE / smlnum;
    }

    // Test the input parameters
    if (!nofact && !equil && fact[0] != 'F' && fact[0] != 'f') {
        *info = -1;
    } else if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -2;
    } else if (n < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -6;
    } else if (ldaf < (n > 1 ? n : 1)) {
        *info = -8;
    } else if (fact[0] == 'F' || fact[0] == 'f') {
        if (!rowequ && !colequ && *equed != 'N' && *equed != 'n') {
            *info = -10;
        }
    }

    if (*info == 0) {
        if (rowequ) {
            rcmin = bignum;
            rcmax = ZERO;
            for (j = 0; j < n; j++) {
                if (R[j] < rcmin) rcmin = R[j];
                if (R[j] > rcmax) rcmax = R[j];
            }
            if (rcmin <= ZERO) {
                *info = -11;
            } else if (n > 0) {
                double rmin = (rcmin > smlnum) ? rcmin : smlnum;
                double rmax = (rcmax < bignum) ? rcmax : bignum;
                rowcnd = rmin / rmax;
            } else {
                rowcnd = ONE;
            }
        }
        if (colequ && *info == 0) {
            rcmin = bignum;
            rcmax = ZERO;
            for (j = 0; j < n; j++) {
                if (C[j] < rcmin) rcmin = C[j];
                if (C[j] > rcmax) rcmax = C[j];
            }
            if (rcmin <= ZERO) {
                *info = -12;
            } else if (n > 0) {
                double cmin = (rcmin > smlnum) ? rcmin : smlnum;
                double cmax = (rcmax < bignum) ? rcmax : bignum;
                colcnd = cmin / cmax;
            } else {
                colcnd = ONE;
            }
        }
        if (*info == 0) {
            if (ldb < (n > 1 ? n : 1)) {
                *info = -14;
            } else if (ldx < (n > 1 ? n : 1)) {
                *info = -16;
            }
        }
    }

    if (*info != 0) {
        xerbla("DGESVX", -(*info));
        return;
    }

    if (equil) {
        // Compute row and column scalings to equilibrate the matrix A
        dgeequ(n, n, A, lda, R, C, &rowcnd, &colcnd, &amax, &infequ);
        if (infequ == 0) {
            // Equilibrate the matrix
            dlaqge(n, n, A, lda, R, C, rowcnd, colcnd, amax, equed);
            rowequ = (*equed == 'R' || *equed == 'r' || *equed == 'B' || *equed == 'b');
            colequ = (*equed == 'C' || *equed == 'c' || *equed == 'B' || *equed == 'b');
        }
    }

    // Scale the right hand side
    if (notran) {
        if (rowequ) {
            for (j = 0; j < nrhs; j++) {
                for (i = 0; i < n; i++) {
                    B[i + j * ldb] = R[i] * B[i + j * ldb];
                }
            }
        }
    } else if (colequ) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                B[i + j * ldb] = C[i] * B[i + j * ldb];
            }
        }
    }

    if (nofact || equil) {
        // Compute the LU factorization of A
        dlacpy("F", n, n, A, lda, AF, ldaf);
        dgetrf(n, n, AF, ldaf, ipiv, info);

        // Return if INFO is non-zero
        if (*info > 0) {
            // Compute the reciprocal pivot growth factor of the
            // leading rank-deficient INFO columns of A
            rpvgrw = dlantr("M", "U", "N", *info, *info, AF, ldaf, work);
            if (rpvgrw == ZERO) {
                rpvgrw = ONE;
            } else {
                rpvgrw = dlange("M", n, *info, A, lda, work) / rpvgrw;
            }
            work[0] = rpvgrw;
            *rcond = ZERO;
            return;
        }
    }

    // Compute the norm of the matrix A and the reciprocal pivot growth factor RPVGRW
    if (notran) {
        norm = '1';
    } else {
        norm = 'I';
    }
    anorm = dlange(&norm, n, n, A, lda, work);
    rpvgrw = dlantr("M", "U", "N", n, n, AF, ldaf, work);
    if (rpvgrw == ZERO) {
        rpvgrw = ONE;
    } else {
        rpvgrw = dlange("M", n, n, A, lda, work) / rpvgrw;
    }

    // Compute the reciprocal of the condition number of A
    dgecon(&norm, n, AF, ldaf, anorm, rcond, work, iwork, info);

    // Compute the solution matrix X
    dlacpy("F", n, nrhs, B, ldb, X, ldx);
    dgetrs(trans, n, nrhs, AF, ldaf, ipiv, X, ldx, info);

    // Use iterative refinement to improve the computed solution and
    // compute error bounds and backward error estimates for it
    dgerfs(trans, n, nrhs, A, lda, AF, ldaf, ipiv, B, ldb, X, ldx,
           ferr, berr, work, iwork, info);

    // Transform the solution matrix X to a solution of the original system
    if (notran) {
        if (colequ) {
            for (j = 0; j < nrhs; j++) {
                for (i = 0; i < n; i++) {
                    X[i + j * ldx] = C[i] * X[i + j * ldx];
                }
            }
            for (j = 0; j < nrhs; j++) {
                ferr[j] = ferr[j] / colcnd;
            }
        }
    } else if (rowequ) {
        for (j = 0; j < nrhs; j++) {
            for (i = 0; i < n; i++) {
                X[i + j * ldx] = R[i] * X[i + j * ldx];
            }
        }
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ferr[j] / rowcnd;
        }
    }

    work[0] = rpvgrw;

    // Set INFO = N+1 if the matrix is singular to working precision
    if (*rcond < DBL_EPSILON) {
        *info = n + 1;
    }
}
