/**
 * @file dpprfs.c
 * @brief DPPRFS improves the computed solution to a system of linear equations with a symmetric positive definite packed matrix.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DPPRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric positive definite
 * and packed, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in]     AP     Array of dimension `n*(n+1)/2`.
 *                       The upper or lower triangle of the symmetric matrix A,
 *                       packed columnwise in a linear array.
 * @param[in]     AFP    Array of dimension `n*(n+1)/2`.
 *                       The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, as computed
 *                       by `dpptrf`, packed columnwise.
 * @param[in]     B      Array of dimension `(ldb,nrhs)`.
 *                       The right hand side matrix B.
 * @param[in]     ldb    The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[in,out] X      Array of dimension `(ldx,nrhs)`.
 *                       On entry, the solution matrix X, as computed by `dpptrs`.
 *                       On exit, the improved solution matrix X.
 * @param[in]     ldx    The leading dimension of the array X. `ldx>=max(1,n)`.
 * @param[out]    ferr   Array of dimension `nrhs`.
 *                       The estimated forward error bound for each solution vector X(j).
 * @param[out]    berr   Array of dimension `nrhs`.
 *                       The componentwise relative backward error of each solution
 *                       vector X(j).
 * @param[out]    work   Array of dimension `3*n`.
 * @param[out]    iwork  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void dpprfs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* restrict AP,
    const f64* restrict AFP,
    const f64* restrict B,
    const INT ldb,
    f64* restrict X,
    const INT ldx,
    f64* restrict ferr,
    f64* restrict berr,
    f64* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const INT ITMAX = 5;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    INT upper;
    INT count, i, ik, j, k, kase, kk, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT locinfo;

    // Test the input parameters
    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -7;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -9;
    }
    if (*info != 0) {
        xerbla("DPPRFS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    // NZ = maximum number of nonzero elements in each row of A, plus 1
    nz = n + 1;
    eps = dlamch("E");
    safmin = dlamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    // Do for each right hand side
    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        // Loop until stopping criterion is satisfied.
        while (1) {
            // Compute residual R = B - A * X
            cblas_dcopy(n, &B[j * ldb], 1, &work[n], 1);
            cblas_dspmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, -ONE, AP, &X[j * ldx], 1, ONE, &work[n], 1);

            // Compute componentwise relative backward error from formula
            //
            // max(i) ( abs(R(i)) / ( abs(A)*abs(X) + abs(B) )(i) )
            //
            // where abs(Z) is the componentwise absolute value of the matrix
            // or vector Z.  If the i-th component of the denominator is less
            // than SAFE2, then SAFE1 is added to the i-th components of the
            // numerator and denominator before dividing.
            for (i = 0; i < n; i++) {
                work[i] = fabs(B[i + j * ldb]);
            }

            // Compute abs(A)*abs(X) + abs(B)
            kk = 0;
            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabs(X[k + j * ldx]);
                    ik = kk;
                    for (i = 0; i < k; i++) {
                        work[i] = work[i] + fabs(AP[ik]) * xk;
                        s = s + fabs(AP[ik]) * fabs(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    work[k] = work[k] + fabs(AP[kk + k]) * xk + s;
                    kk = kk + (k + 1);
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabs(X[k + j * ldx]);
                    work[k] = work[k] + fabs(AP[kk]) * xk;
                    ik = kk + 1;
                    for (i = k + 1; i < n; i++) {
                        work[i] = work[i] + fabs(AP[ik]) * xk;
                        s = s + fabs(AP[ik]) * fabs(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    work[k] = work[k] + s;
                    kk = kk + (n - k);
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (work[i] > safe2) {
                    s = (s > fabs(work[n + i]) / work[i]) ? s : fabs(work[n + i]) / work[i];
                } else {
                    s = (s > (fabs(work[n + i]) + safe1) / (work[i] + safe1)) ?
                        s : (fabs(work[n + i]) + safe1) / (work[i] + safe1);
                }
            }
            berr[j] = s;

            // Test stopping criterion. Continue iterating if
            //    1) The residual BERR(J) is larger than machine epsilon, and
            //    2) BERR(J) decreased by at least a factor of 2 during the
            //       last iteration, and
            //    3) At most ITMAX iterations tried.
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                // Update solution and try again
                dpptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
                cblas_daxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
                lstres = berr[j];
                count = count + 1;
            } else {
                break;
            }
        }

        // Bound error from formula
        //
        // norm(X - XTRUE) / norm(X) .le. FERR =
        // norm( abs(inv(A))*
        //    ( abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) ))) / norm(X)
        //
        // where
        //   norm(Z) is the magnitude of the largest component of Z
        //   inv(A) is the inverse of A
        //   abs(Z) is the componentwise absolute value of the matrix or
        //      vector Z
        //   NZ is the maximum number of nonzeros in any row of A, plus 1
        //   EPS is machine epsilon
        //
        // The i-th component of abs(R)+NZ*EPS*(abs(A)*abs(X)+abs(B))
        // is incremented by SAFE1 if the i-th component of
        // abs(A)*abs(X) + abs(B) is less than SAFE2.
        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabs(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabs(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        // Use DLACN2 to estimate the infinity-norm of the matrix
        //    inv(A) * diag(W),
        // where W = abs(R) + NZ*EPS*( abs(A)*abs(X)+abs(B) )))
        kase = 0;
        while (1) {
            dlacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                // Multiply by diag(W)*inv(A**T)
                dpptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else if (kase == 2) {
                // Multiply by inv(A)*diag(W)
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                dpptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
            }
        }

        // Normalize error
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > fabs(X[i + j * ldx])) ? lstres : fabs(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
