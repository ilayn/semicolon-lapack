/**
 * @file dsprfs.c
 * @brief DSPRFS improves the solution and provides error bounds for packed symmetric systems.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DSPRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric indefinite
 * and packed, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The original packed matrix A. Array of dimension (n*(n+1)/2).
 * @param[in]     AFP    The factored form of A from DSPTRF. Array of dimension (n*(n+1)/2).
 * @param[in]     ipiv   The pivot indices from DSPTRF. Array of dimension (n).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[in,out] X      On entry, the solution matrix X. On exit, the improved solution.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    ferr   The forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr   The backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dsprfs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* restrict AP,
    const f64* restrict AFP,
    const INT* restrict ipiv,
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
    INT info_local;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DSPRFS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    nz = n + 1;
    eps = dlamch("E");
    safmin = dlamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        do {
            // Compute residual R = B - A * X
            cblas_dcopy(n, &B[j * ldb], 1, &work[n], 1);
            cblas_dspmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, -ONE, AP, &X[j * ldx], 1, ONE, &work[n], 1);

            // Compute componentwise relative backward error
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
                    kk = kk + k + 1;
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
                    f64 tmp = fabs(work[n + i]) / work[i];
                    if (s < tmp) s = tmp;
                } else {
                    f64 tmp = (fabs(work[n + i]) + safe1) / (work[i] + safe1);
                    if (s < tmp) s = tmp;
                }
            }
            berr[j] = s;

            // Test stopping criterion
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                dsptrs(uplo, n, 1, AFP, ipiv, &work[n], n, &info_local);
                cblas_daxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
                lstres = berr[j];
                count = count + 1;
            } else {
                break;
            }
        } while (1);

        // Bound error from formula
        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabs(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabs(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        kase = 0;
        do {
            dlacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    // Multiply by diag(W)*inv(A**T)
                    dsptrs(uplo, n, 1, AFP, ipiv, &work[n], n, &info_local);
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                } else if (kase == 2) {
                    // Multiply by inv(A)*diag(W)
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                    dsptrs(uplo, n, 1, AFP, ipiv, &work[n], n, &info_local);
                }
            }
        } while (kase != 0);

        // Normalize error
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            f64 tmp = fabs(X[i + j * ldx]);
            if (lstres < tmp) lstres = tmp;
        }
        if (lstres != ZERO)
            ferr[j] = ferr[j] / lstres;
    }
}
