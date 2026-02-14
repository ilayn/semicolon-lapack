/**
 * @file ssprfs.c
 * @brief SSPRFS improves the solution and provides error bounds for packed symmetric systems.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SSPRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric indefinite
 * and packed, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The original packed matrix A. Array of dimension (n*(n+1)/2).
 * @param[in]     AFP    The factored form of A from SSPTRF. Array of dimension (n*(n+1)/2).
 * @param[in]     ipiv   The pivot indices from SSPTRF. Array of dimension (n).
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
void ssprfs(
    const char* uplo,
    const int n,
    const int nrhs,
    const f32* const restrict AP,
    const f32* const restrict AFP,
    const int* const restrict ipiv,
    const f32* const restrict B,
    const int ldb,
    f32* const restrict X,
    const int ldx,
    f32* const restrict ferr,
    f32* const restrict berr,
    f32* const restrict work,
    int* const restrict iwork,
    int* info)
{
    const int ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    int upper;
    int count, i, ik, j, k, kase, kk, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int info_local;

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
        xerbla("SSPRFS", -(*info));
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
    eps = slamch("E");
    safmin = slamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        do {
            // Compute residual R = B - A * X
            cblas_scopy(n, &B[j * ldb], 1, &work[n], 1);
            cblas_sspmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, -ONE, AP, &X[j * ldx], 1, ONE, &work[n], 1);

            // Compute componentwise relative backward error
            for (i = 0; i < n; i++) {
                work[i] = fabsf(B[i + j * ldb]);
            }

            // Compute abs(A)*abs(X) + abs(B)
            kk = 0;
            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabsf(X[k + j * ldx]);
                    ik = kk;
                    for (i = 0; i < k; i++) {
                        work[i] = work[i] + fabsf(AP[ik]) * xk;
                        s = s + fabsf(AP[ik]) * fabsf(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    work[k] = work[k] + fabsf(AP[kk + k]) * xk + s;
                    kk = kk + k + 1;
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabsf(X[k + j * ldx]);
                    work[k] = work[k] + fabsf(AP[kk]) * xk;
                    ik = kk + 1;
                    for (i = k + 1; i < n; i++) {
                        work[i] = work[i] + fabsf(AP[ik]) * xk;
                        s = s + fabsf(AP[ik]) * fabsf(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    work[k] = work[k] + s;
                    kk = kk + (n - k);
                }
            }
            s = ZERO;
            for (i = 0; i < n; i++) {
                if (work[i] > safe2) {
                    f32 tmp = fabsf(work[n + i]) / work[i];
                    if (s < tmp) s = tmp;
                } else {
                    f32 tmp = (fabsf(work[n + i]) + safe1) / (work[i] + safe1);
                    if (s < tmp) s = tmp;
                }
            }
            berr[j] = s;

            // Test stopping criterion
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                ssptrs(uplo, n, 1, AFP, ipiv, &work[n], n, &info_local);
                cblas_saxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
                lstres = berr[j];
                count = count + 1;
            } else {
                break;
            }
        } while (1);

        // Bound error from formula
        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        kase = 0;
        do {
            slacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    // Multiply by diag(W)*inv(A**T)
                    ssptrs(uplo, n, 1, AFP, ipiv, &work[n], n, &info_local);
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                } else if (kase == 2) {
                    // Multiply by inv(A)*diag(W)
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                    ssptrs(uplo, n, 1, AFP, ipiv, &work[n], n, &info_local);
                }
            }
        } while (kase != 0);

        // Normalize error
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            f32 tmp = fabsf(X[i + j * ldx]);
            if (lstres < tmp) lstres = tmp;
        }
        if (lstres != ZERO)
            ferr[j] = ferr[j] / lstres;
    }
}
