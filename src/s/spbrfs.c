/**
 * @file spbrfs.c
 * @brief SPBRFS improves the solution and provides error bounds for a symmetric positive definite band system.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

#define ITMAX 5

/**
 * SPBRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric positive definite
 * and banded, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AB     The original banded matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in]     AFB    The Cholesky factor from SPBTRF. Array of dimension (ldafb, n).
 * @param[in]     ldafb  The leading dimension of AFB. ldafb >= kd+1.
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
void spbrfs(
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    const f32* restrict AB,
    const INT ldab,
    const f32* restrict AFB,
    const INT ldafb,
    const f32* restrict B,
    const INT ldb,
    f32* restrict X,
    const INT ldx,
    f32* restrict ferr,
    f32* restrict berr,
    f32* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT upper;
    INT count, i, j, k, kase, l, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT info_local;

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kd < 0) {
        *info = -3;
    } else if (nrhs < 0) {
        *info = -4;
    } else if (ldab < kd + 1) {
        *info = -6;
    } else if (ldafb < kd + 1) {
        *info = -8;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -10;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -12;
    }
    if (*info != 0) {
        xerbla("SPBRFS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    // NZ = maximum number of nonzero elements in each row of A, plus 1
    nz = (n + 1 < 2 * kd + 2) ? (n + 1) : (2 * kd + 2);
    eps = slamch("E");
    safmin = slamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    // Do for each right hand side
    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        do {
            // Compute residual R = B - A * X
            cblas_scopy(n, &B[j * ldb], 1, &work[n], 1);
            cblas_ssbmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, kd, -ONE, AB, ldab, &X[j * ldx], 1, ONE, &work[n], 1);

            // Compute componentwise relative backward error
            for (i = 0; i < n; i++) {
                work[i] = fabsf(B[i + j * ldb]);
            }

            // Compute abs(A)*abs(X) + abs(B)
            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabsf(X[k + j * ldx]);
                    l = kd - k;
                    for (i = (0 > k - kd ? 0 : k - kd); i < k; i++) {
                        work[i] = work[i] + fabsf(AB[l + i + k * ldab]) * xk;
                        s = s + fabsf(AB[l + i + k * ldab]) * fabsf(X[i + j * ldx]);
                    }
                    work[k] = work[k] + fabsf(AB[kd + k * ldab]) * xk + s;
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = fabsf(X[k + j * ldx]);
                    work[k] = work[k] + fabsf(AB[0 + k * ldab]) * xk;
                    l = -k;
                    for (i = k + 1; i < (n < k + kd + 1 ? n : k + kd + 1); i++) {
                        work[i] = work[i] + fabsf(AB[l + i + k * ldab]) * xk;
                        s = s + fabsf(AB[l + i + k * ldab]) * fabsf(X[i + j * ldx]);
                    }
                    work[k] = work[k] + s;
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
                // Update solution and try again
                spbtrs(uplo, n, kd, 1, AFB, ldafb, &work[n], n, &info_local);
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
                    spbtrs(uplo, n, kd, 1, AFB, ldafb, &work[n], n, &info_local);
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[n + i] * work[i];
                    }
                } else if (kase == 2) {
                    // Multiply by inv(A)*diag(W)
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[n + i] * work[i];
                    }
                    spbtrs(uplo, n, kd, 1, AFB, ldafb, &work[n], n, &info_local);
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
