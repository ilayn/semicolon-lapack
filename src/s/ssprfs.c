/**
 * @file ssprfs.c
 * @brief SSPRFS improves the solution and provides error bounds for packed symmetric systems.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_single.h"

/**
 * SSPRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric indefinite
 * and packed, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     nrhs   The number of right hand sides, i.e., the number of columns
 *                       of the matrices B and X. `nrhs>=0`.
 * @param[in]     AP     Array of dimension `n*(n+1)/2`.
 *                       The upper or lower triangle of the symmetric matrix A,
 *                       packed columnwise in a linear array. The j-th column of A
 *                       is stored in the array AP as follows: if `uplo='U'`,
 *                       `AP[i + j*(j+1)/2] = A(i,j)` for `0<=i<=j`; if `uplo='L'`,
 *                       `AP[i + j*(2*n-j-1)/2] = A(i,j)` for `j<=i<=n-1`.
 * @param[in]     AFP    Array of dimension `n*(n+1)/2`.
 *                       The factored form of the matrix A. AFP contains the block
 *                       diagonal matrix D and the multipliers used to obtain the
 *                       factor U or L from the factorization A = U*D*U**T or
 *                       A = L*D*L**T as computed by `ssptrf`, stored as a packed
 *                       triangular matrix.
 * @param[in]     ipiv   Array of dimension `n`.
 *                       Details of the interchanges and the block structure of D
 *                       as determined by `ssptrf`.
 * @param[in]     B      Array of dimension `(ldb,nrhs)`.
 *                       The right hand side matrix B.
 * @param[in]     ldb    The leading dimension of the array B. `ldb>=max(1,n)`.
 * @param[in,out] X      Array of dimension `(ldx,nrhs)`.
 *                       On entry, the solution matrix X, as computed by `ssptrs`.
 *                       On exit, the improved solution matrix X.
 * @param[in]     ldx    The leading dimension of the array X. `ldx>=max(1,n)`.
 * @param[out]    ferr   Array of dimension `nrhs`.
 *                       The estimated forward error bound for each solution vector
 *                       `X(j)` (the j-th column of the solution matrix X). If
 *                       `xtrue` is the true solution corresponding to `X(j)`,
 *                       `ferr[j]` is an estimated upper bound for the magnitude of
 *                       the largest element in `(X(j)-xtrue)` divided by the
 *                       magnitude of the largest element in `X(j)`. The estimate is
 *                       as reliable as the estimate for `rcond`, and is almost
 *                       always a slight overestimate of the true error.
 * @param[out]    berr   Array of dimension `nrhs`.
 *                       The componentwise relative backward error of each solution
 *                       vector `X(j)` (i.e., the smallest relative change in any
 *                       element of A or B that makes `X(j)` an exact solution).
 * @param[out]    work   Workspace array of dimension `3*n`.
 * @param[out]    iwork  Integer workspace array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void ssprfs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f32* restrict AP,
    const f32* restrict AFP,
    const INT* restrict ipiv,
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
    const INT ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT upper;
    INT count, i, ik, j, k, kase, kk, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
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
