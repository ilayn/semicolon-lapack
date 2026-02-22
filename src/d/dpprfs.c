/**
 * @file dpprfs.c
 * @brief DPPRFS improves the computed solution to a system of linear equations with a symmetric positive definite packed matrix.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DPPRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric positive definite
 * and packed, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The upper or lower triangle of the symmetric matrix A,
 *                       packed columnwise in a linear array.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     AFP    The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T, as computed
 *                       by DPPTRF, packed columnwise.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[in,out] X      On entry, the solution matrix X, as computed by DPPTRS.
 *                       On exit, the improved solution matrix X.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of the array X. ldx >= max(1,n).
 * @param[out]    ferr   The estimated forward error bound for each solution vector X(j).
 *                       Array of dimension (nrhs).
 * @param[out]    berr   The componentwise relative backward error of each solution
 *                       vector X(j). Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
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
    // dpprfs.f lines 189-198: Parameters
    const INT ITMAX = 5;  // dpprfs.f line 190
    const f64 ZERO = 0.0;  // dpprfs.f line 192
    const f64 ONE = 1.0;  // dpprfs.f line 194
    const f64 TWO = 2.0;  // dpprfs.f line 196
    const f64 THREE = 3.0;  // dpprfs.f line 198

    // dpprfs.f lines 201-206: Local Scalars and Arrays
    INT upper;
    INT count, i, ik, j, k, kase, kk, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT locinfo;

    // dpprfs.f lines 224-240: Test the input parameters
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

    // dpprfs.f lines 244-250: Quick return if possible
    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {  // dpprfs.f line 245: DO 10 J = 1, NRHS
            ferr[j] = ZERO;  // dpprfs.f line 246
            berr[j] = ZERO;  // dpprfs.f line 247
        }
        return;
    }

    // dpprfs.f lines 254-258: NZ = maximum number of nonzero elements in each row of A, plus 1
    nz = n + 1;
    eps = dlamch("E");  // Epsilon
    safmin = dlamch("S");  // Safe minimum
    safe1 = nz * safmin;  // dpprfs.f line 257
    safe2 = safe1 / eps;  // dpprfs.f line 258

    // dpprfs.f lines 262-415: Do for each right hand side
    for (j = 0; j < nrhs; j++) {  // dpprfs.f line 262: DO 140 J = 1, NRHS

        // dpprfs.f lines 264-265
        count = 1;
        lstres = THREE;

        // dpprfs.f line 266: label 20 - Loop until stopping criterion is satisfied
        while (1) {
            // dpprfs.f lines 272-275: Compute residual R = B - A * X
            // CALL DCOPY( N, B( 1, J ), 1, WORK( N+1 ), 1 )
            cblas_dcopy(n, &B[j * ldb], 1, &work[n], 1);
            // CALL DSPMV( UPLO, N, -ONE, AP, X( 1, J ), 1, ONE, WORK( N+1 ), 1 )
            cblas_dspmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, -ONE, AP, &X[j * ldx], 1, ONE, &work[n], 1);

            // dpprfs.f lines 286-288: Compute componentwise relative backward error
            for (i = 0; i < n; i++) {  // dpprfs.f line 286: DO 30 I = 1, N
                work[i] = fabs(B[i + j * ldb]);  // dpprfs.f line 287
            }

            // dpprfs.f lines 292-320: Compute abs(A)*abs(X) + abs(B)
            kk = 0;  // dpprfs.f line 292: KK = 1 (0-based: kk = 0)
            if (upper) {
                // dpprfs.f lines 294-305
                for (k = 0; k < n; k++) {  // dpprfs.f line 294: DO 50 K = 1, N
                    s = ZERO;  // dpprfs.f line 295
                    xk = fabs(X[k + j * ldx]);  // dpprfs.f line 296
                    ik = kk;  // dpprfs.f line 297
                    for (i = 0; i < k; i++) {  // dpprfs.f line 298: DO 40 I = 1, K - 1
                        work[i] = work[i] + fabs(AP[ik]) * xk;  // dpprfs.f line 299
                        s = s + fabs(AP[ik]) * fabs(X[i + j * ldx]);  // dpprfs.f line 300
                        ik = ik + 1;  // dpprfs.f line 301
                    }
                    // dpprfs.f line 303: WORK( K ) = WORK( K ) + ABS( AP( KK+K-1 ) )*XK + S
                    // In 0-based: kk + k is the diagonal element position
                    work[k] = work[k] + fabs(AP[kk + k]) * xk + s;
                    kk = kk + (k + 1);  // dpprfs.f line 304: KK = KK + K
                }
            } else {
                // dpprfs.f lines 307-319
                for (k = 0; k < n; k++) {  // dpprfs.f line 307: DO 70 K = 1, N
                    s = ZERO;  // dpprfs.f line 308
                    xk = fabs(X[k + j * ldx]);  // dpprfs.f line 309
                    work[k] = work[k] + fabs(AP[kk]) * xk;  // dpprfs.f line 310
                    ik = kk + 1;  // dpprfs.f line 311
                    for (i = k + 1; i < n; i++) {  // dpprfs.f line 312: DO 60 I = K + 1, N
                        work[i] = work[i] + fabs(AP[ik]) * xk;  // dpprfs.f line 313
                        s = s + fabs(AP[ik]) * fabs(X[i + j * ldx]);  // dpprfs.f line 314
                        ik = ik + 1;  // dpprfs.f line 315
                    }
                    work[k] = work[k] + s;  // dpprfs.f line 317
                    kk = kk + (n - k);  // dpprfs.f line 318: KK = KK + ( N-K+1 )
                }
            }

            // dpprfs.f lines 321-330
            s = ZERO;  // dpprfs.f line 321
            for (i = 0; i < n; i++) {  // dpprfs.f line 322: DO 80 I = 1, N
                if (work[i] > safe2) {  // dpprfs.f line 323
                    s = (s > fabs(work[n + i]) / work[i]) ? s : fabs(work[n + i]) / work[i];  // dpprfs.f line 324
                } else {
                    s = (s > (fabs(work[n + i]) + safe1) / (work[i] + safe1)) ?
                        s : (fabs(work[n + i]) + safe1) / (work[i] + safe1);  // dpprfs.f lines 326-327
                }
            }
            berr[j] = s;  // dpprfs.f line 330

            // dpprfs.f lines 338-348: Test stopping criterion
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                // dpprfs.f lines 343-344: Update solution and try again
                dpptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
                cblas_daxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
                lstres = berr[j];  // dpprfs.f line 345
                count = count + 1;  // dpprfs.f line 346
                // GO TO 20 (continue while loop)
            } else {
                break;  // Exit while loop
            }
        }

        // dpprfs.f lines 372-378: Bound error from formula
        for (i = 0; i < n; i++) {  // dpprfs.f line 372: DO 90 I = 1, N
            if (work[i] > safe2) {  // dpprfs.f line 373
                work[i] = fabs(work[n + i]) + nz * eps * work[i];  // dpprfs.f line 374
            } else {
                work[i] = fabs(work[n + i]) + nz * eps * work[i] + safe1;  // dpprfs.f line 376
            }
        }

        // dpprfs.f lines 380-404: Use DLACN2 to estimate the infinity-norm
        kase = 0;  // dpprfs.f line 380
        while (1) {  // dpprfs.f line 381: label 100
            // dpprfs.f lines 382-384
            dlacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                // dpprfs.f lines 390-393: Multiply by diag(W)*inv(A**T)
                dpptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
                for (i = 0; i < n; i++) {  // dpprfs.f line 391: DO 110 I = 1, N
                    work[n + i] = work[i] * work[n + i];  // dpprfs.f line 392
                }
            } else if (kase == 2) {
                // dpprfs.f lines 398-401: Multiply by inv(A)*diag(W)
                for (i = 0; i < n; i++) {  // dpprfs.f line 398: DO 120 I = 1, N
                    work[n + i] = work[i] * work[n + i];  // dpprfs.f line 399
                }
                dpptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
            }
            // GO TO 100 (continue while loop)
        }

        // dpprfs.f lines 408-413: Normalize error
        lstres = ZERO;  // dpprfs.f line 408
        for (i = 0; i < n; i++) {  // dpprfs.f line 409: DO 130 I = 1, N
            lstres = (lstres > fabs(X[i + j * ldx])) ? lstres : fabs(X[i + j * ldx]);  // dpprfs.f line 410
        }
        if (lstres != ZERO) {  // dpprfs.f line 412
            ferr[j] = ferr[j] / lstres;  // dpprfs.f line 413
        }
    }
}
