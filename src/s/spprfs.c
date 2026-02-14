/**
 * @file spprfs.c
 * @brief SPPRFS improves the computed solution to a system of linear equations with a symmetric positive definite packed matrix.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SPPRFS improves the computed solution to a system of linear
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
 *                       by SPPTRF, packed columnwise.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[in,out] X      On entry, the solution matrix X, as computed by SPPTRS.
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
void spprfs(
    const char* uplo,
    const int n,
    const int nrhs,
    const f32* const restrict AP,
    const f32* const restrict AFP,
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
    // spprfs.f lines 189-198: Parameters
    const int ITMAX = 5;  // spprfs.f line 190
    const f32 ZERO = 0.0f;  // spprfs.f line 192
    const f32 ONE = 1.0f;  // spprfs.f line 194
    const f32 TWO = 2.0f;  // spprfs.f line 196
    const f32 THREE = 3.0f;  // spprfs.f line 198

    // spprfs.f lines 201-206: Local Scalars and Arrays
    int upper;
    int count, i, ik, j, k, kase, kk, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int locinfo;

    // spprfs.f lines 224-240: Test the input parameters
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
        xerbla("SPPRFS", -(*info));
        return;
    }

    // spprfs.f lines 244-250: Quick return if possible
    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {  // spprfs.f line 245: DO 10 J = 1, NRHS
            ferr[j] = ZERO;  // spprfs.f line 246
            berr[j] = ZERO;  // spprfs.f line 247
        }
        return;
    }

    // spprfs.f lines 254-258: NZ = maximum number of nonzero elements in each row of A, plus 1
    nz = n + 1;
    eps = slamch("E");  // Epsilon
    safmin = slamch("S");  // Safe minimum
    safe1 = nz * safmin;  // spprfs.f line 257
    safe2 = safe1 / eps;  // spprfs.f line 258

    // spprfs.f lines 262-415: Do for each right hand side
    for (j = 0; j < nrhs; j++) {  // spprfs.f line 262: DO 140 J = 1, NRHS

        // spprfs.f lines 264-265
        count = 1;
        lstres = THREE;

        // spprfs.f line 266: label 20 - Loop until stopping criterion is satisfied
        while (1) {
            // spprfs.f lines 272-275: Compute residual R = B - A * X
            // CALL DCOPY( N, B( 1, J ), 1, WORK( N+1 ), 1 )
            cblas_scopy(n, &B[j * ldb], 1, &work[n], 1);
            // CALL DSPMV( UPLO, N, -ONE, AP, X( 1, J ), 1, ONE, WORK( N+1 ), 1 )
            cblas_sspmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, -ONE, AP, &X[j * ldx], 1, ONE, &work[n], 1);

            // spprfs.f lines 286-288: Compute componentwise relative backward error
            for (i = 0; i < n; i++) {  // spprfs.f line 286: DO 30 I = 1, N
                work[i] = fabsf(B[i + j * ldb]);  // spprfs.f line 287
            }

            // spprfs.f lines 292-320: Compute abs(A)*abs(X) + abs(B)
            kk = 0;  // spprfs.f line 292: KK = 1 (0-based: kk = 0)
            if (upper) {
                // spprfs.f lines 294-305
                for (k = 0; k < n; k++) {  // spprfs.f line 294: DO 50 K = 1, N
                    s = ZERO;  // spprfs.f line 295
                    xk = fabsf(X[k + j * ldx]);  // spprfs.f line 296
                    ik = kk;  // spprfs.f line 297
                    for (i = 0; i < k; i++) {  // spprfs.f line 298: DO 40 I = 1, K - 1
                        work[i] = work[i] + fabsf(AP[ik]) * xk;  // spprfs.f line 299
                        s = s + fabsf(AP[ik]) * fabsf(X[i + j * ldx]);  // spprfs.f line 300
                        ik = ik + 1;  // spprfs.f line 301
                    }
                    // spprfs.f line 303: WORK( K ) = WORK( K ) + ABS( AP( KK+K-1 ) )*XK + S
                    // In 0-based: kk + k is the diagonal element position
                    work[k] = work[k] + fabsf(AP[kk + k]) * xk + s;
                    kk = kk + (k + 1);  // spprfs.f line 304: KK = KK + K
                }
            } else {
                // spprfs.f lines 307-319
                for (k = 0; k < n; k++) {  // spprfs.f line 307: DO 70 K = 1, N
                    s = ZERO;  // spprfs.f line 308
                    xk = fabsf(X[k + j * ldx]);  // spprfs.f line 309
                    work[k] = work[k] + fabsf(AP[kk]) * xk;  // spprfs.f line 310
                    ik = kk + 1;  // spprfs.f line 311
                    for (i = k + 1; i < n; i++) {  // spprfs.f line 312: DO 60 I = K + 1, N
                        work[i] = work[i] + fabsf(AP[ik]) * xk;  // spprfs.f line 313
                        s = s + fabsf(AP[ik]) * fabsf(X[i + j * ldx]);  // spprfs.f line 314
                        ik = ik + 1;  // spprfs.f line 315
                    }
                    work[k] = work[k] + s;  // spprfs.f line 317
                    kk = kk + (n - k);  // spprfs.f line 318: KK = KK + ( N-K+1 )
                }
            }

            // spprfs.f lines 321-330
            s = ZERO;  // spprfs.f line 321
            for (i = 0; i < n; i++) {  // spprfs.f line 322: DO 80 I = 1, N
                if (work[i] > safe2) {  // spprfs.f line 323
                    s = (s > fabsf(work[n + i]) / work[i]) ? s : fabsf(work[n + i]) / work[i];  // spprfs.f line 324
                } else {
                    s = (s > (fabsf(work[n + i]) + safe1) / (work[i] + safe1)) ?
                        s : (fabsf(work[n + i]) + safe1) / (work[i] + safe1);  // spprfs.f lines 326-327
                }
            }
            berr[j] = s;  // spprfs.f line 330

            // spprfs.f lines 338-348: Test stopping criterion
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                // spprfs.f lines 343-344: Update solution and try again
                spptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
                cblas_saxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
                lstres = berr[j];  // spprfs.f line 345
                count = count + 1;  // spprfs.f line 346
                // GO TO 20 (continue while loop)
            } else {
                break;  // Exit while loop
            }
        }

        // spprfs.f lines 372-378: Bound error from formula
        for (i = 0; i < n; i++) {  // spprfs.f line 372: DO 90 I = 1, N
            if (work[i] > safe2) {  // spprfs.f line 373
                work[i] = fabsf(work[n + i]) + nz * eps * work[i];  // spprfs.f line 374
            } else {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i] + safe1;  // spprfs.f line 376
            }
        }

        // spprfs.f lines 380-404: Use SLACN2 to estimate the infinity-norm
        kase = 0;  // spprfs.f line 380
        while (1) {  // spprfs.f line 381: label 100
            // spprfs.f lines 382-384
            slacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                // spprfs.f lines 390-393: Multiply by diag(W)*inv(A**T)
                spptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
                for (i = 0; i < n; i++) {  // spprfs.f line 391: DO 110 I = 1, N
                    work[n + i] = work[i] * work[n + i];  // spprfs.f line 392
                }
            } else if (kase == 2) {
                // spprfs.f lines 398-401: Multiply by inv(A)*diag(W)
                for (i = 0; i < n; i++) {  // spprfs.f line 398: DO 120 I = 1, N
                    work[n + i] = work[i] * work[n + i];  // spprfs.f line 399
                }
                spptrs(uplo, n, 1, AFP, &work[n], n, &locinfo);
            }
            // GO TO 100 (continue while loop)
        }

        // spprfs.f lines 408-413: Normalize error
        lstres = ZERO;  // spprfs.f line 408
        for (i = 0; i < n; i++) {  // spprfs.f line 409: DO 130 I = 1, N
            lstres = (lstres > fabsf(X[i + j * ldx])) ? lstres : fabsf(X[i + j * ldx]);  // spprfs.f line 410
        }
        if (lstres != ZERO) {  // spprfs.f line 412
            ferr[j] = ferr[j] / lstres;  // spprfs.f line 413
        }
    }
}
