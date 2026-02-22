/**
 * @file csprfs.c
 * @brief CSPRFS improves the solution and provides error bounds for packed symmetric systems.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CSPRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric indefinite
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
 * @param[in]     AFP    The factored form of the matrix A. AFP contains the block
 *                       diagonal matrix D and the multipliers used to obtain the
 *                       factor U or L from the factorization A = U*D*U**T or
 *                       A = L*D*L**T as computed by CSPTRF, stored as a packed
 *                       triangular matrix.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     ipiv   Details of the interchanges and the block structure of D
 *                       as determined by CSPTRF. Array of dimension (n).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[in,out] X      On entry, the solution matrix X, as computed by CSPTRS.
 *                       On exit, the improved solution matrix X.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of the array X. ldx >= max(1,n).
 * @param[out]    ferr   The estimated forward error bound for each solution vector X(j).
 *                       Array of dimension (nrhs).
 * @param[out]    berr   The componentwise relative backward error of each solution
 *                       vector X(j). Array of dimension (nrhs).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Single precision workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void csprfs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const c64* restrict AP,
    const c64* restrict AFP,
    const INT* restrict ipiv,
    const c64* restrict B,
    const INT ldb,
    c64* restrict X,
    const INT ldx,
    f32* restrict ferr,
    f32* restrict berr,
    c64* restrict work,
    f32* restrict rwork,
    INT* info)
{
    const INT ITMAX = 5;
    const f32 ZERO = 0.0f;
    const c64 ONE = CMPLXF(1.0f, 0.0f);
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT upper;
    INT count, i, ik, j, k, kase, kk, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT locinfo;

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
        xerbla("CSPRFS", -(*info));
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

        while (1) {

            cblas_ccopy(n, &B[j * ldb], 1, work, 1);
            {
                const c64 NEG_ONE = CMPLXF(-1.0f, 0.0f);
                cspmv(uplo, n, NEG_ONE, AP, &X[j * ldx], 1, ONE, work, 1);
            }

            for (i = 0; i < n; i++) {
                rwork[i] = cabs1f(B[i + j * ldb]);
            }

            kk = 0;
            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1f(X[k + j * ldx]);
                    ik = kk;
                    for (i = 0; i < k; i++) {
                        rwork[i] = rwork[i] + cabs1f(AP[ik]) * xk;
                        s = s + cabs1f(AP[ik]) * cabs1f(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    rwork[k] = rwork[k] + cabs1f(AP[kk + k]) * xk + s;
                    kk = kk + k + 1;
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1f(X[k + j * ldx]);
                    rwork[k] = rwork[k] + cabs1f(AP[kk]) * xk;
                    ik = kk + 1;
                    for (i = k + 1; i < n; i++) {
                        rwork[i] = rwork[i] + cabs1f(AP[ik]) * xk;
                        s = s + cabs1f(AP[ik]) * cabs1f(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    rwork[k] = rwork[k] + s;
                    kk = kk + (n - k);
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    s = (s > cabs1f(work[i]) / rwork[i]) ? s : cabs1f(work[i]) / rwork[i];
                } else {
                    s = (s > (cabs1f(work[i]) + safe1) / (rwork[i] + safe1)) ?
                        s : (cabs1f(work[i]) + safe1) / (rwork[i] + safe1);
                }
            }
            berr[j] = s;

            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                csptrs(uplo, n, 1, AFP, ipiv, work, n, &locinfo);
                cblas_caxpy(n, &ONE, work, 1, &X[j * ldx], 1);
                lstres = berr[j];
                count = count + 1;
            } else {
                break;
            }
        }

        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        kase = 0;
        while (1) {
            clacn2(n, &work[n], work, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                csptrs(uplo, n, 1, AFP, ipiv, work, n, &locinfo);
                for (i = 0; i < n; i++) {
                    work[i] = CMPLXF(rwork[i], 0.0f) * work[i];
                }
            } else if (kase == 2) {
                for (i = 0; i < n; i++) {
                    work[i] = CMPLXF(rwork[i], 0.0f) * work[i];
                }
                csptrs(uplo, n, 1, AFP, ipiv, work, n, &locinfo);
            }
        }

        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > cabs1f(X[i + j * ldx])) ? lstres : cabs1f(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
