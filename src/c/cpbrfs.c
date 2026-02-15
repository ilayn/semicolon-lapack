/**
 * @file cpbrfs.c
 * @brief CPBRFS improves the solution and provides error bounds for a Hermitian positive definite band system.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CPBRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is Hermitian positive definite
 * and banded, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of super-diagonals (if uplo='U') or
 *                       sub-diagonals (if uplo='L'). kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AB     The Hermitian band matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
 * @param[in]     AFB    The Cholesky factor from CPBTRF. Array of dimension (ldafb, n).
 * @param[in]     ldafb  The leading dimension of AFB. ldafb >= kd+1.
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[in,out] X      On entry, the solution matrix X. On exit, the improved solution.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    ferr   The forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr   The backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void cpbrfs(
    const char* uplo,
    const int n,
    const int kd,
    const int nrhs,
    const c64* restrict AB,
    const int ldab,
    const c64* restrict AFB,
    const int ldafb,
    const c64* restrict B,
    const int ldb,
    c64* restrict X,
    const int ldx,
    f32* restrict ferr,
    f32* restrict berr,
    c64* restrict work,
    f32* restrict rwork,
    int* info)
{
    const int ITMAX = 5;
    const f32 ZERO = 0.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    int upper;
    int count, i, j, k, kase, l, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int linfo;

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
        xerbla("CPBRFS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    nz = (n + 1 < 2 * kd + 2) ? (n + 1) : (2 * kd + 2);
    eps = slamch("E");
    safmin = slamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        while (1) {
            cblas_ccopy(n, &B[j * ldb], 1, work, 1);
            c64 neg_one = CMPLXF(-1.0f, 0.0f);
            cblas_chbmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, kd, &neg_one, AB, ldab, &X[j * ldx], 1, &CONE,
                        work, 1);

            for (i = 0; i < n; i++) {
                rwork[i] = cabs1f(B[i + j * ldb]);
            }

            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1f(X[k + j * ldx]);
                    l = kd - k;
                    for (i = (0 > k - kd ? 0 : k - kd); i < k; i++) {
                        rwork[i] = rwork[i] + cabs1f(AB[l + i + k * ldab]) * xk;
                        s = s + cabs1f(AB[l + i + k * ldab]) * cabs1f(X[i + j * ldx]);
                    }
                    rwork[k] = rwork[k] + fabsf(crealf(AB[kd + k * ldab])) * xk + s;
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1f(X[k + j * ldx]);
                    rwork[k] = rwork[k] + fabsf(crealf(AB[0 + k * ldab])) * xk;
                    l = -k;
                    for (i = k + 1; i < (n < k + kd + 1 ? n : k + kd + 1); i++) {
                        rwork[i] = rwork[i] + cabs1f(AB[l + i + k * ldab]) * xk;
                        s = s + cabs1f(AB[l + i + k * ldab]) * cabs1f(X[i + j * ldx]);
                    }
                    rwork[k] = rwork[k] + s;
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    s = (s > cabs1f(work[i]) / rwork[i]) ? s : cabs1f(work[i]) / rwork[i];
                } else {
                    s = (s > (cabs1f(work[i]) + safe1) / (rwork[i] + safe1))
                        ? s : (cabs1f(work[i]) + safe1) / (rwork[i] + safe1);
                }
            }
            berr[j] = s;

            if (!(berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX)) {
                break;
            }

            cpbtrs(uplo, n, kd, 1, AFB, ldafb, work, n, &linfo);
            cblas_caxpy(n, &CONE, work, 1, &X[j * ldx], 1);
            lstres = berr[j];
            count = count + 1;
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
                cpbtrs(uplo, n, kd, 1, AFB, ldafb, work, n, &linfo);
                for (i = 0; i < n; i++) {
                    work[i] = CMPLXF(rwork[i], 0.0f) * work[i];
                }
            } else if (kase == 2) {
                for (i = 0; i < n; i++) {
                    work[i] = CMPLXF(rwork[i], 0.0f) * work[i];
                }
                cpbtrs(uplo, n, kd, 1, AFB, ldafb, work, n, &linfo);
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
