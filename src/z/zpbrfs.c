/**
 * @file zpbrfs.c
 * @brief ZPBRFS improves the solution and provides error bounds for a Hermitian positive definite band system.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPBRFS improves the computed solution to a system of linear
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
 * @param[in]     AFB    The Cholesky factor from ZPBTRF. Array of dimension (ldafb, n).
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
void zpbrfs(
    const char* uplo,
    const int n,
    const int kd,
    const int nrhs,
    const c128* restrict AB,
    const int ldab,
    const c128* restrict AFB,
    const int ldafb,
    const c128* restrict B,
    const int ldb,
    c128* restrict X,
    const int ldx,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    f64* restrict rwork,
    int* info)
{
    const int ITMAX = 5;
    const f64 ZERO = 0.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    int upper;
    int count, i, j, k, kase, l, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
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
        xerbla("ZPBRFS", -(*info));
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
    eps = dlamch("E");
    safmin = dlamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        while (1) {
            cblas_zcopy(n, &B[j * ldb], 1, work, 1);
            c128 neg_one = CMPLX(-1.0, 0.0);
            cblas_zhbmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                        n, kd, &neg_one, AB, ldab, &X[j * ldx], 1, &CONE,
                        work, 1);

            for (i = 0; i < n; i++) {
                rwork[i] = cabs1(B[i + j * ldb]);
            }

            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1(X[k + j * ldx]);
                    l = kd - k;
                    for (i = (0 > k - kd ? 0 : k - kd); i < k; i++) {
                        rwork[i] = rwork[i] + cabs1(AB[l + i + k * ldab]) * xk;
                        s = s + cabs1(AB[l + i + k * ldab]) * cabs1(X[i + j * ldx]);
                    }
                    rwork[k] = rwork[k] + fabs(creal(AB[kd + k * ldab])) * xk + s;
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1(X[k + j * ldx]);
                    rwork[k] = rwork[k] + fabs(creal(AB[0 + k * ldab])) * xk;
                    l = -k;
                    for (i = k + 1; i < (n < k + kd + 1 ? n : k + kd + 1); i++) {
                        rwork[i] = rwork[i] + cabs1(AB[l + i + k * ldab]) * xk;
                        s = s + cabs1(AB[l + i + k * ldab]) * cabs1(X[i + j * ldx]);
                    }
                    rwork[k] = rwork[k] + s;
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    s = (s > cabs1(work[i]) / rwork[i]) ? s : cabs1(work[i]) / rwork[i];
                } else {
                    s = (s > (cabs1(work[i]) + safe1) / (rwork[i] + safe1))
                        ? s : (cabs1(work[i]) + safe1) / (rwork[i] + safe1);
                }
            }
            berr[j] = s;

            if (!(berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX)) {
                break;
            }

            zpbtrs(uplo, n, kd, 1, AFB, ldafb, work, n, &linfo);
            cblas_zaxpy(n, &CONE, work, 1, &X[j * ldx], 1);
            lstres = berr[j];
            count = count + 1;
        }

        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        kase = 0;
        while (1) {
            zlacn2(n, &work[n], work, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                zpbtrs(uplo, n, kd, 1, AFB, ldafb, work, n, &linfo);
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
            } else if (kase == 2) {
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
                zpbtrs(uplo, n, kd, 1, AFB, ldafb, work, n, &linfo);
            }
        }

        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > cabs1(X[i + j * ldx])) ? lstres : cabs1(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
