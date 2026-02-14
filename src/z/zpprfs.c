/**
 * @file zpprfs.c
 * @brief ZPPRFS improves the computed solution to a system of linear equations with a Hermitian positive definite packed matrix.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPPRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is Hermitian positive definite
 * and packed, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored;
 *                        = 'L': Lower triangle of A is stored.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The upper or lower triangle of the Hermitian matrix A,
 *                       packed columnwise in a linear array.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     AFP    The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H, as computed
 *                       by ZPPTRF, packed columnwise.
 *                       Array of dimension (n*(n+1)/2).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B. ldb >= max(1,n).
 * @param[in,out] X      On entry, the solution matrix X, as computed by ZPPTRS.
 *                       On exit, the improved solution matrix X.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of the array X. ldx >= max(1,n).
 * @param[out]    ferr   The estimated forward error bound for each solution vector X(j).
 *                       Array of dimension (nrhs).
 * @param[out]    berr   The componentwise relative backward error of each solution
 *                       vector X(j). Array of dimension (nrhs).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Double precision workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void zpprfs(
    const char* uplo,
    const int n,
    const int nrhs,
    const c128* const restrict AP,
    const c128* const restrict AFP,
    const c128* const restrict B,
    const int ldb,
    c128* const restrict X,
    const int ldx,
    f64* const restrict ferr,
    f64* const restrict berr,
    c128* const restrict work,
    f64* const restrict rwork,
    int* info)
{
    const int ITMAX = 5;
    const f64 ZERO = 0.0;
    const c128 CONE = CMPLX(1.0, 0.0);
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    int upper;
    int count, i, ik, j, k, kase, kk, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int locinfo;

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
        xerbla("ZPPRFS", -(*info));
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

        while (1) {
            // Compute residual R = B - A * X
            cblas_zcopy(n, &B[j * ldb], 1, work, 1);
            {
                const c128 NEG_CONE = CMPLX(-1.0, 0.0);
                cblas_zhpmv(CblasColMajor, upper ? CblasUpper : CblasLower,
                            n, &NEG_CONE, AP, &X[j * ldx], 1, &CONE, work, 1);
            }

            for (i = 0; i < n; i++) {
                rwork[i] = cabs1(B[i + j * ldb]);
            }

            // Compute abs(A)*abs(X) + abs(B)
            kk = 0;
            if (upper) {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1(X[k + j * ldx]);
                    ik = kk;
                    for (i = 0; i < k; i++) {
                        rwork[i] = rwork[i] + cabs1(AP[ik]) * xk;
                        s = s + cabs1(AP[ik]) * cabs1(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    rwork[k] = rwork[k] + fabs(creal(AP[kk + k])) * xk + s;
                    kk = kk + (k + 1);
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    xk = cabs1(X[k + j * ldx]);
                    rwork[k] = rwork[k] + fabs(creal(AP[kk])) * xk;
                    ik = kk + 1;
                    for (i = k + 1; i < n; i++) {
                        rwork[i] = rwork[i] + cabs1(AP[ik]) * xk;
                        s = s + cabs1(AP[ik]) * cabs1(X[i + j * ldx]);
                        ik = ik + 1;
                    }
                    rwork[k] = rwork[k] + s;
                    kk = kk + (n - k);
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    s = (s > cabs1(work[i]) / rwork[i]) ? s : cabs1(work[i]) / rwork[i];
                } else {
                    s = (s > (cabs1(work[i]) + safe1) / (rwork[i] + safe1)) ?
                        s : (cabs1(work[i]) + safe1) / (rwork[i] + safe1);
                }
            }
            berr[j] = s;

            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                zpptrs(uplo, n, 1, AFP, work, n, &locinfo);
                cblas_zaxpy(n, &CONE, work, 1, &X[j * ldx], 1);
                lstres = berr[j];
                count = count + 1;
            } else {
                break;
            }
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
                zpptrs(uplo, n, 1, AFP, work, n, &locinfo);
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
            } else if (kase == 2) {
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
                zpptrs(uplo, n, 1, AFP, work, n, &locinfo);
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
