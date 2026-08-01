/**
 * @file cpbrfs.c
 * @brief CPBRFS improves the solution and provides error bounds for a Hermitian positive definite band system.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_complex_single.h"

/**
 * CPBRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is Hermitian positive definite
 * and banded, and provides error bounds and backward error estimates
 * for the solution.
 *
 * @param[in]     uplo
 *                       - `'U'`: Upper triangle of A is stored
 *                       - `'L'`: Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. `n>=0`.
 * @param[in]     kd     The number of superdiagonals of the matrix A if
 *                       `uplo='U'`, or the number of subdiagonals if
 *                       `uplo='L'`. `kd>=0`.
 * @param[in]     nrhs   The number of right hand sides. `nrhs>=0`.
 * @param[in]     AB     Array of dimension (`ldab`, `n`).
 *                       The upper or lower triangle of the Hermitian band
 *                       matrix A, stored in the first `kd+1` rows of the
 *                       array. The j-th column of A is stored in the j-th
 *                       column of the array AB as follows:
 *                       if `uplo='U'`, `AB[kd+i-j + j*ldab] = A(i,j)` for
 *                       `max(0,j-kd)<=i<=j`;
 *                       if `uplo='L'`, `AB[i-j + j*ldab] = A(i,j)` for
 *                       `j<=i<=min(n-1,j+kd)`.
 * @param[in]     ldab   The leading dimension of the array `AB`. `ldab>=kd+1`.
 * @param[in]     AFB    Array of dimension (`ldafb`, `n`).
 *                       The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H of the band
 *                       matrix A as computed by `cpbtrf`, in the same
 *                       storage format as A (see `AB`).
 * @param[in]     ldafb  The leading dimension of the array `AFB`. `ldafb>=kd+1`.
 * @param[in]     B      Array of dimension (`ldb`, `nrhs`).
 *                       The right hand side matrix `B`.
 * @param[in]     ldb    The leading dimension of the array `B`. `ldb>=max(1,n)`.
 * @param[in,out] X      Array of dimension (`ldx`, `nrhs`).
 *                       On entry, the solution matrix X, as computed by
 *                       `cpbtrs`.
 *                       On exit, the improved solution matrix X.
 * @param[in]     ldx    The leading dimension of the array `X`. `ldx>=max(1,n)`.
 * @param[out]    ferr   Array of dimension (`nrhs`).
 *                       The estimated forward error bound for each solution
 *                       vector.
 * @param[out]    berr   Array of dimension (`nrhs`).
 *                       The componentwise relative backward error of each
 *                       solution vector.
 * @param[out]    work   Complex array of dimension `2*n`.
 * @param[out]    rwork  Array of dimension `n`.
 * @param[out]    info
 *                         - `info=0`: successful exit
 *                         - `info<0`: if `info=-i`, the i-th argument had an illegal
 *                           value
 */
void cpbrfs(
    const char* uplo,
    const INT n,
    const INT kd,
    const INT nrhs,
    const c64* restrict AB,
    const INT ldab,
    const c64* restrict AFB,
    const INT ldafb,
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
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT upper;
    INT count, i, j, k, kase, l, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT linfo;

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
