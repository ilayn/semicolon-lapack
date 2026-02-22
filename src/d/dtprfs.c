/**
 * @file dtprfs.c
 * @brief DTPRFS provides error bounds and backward error estimates for packed triangular systems.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DTPRFS provides error bounds and backward error estimates for the
 * solution to a system of linear equations with a triangular packed
 * coefficient matrix.
 *
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     trans  = 'N': A * X = B (No transpose)
 *                        = 'T': A**T * X = B (Transpose)
 *                        = 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     diag   = 'N': A is non-unit triangular
 *                        = 'U': A is unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The packed triangular matrix A. Array of dimension (n*(n+1)/2).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[in]     X      The solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    ferr   The forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr   The backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void dtprfs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const INT n,
    const INT nrhs,
    const f64* restrict AP,
    const f64* restrict B,
    const INT ldb,
    const f64* restrict X,
    const INT ldx,
    f64* restrict ferr,
    f64* restrict berr,
    f64* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;

    INT notran, nounit, upper;
    INT i, j, k, kase, kc, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];

    *info = 0;
    upper = (uplo[0] == 'U' || uplo[0] == 'u');
    notran = (trans[0] == 'N' || trans[0] == 'n');
    nounit = (diag[0] == 'N' || diag[0] == 'n');

    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
        *info = -1;
    } else if (!notran && !(trans[0] == 'T' || trans[0] == 't') &&
               !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -2;
    } else if (!nounit && !(diag[0] == 'U' || diag[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -4;
    } else if (nrhs < 0) {
        *info = -5;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -8;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -10;
    }
    if (*info != 0) {
        xerbla("DTPRFS", -(*info));
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

    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
    CBLAS_TRANSPOSE cblas_trans = notran ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE cblas_transt = notran ? CblasTrans : CblasNoTrans;
    CBLAS_DIAG cblas_diag = nounit ? CblasNonUnit : CblasUnit;

    for (j = 0; j < nrhs; j++) {
        // Compute residual R = B - op(A) * X
        cblas_dcopy(n, &X[j * ldx], 1, &work[n], 1);
        cblas_dtpmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                    n, AP, &work[n], 1);
        cblas_daxpy(n, -ONE, &B[j * ldb], 1, &work[n], 1);

        // Compute componentwise relative backward error
        for (i = 0; i < n; i++) {
            work[i] = fabs(B[i + j * ldb]);
        }

        if (notran) {
            // Compute abs(A)*abs(X) + abs(B)
            if (upper) {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        for (i = 0; i <= k; i++) {
                            work[i] = work[i] + fabs(AP[kc + i]) * xk;
                        }
                        kc = kc + k + 1;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        for (i = 0; i < k; i++) {
                            work[i] = work[i] + fabs(AP[kc + i]) * xk;
                        }
                        work[k] = work[k] + xk;
                        kc = kc + k + 1;
                    }
                }
            } else {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        for (i = k; i < n; i++) {
                            work[i] = work[i] + fabs(AP[kc + i - k]) * xk;
                        }
                        kc = kc + n - k;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        for (i = k + 1; i < n; i++) {
                            work[i] = work[i] + fabs(AP[kc + i - k]) * xk;
                        }
                        work[k] = work[k] + xk;
                        kc = kc + n - k;
                    }
                }
            }
        } else {
            // Compute abs(A**T)*abs(X) + abs(B)
            if (upper) {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        for (i = 0; i <= k; i++) {
                            s = s + fabs(AP[kc + i]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
                        kc = kc + k + 1;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = fabs(X[k + j * ldx]);
                        for (i = 0; i < k; i++) {
                            s = s + fabs(AP[kc + i]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
                        kc = kc + k + 1;
                    }
                }
            } else {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        for (i = k; i < n; i++) {
                            s = s + fabs(AP[kc + i - k]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
                        kc = kc + n - k;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = fabs(X[k + j * ldx]);
                        for (i = k + 1; i < n; i++) {
                            s = s + fabs(AP[kc + i - k]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
                        kc = kc + n - k;
                    }
                }
            }
        }
        s = ZERO;
        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                f64 tmp = fabs(work[n + i]) / work[i];
                if (s < tmp) s = tmp;
            } else {
                f64 tmp = (fabs(work[n + i]) + safe1) / (work[i] + safe1);
                if (s < tmp) s = tmp;
            }
        }
        berr[j] = s;

        // Bound error from formula
        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabs(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabs(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        kase = 0;
        do {
            dlacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    // Multiply by diag(W)*inv(op(A)**T)
                    cblas_dtpsv(CblasColMajor, cblas_uplo, cblas_transt, cblas_diag,
                                n, AP, &work[n], 1);
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                } else {
                    // Multiply by inv(op(A))*diag(W)
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                    cblas_dtpsv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                                n, AP, &work[n], 1);
                }
            }
        } while (kase != 0);

        // Normalize error
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            f64 tmp = fabs(X[i + j * ldx]);
            if (lstres < tmp) lstres = tmp;
        }
        if (lstres != ZERO)
            ferr[j] = ferr[j] / lstres;
    }
}
