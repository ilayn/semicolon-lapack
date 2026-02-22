/**
 * @file dtbrfs.c
 * @brief DTBRFS provides error bounds and backward error estimates for triangular banded systems.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "semicolon_lapack_double.h"

/**
 * DTBRFS provides error bounds and backward error estimates for the
 * solution to a system of linear equations with a triangular band
 * coefficient matrix.
 *
 * The solution matrix X must be computed by DTBTRS or some other
 * means before entering this routine. DTBRFS does not do iterative
 * refinement because doing so cannot improve the backward error.
 *
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     trans  = 'N': A * X = B (No transpose)
 *                        = 'T': A**T * X = B (Transpose)
 *                        = 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     diag   = 'N': A is non-unit triangular
 *                        = 'U': A is unit triangular
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     kd     The number of superdiagonals or subdiagonals. kd >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AB     The triangular band matrix A. Array of dimension (ldab, n).
 * @param[in]     ldab   The leading dimension of AB. ldab >= kd+1.
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
void dtbrfs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const INT n,
    const INT kd,
    const INT nrhs,
    const f64* restrict AB,
    const INT ldab,
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
    INT i, j, k, kase, nz;
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
    } else if (kd < 0) {
        *info = -5;
    } else if (nrhs < 0) {
        *info = -6;
    } else if (ldab < kd + 1) {
        *info = -8;
    } else if (ldb < (1 > n ? 1 : n)) {
        *info = -10;
    } else if (ldx < (1 > n ? 1 : n)) {
        *info = -12;
    }
    if (*info != 0) {
        xerbla("DTBRFS", -(*info));
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
    nz = kd + 2;
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
        cblas_dtbmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                    n, kd, AB, ldab, &work[n], 1);
        cblas_daxpy(n, -ONE, &B[j * ldb], 1, &work[n], 1);

        // Compute componentwise relative backward error
        for (i = 0; i < n; i++) {
            work[i] = fabs(B[i + j * ldb]);
        }

        if (notran) {
            // Compute abs(A)*abs(X) + abs(B)
            if (upper) {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        // Fortran: DO I = MAX(1,K-KD), K -> 0-based: i from max(0,k-kd) to k
                        INT istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i <= k; i++) {
                            // AB(kd+1+i-k, k) in Fortran 1-based -> AB[kd+i-k + k*ldab] in 0-based
                            work[i] = work[i] + fabs(AB[kd + i - k + k * ldab]) * xk;
                        }
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        INT istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i < k; i++) {
                            work[i] = work[i] + fabs(AB[kd + i - k + k * ldab]) * xk;
                        }
                        work[k] = work[k] + xk;
                    }
                }
            } else {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        // Fortran: DO I = K, MIN(N,K+KD) -> 0-based: i from k to min(n-1,k+kd)
                        INT iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k; i <= iend; i++) {
                            // AB(1+i-k, k) in Fortran 1-based -> AB[i-k + k*ldab] in 0-based
                            work[i] = work[i] + fabs(AB[i - k + k * ldab]) * xk;
                        }
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = fabs(X[k + j * ldx]);
                        INT iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k + 1; i <= iend; i++) {
                            work[i] = work[i] + fabs(AB[i - k + k * ldab]) * xk;
                        }
                        work[k] = work[k] + xk;
                    }
                }
            }
        } else {
            // Compute abs(A**T)*abs(X) + abs(B)
            if (upper) {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        INT istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i <= k; i++) {
                            s = s + fabs(AB[kd + i - k + k * ldab]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = fabs(X[k + j * ldx]);
                        INT istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i < k; i++) {
                            s = s + fabs(AB[kd + i - k + k * ldab]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
                    }
                }
            } else {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        INT iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k; i <= iend; i++) {
                            s = s + fabs(AB[i - k + k * ldab]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = fabs(X[k + j * ldx]);
                        INT iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k + 1; i <= iend; i++) {
                            s = s + fabs(AB[i - k + k * ldab]) * fabs(X[i + j * ldx]);
                        }
                        work[k] = work[k] + s;
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
                    cblas_dtbsv(CblasColMajor, cblas_uplo, cblas_transt, cblas_diag,
                                n, kd, AB, ldab, &work[n], 1);
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                } else {
                    // Multiply by inv(op(A))*diag(W)
                    for (i = 0; i < n; i++) {
                        work[n + i] = work[i] * work[n + i];
                    }
                    cblas_dtbsv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                                n, kd, AB, ldab, &work[n], 1);
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
