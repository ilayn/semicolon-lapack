/**
 * @file ztprfs.c
 * @brief ZTPRFS provides error bounds and backward error estimates for packed triangular systems.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTPRFS provides error bounds and backward error estimates for the
 * solution to a system of linear equations with a triangular packed
 * coefficient matrix.
 *
 * The solution matrix X must be computed by ZTPTRS or some other
 * means before entering this routine.  ZTPRFS does not do iterative
 * refinement because doing so cannot improve the backward error.
 *
 * @param[in]     uplo   = 'U': A is upper triangular;
 *                        = 'L': A is lower triangular.
 * @param[in]     trans  = 'N': A * X = B (No transpose)
 *                        = 'T': A**T * X = B (Transpose)
 *                        = 'C': A**H * X = B (Conjugate transpose)
 * @param[in]     diag   = 'N': A is non-unit triangular;
 *                        = 'U': A is unit triangular.
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     AP     The packed triangular matrix A. Array of dimension (n*(n+1)/2).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1,n).
 * @param[in]     X      The solution matrix X. Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1,n).
 * @param[out]    ferr   The forward error bound for each solution vector. Array of dimension (nrhs).
 * @param[out]    berr   The backward error for each solution vector. Array of dimension (nrhs).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Double precision workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void ztprfs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int nrhs,
    const c128* restrict AP,
    const c128* restrict B,
    const int ldb,
    const c128* restrict X,
    const int ldx,
    f64* restrict ferr,
    f64* restrict berr,
    c128* restrict work,
    f64* restrict rwork,
    int* info)
{
    const f64 ZERO = 0.0;

    int notran, nounit, upper;
    int i, j, k, kase, kc, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];

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
        xerbla("ZTPRFS", -(*info));
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
    CBLAS_DIAG cblas_diag = nounit ? CblasNonUnit : CblasUnit;
    CBLAS_TRANSPOSE cblas_trans;
    if (trans[0] == 'N' || trans[0] == 'n') {
        cblas_trans = CblasNoTrans;
    } else if (trans[0] == 'T' || trans[0] == 't') {
        cblas_trans = CblasTrans;
    } else {
        cblas_trans = CblasConjTrans;
    }
    CBLAS_TRANSPOSE cblas_transn = notran ? CblasNoTrans : CblasConjTrans;
    CBLAS_TRANSPOSE cblas_transt = notran ? CblasConjTrans : CblasNoTrans;

    for (j = 0; j < nrhs; j++) {

        cblas_zcopy(n, &X[j * ldx], 1, work, 1);
        cblas_ztpmv(CblasColMajor, cblas_uplo, cblas_trans, cblas_diag,
                    n, AP, work, 1);
        {
            const c128 NEG_ONE = CMPLX(-1.0, 0.0);
            cblas_zaxpy(n, &NEG_ONE, &B[j * ldb], 1, work, 1);
        }

        for (i = 0; i < n; i++) {
            rwork[i] = cabs1(B[i + j * ldb]);
        }

        if (notran) {

            if (upper) {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        for (i = 0; i <= k; i++) {
                            rwork[i] = rwork[i] + cabs1(AP[kc + i]) * xk;
                        }
                        kc = kc + k + 1;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        for (i = 0; i < k; i++) {
                            rwork[i] = rwork[i] + cabs1(AP[kc + i]) * xk;
                        }
                        rwork[k] = rwork[k] + xk;
                        kc = kc + k + 1;
                    }
                }
            } else {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        for (i = k; i < n; i++) {
                            rwork[i] = rwork[i] + cabs1(AP[kc + i - k]) * xk;
                        }
                        kc = kc + n - k;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        for (i = k + 1; i < n; i++) {
                            rwork[i] = rwork[i] + cabs1(AP[kc + i - k]) * xk;
                        }
                        rwork[k] = rwork[k] + xk;
                        kc = kc + n - k;
                    }
                }
            }
        } else {

            if (upper) {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        for (i = 0; i <= k; i++) {
                            s = s + cabs1(AP[kc + i]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                        kc = kc + k + 1;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = cabs1(X[k + j * ldx]);
                        for (i = 0; i < k; i++) {
                            s = s + cabs1(AP[kc + i]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                        kc = kc + k + 1;
                    }
                }
            } else {
                kc = 0;
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        for (i = k; i < n; i++) {
                            s = s + cabs1(AP[kc + i - k]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                        kc = kc + n - k;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = cabs1(X[k + j * ldx]);
                        for (i = k + 1; i < n; i++) {
                            s = s + cabs1(AP[kc + i - k]) * cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                        kc = kc + n - k;
                    }
                }
            }
        }
        s = ZERO;
        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                f64 tmp = cabs1(work[i]) / rwork[i];
                if (s < tmp) s = tmp;
            } else {
                f64 tmp = (cabs1(work[i]) + safe1) / (rwork[i] + safe1);
                if (s < tmp) s = tmp;
            }
        }
        berr[j] = s;

        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        kase = 0;
        do {
            zlacn2(n, &work[n], work, &ferr[j], &kase, isave);
            if (kase != 0) {
                if (kase == 1) {
                    cblas_ztpsv(CblasColMajor, cblas_uplo, cblas_transt, cblas_diag,
                                n, AP, work, 1);
                    for (i = 0; i < n; i++) {
                        work[i] = CMPLX(rwork[i], 0.0) * work[i];
                    }
                } else {
                    for (i = 0; i < n; i++) {
                        work[i] = CMPLX(rwork[i], 0.0) * work[i];
                    }
                    cblas_ztpsv(CblasColMajor, cblas_uplo, cblas_transn, cblas_diag,
                                n, AP, work, 1);
                }
            }
        } while (kase != 0);

        lstres = ZERO;
        for (i = 0; i < n; i++) {
            f64 tmp = cabs1(X[i + j * ldx]);
            if (lstres < tmp) lstres = tmp;
        }
        if (lstres != ZERO)
            ferr[j] = ferr[j] / lstres;
    }
}
