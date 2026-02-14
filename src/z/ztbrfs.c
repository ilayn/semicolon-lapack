/**
 * @file ztbrfs.c
 * @brief ZTBRFS provides error bounds and backward error estimates for triangular banded systems.
 */

#include <math.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZTBRFS provides error bounds and backward error estimates for the
 * solution to a system of linear equations with a triangular band
 * coefficient matrix.
 *
 * The solution matrix X must be computed by ZTBTRS or some other
 * means before entering this routine. ZTBRFS does not do iterative
 * refinement because doing so cannot improve the backward error.
 *
 * @param[in]     uplo   = 'U': A is upper triangular
 *                        = 'L': A is lower triangular
 * @param[in]     trans  = 'N': A * X = B (No transpose)
 *                        = 'T': A**T * X = B (Transpose)
 *                        = 'C': A**H * X = B (Conjugate transpose)
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
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void ztbrfs(
    const char* uplo,
    const char* trans,
    const char* diag,
    const int n,
    const int kd,
    const int nrhs,
    const double complex* const restrict AB,
    const int ldab,
    const double complex* const restrict B,
    const int ldb,
    const double complex* const restrict X,
    const int ldx,
    double* const restrict ferr,
    double* const restrict berr,
    double complex* const restrict work,
    double* const restrict rwork,
    int* info)
{
    const double ZERO = 0.0;
    const double complex ONE = CMPLX(1.0, 0.0);

    int notran, nounit, upper;
    int i, j, k, kase, nz;
    double eps, lstres, s, safe1, safe2, safmin, xk;
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
        xerbla("ZTBRFS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    const char* transn;
    const char* transt;
    if (notran) {
        transn = "N";
        transt = "C";
    } else {
        transn = "C";
        transt = "N";
    }

    nz = kd + 2;
    eps = dlamch("E");
    safmin = dlamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
    CBLAS_DIAG cblas_diag = nounit ? CblasNonUnit : CblasUnit;
    CBLAS_TRANSPOSE cblas_transn = (transn[0] == 'N') ? CblasNoTrans : CblasConjTrans;
    CBLAS_TRANSPOSE cblas_transt = (transt[0] == 'N') ? CblasNoTrans : CblasConjTrans;

    for (j = 0; j < nrhs; j++) {

        cblas_zcopy(n, &X[j * ldx], 1, work, 1);
        cblas_ztbmv(CblasColMajor, cblas_uplo, cblas_transn, cblas_diag,
                    n, kd, AB, ldab, work, 1);
        double complex neg_one = -ONE;
        cblas_zaxpy(n, &neg_one, &B[j * ldb], 1, work, 1);

        for (i = 0; i < n; i++) {
            rwork[i] = cabs1(B[i + j * ldb]);
        }

        if (notran) {

            if (upper) {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        int istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i <= k; i++) {
                            rwork[i] = rwork[i] +
                                       cabs1(AB[kd + i - k + k * ldab]) * xk;
                        }
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        int istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i < k; i++) {
                            rwork[i] = rwork[i] +
                                       cabs1(AB[kd + i - k + k * ldab]) * xk;
                        }
                        rwork[k] = rwork[k] + xk;
                    }
                }
            } else {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        int iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k; i <= iend; i++) {
                            rwork[i] = rwork[i] +
                                       cabs1(AB[i - k + k * ldab]) * xk;
                        }
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        xk = cabs1(X[k + j * ldx]);
                        int iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k + 1; i <= iend; i++) {
                            rwork[i] = rwork[i] +
                                       cabs1(AB[i - k + k * ldab]) * xk;
                        }
                        rwork[k] = rwork[k] + xk;
                    }
                }
            }
        } else {

            if (upper) {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        int istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i <= k; i++) {
                            s = s + cabs1(AB[kd + i - k + k * ldab]) *
                                    cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = cabs1(X[k + j * ldx]);
                        int istart = (0 > k - kd) ? 0 : k - kd;
                        for (i = istart; i < k; i++) {
                            s = s + cabs1(AB[kd + i - k + k * ldab]) *
                                    cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                    }
                }
            } else {
                if (nounit) {
                    for (k = 0; k < n; k++) {
                        s = ZERO;
                        int iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k; i <= iend; i++) {
                            s = s + cabs1(AB[i - k + k * ldab]) *
                                    cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                    }
                } else {
                    for (k = 0; k < n; k++) {
                        s = cabs1(X[k + j * ldx]);
                        int iend = (n - 1 < k + kd) ? n - 1 : k + kd;
                        for (i = k + 1; i <= iend; i++) {
                            s = s + cabs1(AB[i - k + k * ldab]) *
                                    cabs1(X[i + j * ldx]);
                        }
                        rwork[k] = rwork[k] + s;
                    }
                }
            }
        }
        s = ZERO;
        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                double tmp = cabs1(work[i]) / rwork[i];
                if (s < tmp) s = tmp;
            } else {
                double tmp = (cabs1(work[i]) + safe1) / (rwork[i] + safe1);
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
                    cblas_ztbsv(CblasColMajor, cblas_uplo, cblas_transt, cblas_diag,
                                n, kd, AB, ldab, work, 1);
                    for (i = 0; i < n; i++) {
                        work[i] = CMPLX(rwork[i], 0.0) * work[i];
                    }
                } else {
                    for (i = 0; i < n; i++) {
                        work[i] = CMPLX(rwork[i], 0.0) * work[i];
                    }
                    cblas_ztbsv(CblasColMajor, cblas_uplo, cblas_transn, cblas_diag,
                                n, kd, AB, ldab, work, 1);
                }
            }
        } while (kase != 0);

        lstres = ZERO;
        for (i = 0; i < n; i++) {
            double tmp = cabs1(X[i + j * ldx]);
            if (lstres < tmp) lstres = tmp;
        }
        if (lstres != ZERO)
            ferr[j] = ferr[j] / lstres;
    }
}
