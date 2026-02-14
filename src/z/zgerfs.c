/**
 * @file zgerfs.c
 * @brief ZGERFS improves the computed solution and provides error bounds.
 */

#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZGERFS improves the computed solution to a system of linear
 * equations and provides error bounds and backward error estimates for
 * the solution.
 *
 * @param[in]     trans  Specifies the form of the system of equations:
 *                       - 'N': A * X = B (No transpose)
 *                       - 'T': A**T * X = B (Transpose)
 *                       - 'C': A**H * X = B (Conjugate transpose)
 * @param[in]     n      The order of the matrix A (n >= 0).
 * @param[in]     nrhs   The number of right hand sides (nrhs >= 0).
 * @param[in]     A      The original N-by-N matrix A. Complex array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A (lda >= max(1,n)).
 * @param[in]     AF     The factors L and U from the factorization A = P*L*U
 *                       as computed by zgetrf. Complex array of dimension (ldaf, n).
 * @param[in]     ldaf   The leading dimension of the array AF (ldaf >= max(1,n)).
 * @param[in]     ipiv   The pivot indices from zgetrf. Array of dimension n.
 * @param[in]     B      The right hand side matrix B. Complex array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B (ldb >= max(1,n)).
 * @param[in,out] X      On entry, the solution matrix X, as computed by zgetrs.
 *                       On exit, the improved solution matrix X.
 *                       Complex array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of the array X (ldx >= max(1,n)).
 * @param[out]    ferr   The estimated forward error bound for each solution vector
 *                       X(j). Real array of dimension nrhs.
 * @param[out]    berr   The componentwise relative backward error of each solution
 *                       vector X(j). Real array of dimension nrhs.
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Real workspace array of dimension (n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 */
void zgerfs(
    const char* trans,
    const int n,
    const int nrhs,
    const c128* const restrict A,
    const int lda,
    const c128* const restrict AF,
    const int ldaf,
    const int* const restrict ipiv,
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
    const c128 ONE = CMPLX(1.0, 0.0);
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    int notran;
    char transn, transt;
    int count, i, j, k, kase, nz;
    f64 eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int linfo;

    *info = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');
    if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else if (lda < (n > 1 ? n : 1)) {
        *info = -5;
    } else if (ldaf < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -10;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -12;
    }

    if (*info != 0) {
        xerbla("ZGERFS", -(*info));
        return;
    }

    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    if (notran) {
        transn = 'N';
        transt = 'C';
    } else {
        transn = 'C';
        transt = 'N';
    }

    nz = n + 1;
    eps = DBL_EPSILON;
    safmin = DBL_MIN;
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        while (1) {
            // Compute residual R = B - op(A) * X
            cblas_zcopy(n, &B[j * ldb], 1, work, 1);
            c128 neg_one = CMPLX(-1.0, 0.0);
            cblas_zgemv(CblasColMajor,
                        notran ? CblasNoTrans :
                        (trans[0] == 'T' || trans[0] == 't') ? CblasTrans : CblasConjTrans,
                        n, n, &neg_one, A, lda, &X[j * ldx], 1, &ONE, work, 1);

            for (i = 0; i < n; i++) {
                rwork[i] = cabs1(B[i + j * ldb]);
            }

            if (notran) {
                for (k = 0; k < n; k++) {
                    xk = cabs1(X[k + j * ldx]);
                    for (i = 0; i < n; i++) {
                        rwork[i] = rwork[i] + cabs1(A[i + k * lda]) * xk;
                    }
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    for (i = 0; i < n; i++) {
                        s = s + cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
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

            zgetrs(trans, n, 1, AF, ldaf, ipiv, work, n, &linfo);
            cblas_zaxpy(n, &ONE, work, 1, &X[j * ldx], 1);
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
                zgetrs(&transt, n, 1, AF, ldaf, ipiv, work, n, &linfo);
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
            } else {
                for (i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
                zgetrs(&transn, n, 1, AF, ldaf, ipiv, work, n, &linfo);
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
