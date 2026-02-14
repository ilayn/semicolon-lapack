/**
 * @file zporfs.c
 * @brief ZPORFS improves the computed solution and provides error bounds
 *        for Hermitian positive definite systems.
 */

#include <complex.h>
#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZPORFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is Hermitian positive definite,
 * and provides error bounds and backward error estimates for the
 * solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     A      The Hermitian matrix A. Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[in]     AF     The triangular factor U or L from the Cholesky
 *                       factorization A = U**H*U or A = L*L**H.
 *                       Array of dimension (ldaf, n).
 * @param[in]     ldaf   The leading dimension of AF. ldaf >= max(1, n).
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] X      On entry, the solution matrix X.
 *                       On exit, the improved solution matrix X.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of X. ldx >= max(1, n).
 * @param[out]    ferr   The estimated forward error bound for each solution
 *                       vector. Array of dimension (nrhs).
 * @param[out]    berr   The componentwise relative backward error.
 *                       Array of dimension (nrhs).
 * @param[out]    work   Complex workspace array of dimension (2*n).
 * @param[out]    rwork  Double precision workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void zporfs(
    const char* uplo,
    const int n,
    const int nrhs,
    const c128* const restrict A,
    const int lda,
    const c128* const restrict AF,
    const int ldaf,
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
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;
    const c128 ONE = CMPLX(1.0, 0.0);

    *info = 0;
    int upper = (uplo[0] == 'U' || uplo[0] == 'u');
    if (!upper && !(uplo[0] == 'L' || uplo[0] == 'l')) {
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
        *info = -9;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("ZPORFS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        for (int j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    // NZ = maximum number of nonzero elements in each row of A, plus 1
    int nz = n + 1;
    f64 eps = dlamch("E");
    f64 safmin = dlamch("S");
    f64 safe1 = nz * safmin;
    f64 safe2 = safe1 / eps;

    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;
    const c128 NEG_ONE = CMPLX(-1.0, 0.0);

    // Do for each right hand side
    for (int j = 0; j < nrhs; j++) {
        int count = 1;
        f64 lstres = THREE;

        for (;;) {
            // Compute residual R = B - A * X
            cblas_zcopy(n, &B[j * ldb], 1, work, 1);
            cblas_zhemv(CblasColMajor, cblas_uplo, n, &NEG_ONE, A, lda,
                        &X[j * ldx], 1, &ONE, work, 1);

            // Compute componentwise relative backward error
            // max(i) ( |R(i)| / ( |A|*|X| + |B| )(i) )
            for (int i = 0; i < n; i++) {
                rwork[i] = cabs1(B[i + j * ldb]);
            }

            // Compute |A|*|X| + |B|
            if (upper) {
                for (int k = 0; k < n; k++) {
                    f64 s = ZERO;
                    f64 xk = cabs1(X[k + j * ldx]);
                    for (int i = 0; i < k; i++) {
                        rwork[i] += cabs1(A[i + k * lda]) * xk;
                        s += cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                    }
                    rwork[k] += fabs(creal(A[k + k * lda])) * xk + s;
                }
            } else {
                for (int k = 0; k < n; k++) {
                    f64 s = ZERO;
                    f64 xk = cabs1(X[k + j * ldx]);
                    rwork[k] += fabs(creal(A[k + k * lda])) * xk;
                    for (int i = k + 1; i < n; i++) {
                        rwork[i] += cabs1(A[i + k * lda]) * xk;
                        s += cabs1(A[i + k * lda]) * cabs1(X[i + j * ldx]);
                    }
                    rwork[k] += s;
                }
            }

            f64 s = ZERO;
            for (int i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    f64 tmp = cabs1(work[i]) / rwork[i];
                    if (tmp > s) s = tmp;
                } else {
                    f64 tmp = (cabs1(work[i]) + safe1) / (rwork[i] + safe1);
                    if (tmp > s) s = tmp;
                }
            }
            berr[j] = s;

            // Test stopping criterion
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                // Update solution and try again
                int linfo;
                zpotrs(uplo, n, 1, AF, ldaf, work, n, &linfo);
                cblas_zaxpy(n, &ONE, work, 1, &X[j * ldx], 1);
                lstres = berr[j];
                count++;
            } else {
                break;
            }
        }

        // Bound error from formula
        // norm(X - XTRUE) / norm(X) <= FERR =
        // norm( |inv(A)| * ( |R| + NZ*EPS*( |A|*|X|+|B| ))) / norm(X)
        for (int i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        int kase = 0;
        int isave[3] = {0, 0, 0};
        for (;;) {
            zlacn2(n, &work[n], work, &ferr[j], &kase, isave);
            if (kase == 0) break;

            if (kase == 1) {
                // Multiply by diag(W)*inv(A**H)
                int linfo;
                zpotrs(uplo, n, 1, AF, ldaf, work, n, &linfo);
                for (int i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
            } else if (kase == 2) {
                // Multiply by inv(A)*diag(W)
                for (int i = 0; i < n; i++) {
                    work[i] = CMPLX(rwork[i], 0.0) * work[i];
                }
                int linfo;
                zpotrs(uplo, n, 1, AF, ldaf, work, n, &linfo);
            }
        }

        // Normalize error
        lstres = ZERO;
        for (int i = 0; i < n; i++) {
            f64 tmp = cabs1(X[i + j * ldx]);
            if (tmp > lstres) lstres = tmp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
