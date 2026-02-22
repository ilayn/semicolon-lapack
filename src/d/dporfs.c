/**
 * @file dporfs.c
 * @brief DPORFS improves the computed solution and provides error bounds
 *        for symmetric positive definite systems.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DPORFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is symmetric positive definite,
 * and provides error bounds and backward error estimates for the
 * solution.
 *
 * @param[in]     uplo   = 'U': Upper triangle of A is stored
 *                        = 'L': Lower triangle of A is stored
 * @param[in]     n      The order of the matrix A. n >= 0.
 * @param[in]     nrhs   The number of right hand sides. nrhs >= 0.
 * @param[in]     A      The symmetric matrix A. Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of A. lda >= max(1, n).
 * @param[in]     AF     The triangular factor U or L from the Cholesky
 *                       factorization A = U**T*U or A = L*L**T.
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
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -k, the k-th argument had an illegal value
 */
void dporfs(
    const char* uplo,
    const INT n,
    const INT nrhs,
    const f64* restrict A,
    const INT lda,
    const f64* restrict AF,
    const INT ldaf,
    const f64* restrict B,
    const INT ldb,
    f64* restrict X,
    const INT ldx,
    f64* restrict ferr,
    f64* restrict berr,
    f64* restrict work,
    INT* restrict iwork,
    INT* info)
{
    // Parameters from the Fortran source
    const INT ITMAX = 5;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 THREE = 3.0;

    // Test the input parameters
    *info = 0;
    INT upper = (uplo[0] == 'U' || uplo[0] == 'u');
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
        xerbla("DPORFS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        for (INT j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    // NZ = maximum number of nonzero elements in each row of A, plus 1
    INT nz = n + 1;
    f64 eps = dlamch("E");
    f64 safmin = dlamch("S");
    f64 safe1 = nz * safmin;
    f64 safe2 = safe1 / eps;

    CBLAS_UPLO cblas_uplo = upper ? CblasUpper : CblasLower;

    // Do for each right hand side
    for (INT j = 0; j < nrhs; j++) {
        INT count = 1;
        f64 lstres = THREE;

        for (;;) {
            // Compute residual R = B - A * X
            // Copy B(:,j) into work[n..2n-1]
            cblas_dcopy(n, &B[j * ldb], 1, &work[n], 1);
            // work[n..2n-1] = -A * X(:,j) + work[n..2n-1]
            cblas_dsymv(CblasColMajor, cblas_uplo, n, -ONE, A, lda,
                        &X[j * ldx], 1, ONE, &work[n], 1);

            // Compute componentwise relative backward error
            // max(i) ( |R(i)| / ( |A|*|X| + |B| )(i) )
            for (INT i = 0; i < n; i++) {
                work[i] = fabs(B[i + j * ldb]);
            }

            // Compute |A|*|X| + |B|
            if (upper) {
                for (INT k = 0; k < n; k++) {
                    f64 s = ZERO;
                    f64 xk = fabs(X[k + j * ldx]);
                    for (INT i = 0; i < k; i++) {
                        work[i] += fabs(A[i + k * lda]) * xk;
                        s += fabs(A[i + k * lda]) * fabs(X[i + j * ldx]);
                    }
                    work[k] += fabs(A[k + k * lda]) * xk + s;
                }
            } else {
                for (INT k = 0; k < n; k++) {
                    f64 s = ZERO;
                    f64 xk = fabs(X[k + j * ldx]);
                    work[k] += fabs(A[k + k * lda]) * xk;
                    for (INT i = k + 1; i < n; i++) {
                        work[i] += fabs(A[i + k * lda]) * xk;
                        s += fabs(A[i + k * lda]) * fabs(X[i + j * ldx]);
                    }
                    work[k] += s;
                }
            }

            f64 s = ZERO;
            for (INT i = 0; i < n; i++) {
                if (work[i] > safe2) {
                    f64 tmp = fabs(work[n + i]) / work[i];
                    if (tmp > s) s = tmp;
                } else {
                    f64 tmp = (fabs(work[n + i]) + safe1) / (work[i] + safe1);
                    if (tmp > s) s = tmp;
                }
            }
            berr[j] = s;

            // Test stopping criterion
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                // Update solution and try again
                INT linfo;
                dpotrs(uplo, n, 1, AF, ldaf, &work[n], n, &linfo);
                cblas_daxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
                lstres = berr[j];
                count++;
            } else {
                break;
            }
        }

        // Bound error from formula
        // norm(X - XTRUE) / norm(X) <= FERR =
        // norm( |inv(A)| * ( |R| + NZ*EPS*( |A|*|X|+|B| ))) / norm(X)
        for (INT i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabs(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabs(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        INT kase = 0;
        INT isave[3] = {0, 0, 0};
        for (;;) {
            dlacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase == 0) break;

            if (kase == 1) {
                // Multiply by diag(W)*inv(A**T)
                INT linfo;
                dpotrs(uplo, n, 1, AF, ldaf, &work[n], n, &linfo);
                for (INT i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else if (kase == 2) {
                // Multiply by inv(A)*diag(W)
                for (INT i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                INT linfo;
                dpotrs(uplo, n, 1, AF, ldaf, &work[n], n, &linfo);
            }
        }

        // Normalize error
        lstres = ZERO;
        for (INT i = 0; i < n; i++) {
            f64 tmp = fabs(X[i + j * ldx]);
            if (tmp > lstres) lstres = tmp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
