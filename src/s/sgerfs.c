/**
 * @file sgerfs.c
 * @brief Improves the computed solution and provides error bounds.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGERFS improves the computed solution to a system of linear
 * equations and provides error bounds and backward error estimates for
 * the solution.
 *
 * @param[in]     trans  Specifies the form of the system of equations:
 *                       - 'N': A * X = B (No transpose)
 *                       - 'T': A**T * X = B (Transpose)
 *                       - 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     n      The order of the matrix A (n >= 0).
 * @param[in]     nrhs   The number of right hand sides (nrhs >= 0).
 * @param[in]     A      The original N-by-N matrix A. Array of dimension (lda, n).
 * @param[in]     lda    The leading dimension of the array A (lda >= max(1,n)).
 * @param[in]     AF     The factors L and U from the factorization A = P*L*U
 *                       as computed by sgetrf. Array of dimension (ldaf, n).
 * @param[in]     ldaf   The leading dimension of the array AF (ldaf >= max(1,n)).
 * @param[in]     ipiv   The pivot indices from sgetrf. Array of dimension n.
 * @param[in]     B      The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb    The leading dimension of the array B (ldb >= max(1,n)).
 * @param[in,out] X      On entry, the solution matrix X, as computed by sgetrs.
 *                       On exit, the improved solution matrix X.
 *                       Array of dimension (ldx, nrhs).
 * @param[in]     ldx    The leading dimension of the array X (ldx >= max(1,n)).
 * @param[out]    ferr   The estimated forward error bound for each solution vector
 *                       X(j). Array of dimension nrhs.
 * @param[out]    berr   The componentwise relative backward error of each solution
 *                       vector X(j). Array of dimension nrhs.
 * @param[out]    work   Workspace array of dimension (3*n).
 * @param[out]    iwork  Integer workspace array of dimension (n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 */
void sgerfs(
    const char* trans,
    const INT n,
    const INT nrhs,
    const f32* restrict A,
    const INT lda,
    const f32* restrict AF,
    const INT ldaf,
    const INT* restrict ipiv,
    const f32* restrict B,
    const INT ldb,
    f32* restrict X,
    const INT ldx,
    f32* restrict ferr,
    f32* restrict berr,
    f32* restrict work,
    INT* restrict iwork,
    INT* info)
{
    const INT ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT notran;
    char transt;
    INT count, i, j, k, kase, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT linfo;

    // Test the input parameters
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
        xerbla("SGERFS", -(*info));
        return;
    }

    // Quick return if possible
    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    if (notran) {
        transt = 'T';
    } else {
        transt = 'N';
    }

    // NZ = maximum number of nonzero elements in each row of A, plus 1
    nz = n + 1;
    eps = FLT_EPSILON;
    safmin = FLT_MIN;
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    // Do for each right hand side
    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        // Iterative refinement loop
        while (1) {
            // Compute residual R = B - op(A) * X,
            // where op(A) = A, A**T, or A**H, depending on TRANS
            cblas_scopy(n, &B[j * ldb], 1, &work[n], 1);
            cblas_sgemv(CblasColMajor, notran ? CblasNoTrans : CblasTrans,
                        n, n, -ONE, A, lda, &X[j * ldx], 1, ONE, &work[n], 1);

            // Compute componentwise relative backward error from formula
            //   max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )

            for (i = 0; i < n; i++) {
                work[i] = fabsf(B[i + j * ldb]);
            }

            // Compute abs(op(A))*abs(X) + abs(B)
            if (notran) {
                for (k = 0; k < n; k++) {
                    xk = fabsf(X[k + j * ldx]);
                    for (i = 0; i < n; i++) {
                        work[i] = work[i] + fabsf(A[i + k * lda]) * xk;
                    }
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    for (i = 0; i < n; i++) {
                        s = s + fabsf(A[i + k * lda]) * fabsf(X[i + j * ldx]);
                    }
                    work[k] = work[k] + s;
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (work[i] > safe2) {
                    s = (s > fabsf(work[n + i]) / work[i]) ? s : fabsf(work[n + i]) / work[i];
                } else {
                    s = (s > (fabsf(work[n + i]) + safe1) / (work[i] + safe1))
                        ? s : (fabsf(work[n + i]) + safe1) / (work[i] + safe1);
                }
            }
            berr[j] = s;

            // Stop if convergence criterion is met
            if (!(berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX)) {
                break;
            }

            // Update solution and try again
            sgetrs(trans, n, 1, AF, ldaf, ipiv, &work[n], n, &linfo);
            cblas_saxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
            lstres = berr[j];
            count = count + 1;
        }

        // Bound error from formula
        //   norm(X - XTRUE) / norm(X) .le. FERR =
        //   norm( abs(inv(op(A)))* ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)

        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        kase = 0;
        while (1) {
            slacn2(n, &work[2 * n], &work[n], iwork, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                // Multiply by diag(W)*inv(op(A)**T)
                sgetrs(&transt, n, 1, AF, ldaf, ipiv, &work[n], n, &linfo);
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else {
                // Multiply by inv(op(A))*diag(W)
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                sgetrs(trans, n, 1, AF, ldaf, ipiv, &work[n], n, &linfo);
            }
        }

        // Normalize error
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > fabsf(X[i + j * ldx])) ? lstres : fabsf(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
