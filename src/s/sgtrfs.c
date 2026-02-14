/**
 * @file sgtrfs.c
 * @brief SGTRFS improves the computed solution to a system of linear equations
 *        when the coefficient matrix is tridiagonal.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGTRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is tridiagonal, and provides
 * error bounds and backward error estimates for the solution.
 *
 * @param[in]     trans Specifies the form of the system of equations:
 *                      = 'N': A * X = B     (No transpose)
 *                      = 'T': A**T * X = B  (Transpose)
 *                      = 'C': A**H * X = B  (Conjugate transpose = Transpose)
 * @param[in]     n     The order of the matrix A. n >= 0.
 * @param[in]     nrhs  The number of right hand sides. nrhs >= 0.
 * @param[in]     DL    The (n-1) subdiagonal elements of A. Array of dimension (n-1).
 * @param[in]     D     The diagonal elements of A. Array of dimension (n).
 * @param[in]     DU    The (n-1) superdiagonal elements of A. Array of dimension (n-1).
 * @param[in]     DLF   The (n-1) multipliers that define the matrix L from the
 *                      LU factorization of A. Array of dimension (n-1).
 * @param[in]     DF    The n diagonal elements of U. Array of dimension (n).
 * @param[in]     DUF   The (n-1) elements of the first superdiagonal of U.
 *                      Array of dimension (n-1).
 * @param[in]     DU2   The (n-2) elements of the second superdiagonal of U.
 *                      Array of dimension (n-2).
 * @param[in]     ipiv  The pivot indices. Array of dimension (n).
 * @param[in]     B     The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb   The leading dimension of B. ldb >= max(1, n).
 * @param[in,out] X     On entry, the solution matrix X, as computed by SGTTRS.
 *                      On exit, the improved solution matrix X.
 *                      Array of dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of X. ldx >= max(1, n).
 * @param[out]    ferr  The estimated forward error bound for each solution vector.
 *                      Array of dimension (nrhs).
 * @param[out]    berr  The componentwise relative backward error of each solution.
 *                      Array of dimension (nrhs).
 * @param[out]    work  Workspace array of dimension (3*n).
 * @param[out]    iwork Integer workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void sgtrfs(
    const char* trans,
    const int n,
    const int nrhs,
    const f32 * const restrict DL,
    const f32 * const restrict D,
    const f32 * const restrict DU,
    const f32 * const restrict DLF,
    const f32 * const restrict DF,
    const f32 * const restrict DUF,
    const f32 * const restrict DU2,
    const int * const restrict ipiv,
    const f32 * const restrict B,
    const int ldb,
    f32 * const restrict X,
    const int ldx,
    f32 * const restrict ferr,
    f32 * const restrict berr,
    f32 * const restrict work,
    int * const restrict iwork,
    int *info)
{
    const int ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    int notran;
    char transn, transt;
    int count, i, j, kase, nz;
    f32 eps, lstres, s, safe1, safe2, safmin;
    int isave[3];
    int ldb_min, ldx_min;
    int gttrs_info;

    /* Test the input parameters */
    *info = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');

    if (!notran && !(trans[0] == 'T' || trans[0] == 't') && !(trans[0] == 'C' || trans[0] == 'c')) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (nrhs < 0) {
        *info = -3;
    } else {
        ldb_min = (n > 1) ? n : 1;
        ldx_min = (n > 1) ? n : 1;
        if (ldb < ldb_min) {
            *info = -13;
        } else if (ldx < ldx_min) {
            *info = -15;
        }
    }

    if (*info != 0) {
        xerbla("SGTRFS", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0 || nrhs == 0) {
        for (j = 0; j < nrhs; j++) {
            ferr[j] = ZERO;
            berr[j] = ZERO;
        }
        return;
    }

    if (notran) {
        transn = 'N';
        transt = 'T';
    } else {
        transn = 'T';
        transt = 'N';
    }

    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */
    nz = 4;
    eps = slamch("E");
    safmin = slamch("S");
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    /* Do for each right hand side */
    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        /* Loop until stopping criterion is satisfied */
        for (;;) {
            /* Compute residual R = B - op(A) * X */
            /* Copy B(:,j) to work[n:2n-1] */
            cblas_scopy(n, B + j * ldb, 1, work + n, 1);
            /* work[n:2n-1] = -1.0 * op(A) * X(:,j) + 1.0 * work[n:2n-1] */
            slagtm(trans, n, 1, -ONE, DL, D, DU, X + j * ldx, ldx, ONE, work + n, n);

            /* Compute abs(op(A))*abs(x) + abs(b) for use in backward error bound */
            if (notran) {
                if (n == 1) {
                    work[0] = fabsf(B[j * ldb]) + fabsf(D[0] * X[j * ldx]);
                } else {
                    work[0] = fabsf(B[j * ldb]) + fabsf(D[0] * X[j * ldx])
                              + fabsf(DU[0] * X[1 + j * ldx]);
                    for (i = 1; i < n - 1; i++) {
                        work[i] = fabsf(B[i + j * ldb])
                                  + fabsf(DL[i - 1] * X[(i - 1) + j * ldx])
                                  + fabsf(D[i] * X[i + j * ldx])
                                  + fabsf(DU[i] * X[(i + 1) + j * ldx]);
                    }
                    work[n - 1] = fabsf(B[(n - 1) + j * ldb])
                                  + fabsf(DL[n - 2] * X[(n - 2) + j * ldx])
                                  + fabsf(D[n - 1] * X[(n - 1) + j * ldx]);
                }
            } else {
                if (n == 1) {
                    work[0] = fabsf(B[j * ldb]) + fabsf(D[0] * X[j * ldx]);
                } else {
                    work[0] = fabsf(B[j * ldb]) + fabsf(D[0] * X[j * ldx])
                              + fabsf(DL[0] * X[1 + j * ldx]);
                    for (i = 1; i < n - 1; i++) {
                        work[i] = fabsf(B[i + j * ldb])
                                  + fabsf(DU[i - 1] * X[(i - 1) + j * ldx])
                                  + fabsf(D[i] * X[i + j * ldx])
                                  + fabsf(DL[i] * X[(i + 1) + j * ldx]);
                    }
                    work[n - 1] = fabsf(B[(n - 1) + j * ldb])
                                  + fabsf(DU[n - 2] * X[(n - 2) + j * ldx])
                                  + fabsf(D[n - 1] * X[(n - 1) + j * ldx]);
                }
            }

            /* Compute componentwise relative backward error */
            s = ZERO;
            for (i = 0; i < n; i++) {
                if (work[i] > safe2) {
                    f32 temp = fabsf(work[n + i]) / work[i];
                    if (s < temp) s = temp;
                } else {
                    f32 temp = (fabsf(work[n + i]) + safe1) / (work[i] + safe1);
                    if (s < temp) s = temp;
                }
            }
            berr[j] = s;

            /* Test stopping criterion */
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                /* Update solution and try again */
                sgttrs(trans, n, 1, DLF, DF, DUF, DU2, ipiv, work + n, n, &gttrs_info);
                cblas_saxpy(n, ONE, work + n, 1, X + j * ldx, 1);
                lstres = berr[j];
                count++;
            } else {
                break;
            }
        }

        /* Compute forward error bound using norm estimation */
        /* Prepare work[0:n-1] for the error bound computation */
        for (i = 0; i < n; i++) {
            if (work[i] > safe2) {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabsf(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        kase = 0;
        for (;;) {
            slacn2(n, work + 2 * n, work + n, iwork, &ferr[j], &kase, isave);

            if (kase == 0) {
                break;
            }

            if (kase == 1) {
                /* Multiply by diag(W)*inv(op(A)**T) */
                sgttrs(&transt, n, 1, DLF, DF, DUF, DU2, ipiv, work + n, n, &gttrs_info);
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else {
                /* Multiply by inv(op(A))*diag(W) */
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                sgttrs(&transn, n, 1, DLF, DF, DUF, DU2, ipiv, work + n, n, &gttrs_info);
            }
        }

        /* Normalize error */
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            f32 temp = fabsf(X[i + j * ldx]);
            if (lstres < temp) lstres = temp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
