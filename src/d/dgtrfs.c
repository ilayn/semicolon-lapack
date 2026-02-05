/**
 * @file dgtrfs.c
 * @brief DGTRFS improves the computed solution to a system of linear equations
 *        when the coefficient matrix is tridiagonal.
 */

#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_double.h"

/**
 * DGTRFS improves the computed solution to a system of linear
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
 * @param[in,out] X     On entry, the solution matrix X, as computed by DGTTRS.
 *                      On exit, the improved solution matrix X.
 *                      Array of dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of X. ldx >= max(1, n).
 * @param[out]    ferr  The estimated forward error bound for each solution vector.
 *                      Array of dimension (nrhs).
 * @param[out]    berr  The componentwise relative backward error of each solution.
 *                      Array of dimension (nrhs).
 * @param[out]    work  Workspace array of dimension (3*n).
 * @param[out]    iwork Integer workspace array of dimension (n).
 * @param[out]    info  = 0: successful exit
 *                      < 0: if info = -i, the i-th argument had an illegal value
 */
void dgtrfs(
    const char* trans,
    const int n,
    const int nrhs,
    const double * const restrict DL,
    const double * const restrict D,
    const double * const restrict DU,
    const double * const restrict DLF,
    const double * const restrict DF,
    const double * const restrict DUF,
    const double * const restrict DU2,
    const int * const restrict ipiv,
    const double * const restrict B,
    const int ldb,
    double * const restrict X,
    const int ldx,
    double * const restrict ferr,
    double * const restrict berr,
    double * const restrict work,
    int * const restrict iwork,
    int *info)
{
    const int ITMAX = 5;
    const double ZERO = 0.0;
    const double ONE = 1.0;
    const double TWO = 2.0;
    const double THREE = 3.0;

    int notran;
    char transn, transt;
    int count, i, j, kase, nz;
    double eps, lstres, s, safe1, safe2, safmin;
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
        xerbla("DGTRFS", -(*info));
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
    eps = dlamch("E");
    safmin = dlamch("S");
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
            cblas_dcopy(n, B + j * ldb, 1, work + n, 1);
            /* work[n:2n-1] = -1.0 * op(A) * X(:,j) + 1.0 * work[n:2n-1] */
            dlagtm(trans, n, 1, -ONE, DL, D, DU, X + j * ldx, ldx, ONE, work + n, n);

            /* Compute abs(op(A))*abs(x) + abs(b) for use in backward error bound */
            if (notran) {
                if (n == 1) {
                    work[0] = fabs(B[j * ldb]) + fabs(D[0] * X[j * ldx]);
                } else {
                    work[0] = fabs(B[j * ldb]) + fabs(D[0] * X[j * ldx])
                              + fabs(DU[0] * X[1 + j * ldx]);
                    for (i = 1; i < n - 1; i++) {
                        work[i] = fabs(B[i + j * ldb])
                                  + fabs(DL[i - 1] * X[(i - 1) + j * ldx])
                                  + fabs(D[i] * X[i + j * ldx])
                                  + fabs(DU[i] * X[(i + 1) + j * ldx]);
                    }
                    work[n - 1] = fabs(B[(n - 1) + j * ldb])
                                  + fabs(DL[n - 2] * X[(n - 2) + j * ldx])
                                  + fabs(D[n - 1] * X[(n - 1) + j * ldx]);
                }
            } else {
                if (n == 1) {
                    work[0] = fabs(B[j * ldb]) + fabs(D[0] * X[j * ldx]);
                } else {
                    work[0] = fabs(B[j * ldb]) + fabs(D[0] * X[j * ldx])
                              + fabs(DL[0] * X[1 + j * ldx]);
                    for (i = 1; i < n - 1; i++) {
                        work[i] = fabs(B[i + j * ldb])
                                  + fabs(DU[i - 1] * X[(i - 1) + j * ldx])
                                  + fabs(D[i] * X[i + j * ldx])
                                  + fabs(DL[i] * X[(i + 1) + j * ldx]);
                    }
                    work[n - 1] = fabs(B[(n - 1) + j * ldb])
                                  + fabs(DU[n - 2] * X[(n - 2) + j * ldx])
                                  + fabs(D[n - 1] * X[(n - 1) + j * ldx]);
                }
            }

            /* Compute componentwise relative backward error */
            s = ZERO;
            for (i = 0; i < n; i++) {
                if (work[i] > safe2) {
                    double temp = fabs(work[n + i]) / work[i];
                    if (s < temp) s = temp;
                } else {
                    double temp = (fabs(work[n + i]) + safe1) / (work[i] + safe1);
                    if (s < temp) s = temp;
                }
            }
            berr[j] = s;

            /* Test stopping criterion */
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                /* Update solution and try again */
                dgttrs(trans, n, 1, DLF, DF, DUF, DU2, ipiv, work + n, n, &gttrs_info);
                cblas_daxpy(n, ONE, work + n, 1, X + j * ldx, 1);
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
                work[i] = fabs(work[n + i]) + nz * eps * work[i];
            } else {
                work[i] = fabs(work[n + i]) + nz * eps * work[i] + safe1;
            }
        }

        kase = 0;
        for (;;) {
            dlacn2(n, work + 2 * n, work + n, iwork, &ferr[j], &kase, isave);

            if (kase == 0) {
                break;
            }

            if (kase == 1) {
                /* Multiply by diag(W)*inv(op(A)**T) */
                dgttrs(&transt, n, 1, DLF, DF, DUF, DU2, ipiv, work + n, n, &gttrs_info);
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else {
                /* Multiply by inv(op(A))*diag(W) */
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                dgttrs(&transn, n, 1, DLF, DF, DUF, DU2, ipiv, work + n, n, &gttrs_info);
            }
        }

        /* Normalize error */
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            double temp = fabs(X[i + j * ldx]);
            if (lstres < temp) lstres = temp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
