/**
 * @file cgtrfs.c
 * @brief CGTRFS improves the computed solution to a system of linear equations
 *        when the coefficient matrix is tridiagonal.
 */

#include "internal_build_defs.h"
#include <complex.h>
#include <math.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGTRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is tridiagonal, and provides
 * error bounds and backward error estimates for the solution.
 *
 * @param[in]     trans Specifies the form of the system of equations:
 *                      = 'N': A * X = B     (No transpose)
 *                      = 'T': A**T * X = B  (Transpose)
 *                      = 'C': A**H * X = B  (Conjugate transpose)
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
 * @param[in,out] X     On entry, the solution matrix X, as computed by CGTTRS.
 *                      On exit, the improved solution matrix X.
 *                      Array of dimension (ldx, nrhs).
 * @param[in]     ldx   The leading dimension of X. ldx >= max(1, n).
 * @param[out]    ferr  The estimated forward error bound for each solution vector.
 *                      Array of dimension (nrhs).
 * @param[out]    berr  The componentwise relative backward error of each solution.
 *                      Array of dimension (nrhs).
 * @param[out]    work  Workspace array of dimension (2*n).
 * @param[out]    rwork Real workspace array of dimension (n).
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 */
void cgtrfs(
    const char* trans,
    const INT n,
    const INT nrhs,
    const c64* restrict DL,
    const c64* restrict D,
    const c64* restrict DU,
    const c64* restrict DLF,
    const c64* restrict DF,
    const c64* restrict DUF,
    const c64* restrict DU2,
    const INT* restrict ipiv,
    const c64* restrict B,
    const INT ldb,
    c64* restrict X,
    const INT ldx,
    f32* restrict ferr,
    f32* restrict berr,
    c64* restrict work,
    f32* restrict rwork,
    INT* info)
{
    const INT ITMAX = 5;
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;

    INT notran;
    char transn, transt;
    INT count, i, j, kase, nz;
    f32 eps, lstres, s, safe1, safe2, safmin;
    INT isave[3];
    INT ldb_min, ldx_min;
    INT gttrs_info;

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
        xerbla("CGTRFS", -(*info));
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
        transt = 'C';
    } else {
        transn = 'C';
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
            cblas_ccopy(n, &B[j * ldb], 1, work, 1);
            {
                const f32 neg_one = -ONE;
                const f32 one = ONE;
                clagtm(trans, n, 1, neg_one, DL, D, DU, &X[j * ldx], ldx,
                        one, work, n);
            }

            /* Compute abs(op(A))*abs(x) + abs(b) for use in backward error bound */
            if (notran) {
                if (n == 1) {
                    rwork[0] = cabs1f(B[j * ldb])
                               + cabs1f(D[0]) * cabs1f(X[j * ldx]);
                } else {
                    rwork[0] = cabs1f(B[j * ldb])
                               + cabs1f(D[0]) * cabs1f(X[j * ldx])
                               + cabs1f(DU[0]) * cabs1f(X[1 + j * ldx]);
                    for (i = 1; i < n - 1; i++) {
                        rwork[i] = cabs1f(B[i + j * ldb])
                                   + cabs1f(DL[i - 1]) * cabs1f(X[(i - 1) + j * ldx])
                                   + cabs1f(D[i]) * cabs1f(X[i + j * ldx])
                                   + cabs1f(DU[i]) * cabs1f(X[(i + 1) + j * ldx]);
                    }
                    rwork[n - 1] = cabs1f(B[(n - 1) + j * ldb])
                                   + cabs1f(DL[n - 2]) * cabs1f(X[(n - 2) + j * ldx])
                                   + cabs1f(D[n - 1]) * cabs1f(X[(n - 1) + j * ldx]);
                }
            } else {
                if (n == 1) {
                    rwork[0] = cabs1f(B[j * ldb])
                               + cabs1f(D[0]) * cabs1f(X[j * ldx]);
                } else {
                    rwork[0] = cabs1f(B[j * ldb])
                               + cabs1f(D[0]) * cabs1f(X[j * ldx])
                               + cabs1f(DL[0]) * cabs1f(X[1 + j * ldx]);
                    for (i = 1; i < n - 1; i++) {
                        rwork[i] = cabs1f(B[i + j * ldb])
                                   + cabs1f(DU[i - 1]) * cabs1f(X[(i - 1) + j * ldx])
                                   + cabs1f(D[i]) * cabs1f(X[i + j * ldx])
                                   + cabs1f(DL[i]) * cabs1f(X[(i + 1) + j * ldx]);
                    }
                    rwork[n - 1] = cabs1f(B[(n - 1) + j * ldb])
                                   + cabs1f(DU[n - 2]) * cabs1f(X[(n - 2) + j * ldx])
                                   + cabs1f(D[n - 1]) * cabs1f(X[(n - 1) + j * ldx]);
                }
            }

            /* Compute componentwise relative backward error */
            s = ZERO;
            for (i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    f32 temp = cabs1f(work[i]) / rwork[i];
                    if (s < temp) s = temp;
                } else {
                    f32 temp = (cabs1f(work[i]) + safe1) / (rwork[i] + safe1);
                    if (s < temp) s = temp;
                }
            }
            berr[j] = s;

            /* Test stopping criterion */
            if (berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX) {
                /* Update solution and try again */
                cgttrs(trans, n, 1, DLF, DF, DUF, DU2, ipiv, work, n, &gttrs_info);
                {
                    const c64 zone = CMPLXF(ONE, 0.0f);
                    cblas_caxpy(n, &zone, work, 1, &X[j * ldx], 1);
                }
                lstres = berr[j];
                count++;
            } else {
                break;
            }
        }

        /* Compute forward error bound using norm estimation */
        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        kase = 0;
        for (;;) {
            clacn2(n, &work[n], work, &ferr[j], &kase, isave);

            if (kase == 0) {
                break;
            }

            if (kase == 1) {
                /* Multiply by diag(W)*inv(op(A)**H) */
                cgttrs(&transt, n, 1, DLF, DF, DUF, DU2, ipiv, work, n, &gttrs_info);
                for (i = 0; i < n; i++) {
                    work[i] = rwork[i] * work[i];
                }
            } else {
                /* Multiply by inv(op(A))*diag(W) */
                for (i = 0; i < n; i++) {
                    work[i] = rwork[i] * work[i];
                }
                cgttrs(&transn, n, 1, DLF, DF, DUF, DU2, ipiv, work, n, &gttrs_info);
            }
        }

        /* Normalize error */
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            f32 temp = cabs1f(X[i + j * ldx]);
            if (lstres < temp) lstres = temp;
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
