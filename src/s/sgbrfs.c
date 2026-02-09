/**
 * @file sgbrfs.c
 * @brief Improves the computed solution for banded systems and provides error bounds.
 */

#include <math.h>
#include <float.h>
#include <cblas.h>
#include "semicolon_lapack_single.h"

/**
 * SGBRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is banded, and provides
 * error bounds and backward error estimates for the solution.
 *
 * @param[in]     trans   Specifies the form of the system of equations:
 *                        - 'N': A * X = B (No transpose)
 *                        - 'T': A**T * X = B (Transpose)
 *                        - 'C': A**H * X = B (Conjugate transpose = Transpose)
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     kl      The number of subdiagonals within the band of A (kl >= 0).
 * @param[in]     ku      The number of superdiagonals within the band of A (ku >= 0).
 * @param[in]     nrhs    The number of right hand sides (nrhs >= 0).
 * @param[in]     AB      The original band matrix A, stored in rows 0 to kl+ku.
 *                        The j-th column of A is stored in the j-th column of AB:
 *                        AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku)<=i<=min(n-1,j+kl).
 *                        Array of dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of AB (ldab >= kl+ku+1).
 * @param[in]     AFB     The LU factorization of A, as computed by sgbtrf.
 *                        U is stored in rows 0 to kl+ku, and the multipliers
 *                        are stored in rows kl+ku+1 to 2*kl+ku.
 *                        Array of dimension (ldafb, n).
 * @param[in]     ldafb   The leading dimension of AFB (ldafb >= 2*kl+ku+1).
 * @param[in]     ipiv    The pivot indices from sgbtrf. Array of dimension n.
 * @param[in]     B       The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb     The leading dimension of B (ldb >= max(1,n)).
 * @param[in,out] X       On entry, the solution matrix X, as computed by sgbtrs.
 *                        On exit, the improved solution matrix X.
 *                        Array of dimension (ldx, nrhs).
 * @param[in]     ldx     The leading dimension of X (ldx >= max(1,n)).
 * @param[out]    ferr    The estimated forward error bound for each solution vector
 *                        X(j). Array of dimension nrhs.
 * @param[out]    berr    The componentwise relative backward error of each solution
 *                        vector X(j). Array of dimension nrhs.
 * @param[out]    work    Workspace array of dimension (3*n).
 * @param[out]    iwork   Integer workspace array of dimension (n).
 * @param[out]    info    Exit status:
 *                        - = 0: successful exit
 *                        - < 0: if info = -i, the i-th argument had an illegal value
 */
void sgbrfs(
    const char* trans,
    const int n,
    const int kl,
    const int ku,
    const int nrhs,
    const float * const restrict AB,
    const int ldab,
    const float * const restrict AFB,
    const int ldafb,
    const int * const restrict ipiv,
    const float * const restrict B,
    const int ldb,
    float * const restrict X,
    const int ldx,
    float * const restrict ferr,
    float * const restrict berr,
    float * const restrict work,
    int * const restrict iwork,
    int *info)
{
    const int ITMAX = 5;
    const float ZERO = 0.0f;
    const float ONE = 1.0f;
    const float TWO = 2.0f;
    const float THREE = 3.0f;

    int notran;
    char transt;
    int count, i, j, k, kase, kk, nz;
    float eps, lstres, s, safe1, safe2, safmin, xk;
    int isave[3];
    int linfo;

    /* Test the input parameters */
    *info = 0;
    notran = (trans[0] == 'N' || trans[0] == 'n');
    if (!notran && trans[0] != 'T' && trans[0] != 't' && trans[0] != 'C' && trans[0] != 'c') {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (kl < 0) {
        *info = -3;
    } else if (ku < 0) {
        *info = -4;
    } else if (nrhs < 0) {
        *info = -5;
    } else if (ldab < kl + ku + 1) {
        *info = -7;
    } else if (ldafb < 2 * kl + ku + 1) {
        *info = -9;
    } else if (ldb < (n > 1 ? n : 1)) {
        *info = -12;
    } else if (ldx < (n > 1 ? n : 1)) {
        *info = -14;
    }

    if (*info != 0) {
        xerbla("SGBRFS", -(*info));
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
        transt = 'T';
    } else {
        transt = 'N';
    }

    /* NZ = maximum number of nonzero elements in each row of A, plus 1 */
    nz = (kl + ku + 2 < n + 1) ? kl + ku + 2 : n + 1;
    eps = FLT_EPSILON;
    safmin = FLT_MIN;
    safe1 = nz * safmin;
    safe2 = safe1 / eps;

    /* Do for each right hand side */
    for (j = 0; j < nrhs; j++) {
        count = 1;
        lstres = THREE;

        /* Iterative refinement loop until stopping criterion is satisfied */
        while (1) {
            /* Compute residual R = B - op(A) * X,
             * where op(A) = A, A**T, or A**H, depending on TRANS
             */
            cblas_scopy(n, &B[j * ldb], 1, &work[n], 1);
            cblas_sgbmv(CblasColMajor, notran ? CblasNoTrans : CblasTrans,
                        n, n, kl, ku, -ONE, AB, ldab, &X[j * ldx], 1, ONE, &work[n], 1);

            /* Compute componentwise relative backward error from formula
             *   max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
             */
            for (i = 0; i < n; i++) {
                work[i] = fabsf(B[i + j * ldb]);
            }

            /* Compute abs(op(A))*abs(X) + abs(B) */
            if (notran) {
                for (k = 0; k < n; k++) {
                    kk = ku - k;
                    xk = fabsf(X[k + j * ldx]);
                    int i_start = (k - ku > 0) ? k - ku : 0;
                    int i_end = (k + kl < n - 1) ? k + kl : n - 1;
                    for (i = i_start; i <= i_end; i++) {
                        work[i] = work[i] + fabsf(AB[kk + i + k * ldab]) * xk;
                    }
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    kk = ku - k;
                    int i_start = (k - ku > 0) ? k - ku : 0;
                    int i_end = (k + kl < n - 1) ? k + kl : n - 1;
                    for (i = i_start; i <= i_end; i++) {
                        s = s + fabsf(AB[kk + i + k * ldab]) * fabsf(X[i + j * ldx]);
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

            /* Stop iterating if
             *   1) The residual BERR(J) is not larger than machine epsilon, or
             *   2) BERR(J) did not decrease by at least a factor of 2, or
             *   3) More than ITMAX iterations tried.
             */
            if (!(berr[j] > eps && TWO * berr[j] <= lstres && count <= ITMAX)) {
                break;
            }

            /* Update solution and try again */
            sgbtrs(trans, n, kl, ku, 1, AFB, ldafb, ipiv, &work[n], n, &linfo);
            cblas_saxpy(n, ONE, &work[n], 1, &X[j * ldx], 1);
            lstres = berr[j];
            count = count + 1;
        }

        /* Bound error from formula
         *   norm(X - XTRUE) / norm(X) .le. FERR =
         *   norm( abs(inv(op(A)))* ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)
         */
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
                /* Multiply by diag(W)*inv(op(A)**T) */
                sgbtrs(&transt, n, kl, ku, 1, AFB, ldafb, ipiv, &work[n], n, &linfo);
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
            } else {
                /* Multiply by inv(op(A))*diag(W) */
                for (i = 0; i < n; i++) {
                    work[n + i] = work[i] * work[n + i];
                }
                sgbtrs(trans, n, kl, ku, 1, AFB, ldafb, ipiv, &work[n], n, &linfo);
            }
        }

        /* Normalize error */
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > fabsf(X[i + j * ldx])) ? lstres : fabsf(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
