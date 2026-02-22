/**
 * @file cgbrfs.c
 * @brief Improves the computed solution for banded systems and provides error bounds.
 */

#include "internal_build_defs.h"
#include <math.h>
#include <float.h>
#include <complex.h>
#include <cblas.h>
#include "semicolon_lapack_complex_single.h"

/**
 * CGBRFS improves the computed solution to a system of linear
 * equations when the coefficient matrix is banded, and provides
 * error bounds and backward error estimates for the solution.
 *
 * @param[in]     trans   Specifies the form of the system of equations:
 *                        - 'N': A * X = B (No transpose)
 *                        - 'T': A**T * X = B (Transpose)
 *                        - 'C': A**H * X = B (Conjugate transpose)
 * @param[in]     n       The order of the matrix A (n >= 0).
 * @param[in]     kl      The number of subdiagonals within the band of A (kl >= 0).
 * @param[in]     ku      The number of superdiagonals within the band of A (ku >= 0).
 * @param[in]     nrhs    The number of right hand sides (nrhs >= 0).
 * @param[in]     AB      The original band matrix A, stored in rows 0 to kl+ku.
 *                        The j-th column of A is stored in the j-th column of AB:
 *                        AB[ku+i-j + j*ldab] = A(i,j) for max(0,j-ku)<=i<=min(n-1,j+kl).
 *                        Array of dimension (ldab, n).
 * @param[in]     ldab    The leading dimension of AB (ldab >= kl+ku+1).
 * @param[in]     AFB     The LU factorization of A, as computed by cgbtrf.
 *                        U is stored in rows 0 to kl+ku, and the multipliers
 *                        are stored in rows kl+ku+1 to 2*kl+ku.
 *                        Array of dimension (ldafb, n).
 * @param[in]     ldafb   The leading dimension of AFB (ldafb >= 2*kl+ku+1).
 * @param[in]     ipiv    The pivot indices from cgbtrf. Array of dimension n.
 * @param[in]     B       The right hand side matrix B. Array of dimension (ldb, nrhs).
 * @param[in]     ldb     The leading dimension of B (ldb >= max(1,n)).
 * @param[in,out] X       On entry, the solution matrix X, as computed by cgbtrs.
 *                        On exit, the improved solution matrix X.
 *                        Array of dimension (ldx, nrhs).
 * @param[in]     ldx     The leading dimension of X (ldx >= max(1,n)).
 * @param[out]    ferr    The estimated forward error bound for each solution vector
 *                        X(j). Array of dimension nrhs.
 * @param[out]    berr    The componentwise relative backward error of each solution
 *                        vector X(j). Array of dimension nrhs.
 * @param[out]    work    Complex workspace array of dimension (2*n).
 * @param[out]    rwork   Real workspace array of dimension (n).
 * @param[out]    info
 *                           Exit status:
 *                           - = 0: successful exit
 *                           - < 0: if info = -i, the i-th argument had an illegal value
 */
void cgbrfs(
    const char* trans,
    const INT n,
    const INT kl,
    const INT ku,
    const INT nrhs,
    const c64* restrict AB,
    const INT ldab,
    const c64* restrict AFB,
    const INT ldafb,
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
    const f32 TWO = 2.0f;
    const f32 THREE = 3.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
    const c64 NEG_CONE = CMPLXF(-1.0f, 0.0f);

    INT notran;
    const char* transn;
    const char* transt;
    INT count, i, j, k, kase, kk, nz;
    f32 eps, lstres, s, safe1, safe2, safmin, xk;
    INT isave[3];
    INT linfo;

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
        xerbla("CGBRFS", -(*info));
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
        transn = "N";
        transt = "C";
    } else {
        transn = "C";
        transt = "N";
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
             * where op(A) = A, A**T, or A**H, depending on TRANS.
             */
            cblas_ccopy(n, &B[j * ldb], 1, work, 1);
            cblas_cgbmv(CblasColMajor,
                        notran ? CblasNoTrans : (trans[0] == 'T' || trans[0] == 't' ? CblasTrans : CblasConjTrans),
                        n, n, kl, ku, &NEG_CONE, AB, ldab, &X[j * ldx], 1,
                        &CONE, work, 1);

            /* Compute componentwise relative backward error from formula
             *   max(i) ( abs(R(i)) / ( abs(op(A))*abs(X) + abs(B) )(i) )
             */
            for (i = 0; i < n; i++) {
                rwork[i] = cabs1f(B[i + j * ldb]);
            }

            /* Compute abs(op(A))*abs(X) + abs(B) */
            if (notran) {
                for (k = 0; k < n; k++) {
                    kk = ku - k;
                    xk = cabs1f(X[k + j * ldx]);
                    INT i_start = (k - ku > 0) ? k - ku : 0;
                    INT i_end = (k + kl < n - 1) ? k + kl : n - 1;
                    for (i = i_start; i <= i_end; i++) {
                        rwork[i] = rwork[i] + cabs1f(AB[kk + i + k * ldab]) * xk;
                    }
                }
            } else {
                for (k = 0; k < n; k++) {
                    s = ZERO;
                    kk = ku - k;
                    INT i_start = (k - ku > 0) ? k - ku : 0;
                    INT i_end = (k + kl < n - 1) ? k + kl : n - 1;
                    for (i = i_start; i <= i_end; i++) {
                        s = s + cabs1f(AB[kk + i + k * ldab]) * cabs1f(X[i + j * ldx]);
                    }
                    rwork[k] = rwork[k] + s;
                }
            }

            s = ZERO;
            for (i = 0; i < n; i++) {
                if (rwork[i] > safe2) {
                    s = (s > cabs1f(work[i]) / rwork[i]) ? s : cabs1f(work[i]) / rwork[i];
                } else {
                    s = (s > (cabs1f(work[i]) + safe1) / (rwork[i] + safe1))
                        ? s : (cabs1f(work[i]) + safe1) / (rwork[i] + safe1);
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
            cgbtrs(trans, n, kl, ku, 1, AFB, ldafb, ipiv, work, n, &linfo);
            cblas_caxpy(n, &CONE, work, 1, &X[j * ldx], 1);
            lstres = berr[j];
            count = count + 1;
        }

        /* Bound error from formula
         *   norm(X - XTRUE) / norm(X) .le. FERR =
         *   norm( abs(inv(op(A)))* ( abs(R) + NZ*EPS*( abs(op(A))*abs(X)+abs(B) ))) / norm(X)
         */
        for (i = 0; i < n; i++) {
            if (rwork[i] > safe2) {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i];
            } else {
                rwork[i] = cabs1f(work[i]) + nz * eps * rwork[i] + safe1;
            }
        }

        kase = 0;
        while (1) {
            clacn2(n, &work[n], work, &ferr[j], &kase, isave);
            if (kase == 0) {
                break;
            }
            if (kase == 1) {
                /* Multiply by diag(W)*inv(op(A)**H) */
                cgbtrs(transt, n, kl, ku, 1, AFB, ldafb, ipiv, work, n, &linfo);
                for (i = 0; i < n; i++) {
                    work[i] = CMPLXF(rwork[i], 0.0f) * work[i];
                }
            } else {
                /* Multiply by inv(op(A))*diag(W) */
                for (i = 0; i < n; i++) {
                    work[i] = CMPLXF(rwork[i], 0.0f) * work[i];
                }
                cgbtrs(transn, n, kl, ku, 1, AFB, ldafb, ipiv, work, n, &linfo);
            }
        }

        /* Normalize error */
        lstres = ZERO;
        for (i = 0; i < n; i++) {
            lstres = (lstres > cabs1f(X[i + j * ldx])) ? lstres : cabs1f(X[i + j * ldx]);
        }
        if (lstres != ZERO) {
            ferr[j] = ferr[j] / lstres;
        }
    }
}
