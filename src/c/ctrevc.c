/**
 * @file ctrevc.c
 * @brief CTREVC computes eigenvectors of a complex upper triangular matrix.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"
#include <math.h>
#include <float.h>

/**
 * CTREVC computes some or all of the right and/or left eigenvectors of
 * a complex upper triangular matrix T.
 * Matrices of this type are produced by the Schur factorization of
 * a complex general matrix:  A = Q*T*Q**H, as computed by CHSEQR.
 *
 * The right eigenvector x and the left eigenvector y of T corresponding
 * to an eigenvalue w are defined by:
 *
 *    T*x = w*x,     (y**H)*T = w*(y**H)
 *
 * where y**H denotes the conjugate transpose of the vector y.
 * The eigenvalues are not input to this routine, but are read directly
 * from the diagonal of T.
 *
 * This routine returns the matrices X and/or Y of right and left
 * eigenvectors of T, or the products Q*X and/or Q*Y, where Q is an
 * input matrix.  If Q is the unitary factor that reduces a matrix A to
 * Schur form T, then Q*X and Q*Y are the matrices of right and left
 * eigenvectors of A.
 *
 * @param[in] side    'R': compute right eigenvectors only;
 *                    'L': compute left eigenvectors only;
 *                    'B': compute both right and left eigenvectors.
 * @param[in] howmny  'A': compute all right and/or left eigenvectors;
 *                    'B': compute all, backtransformed by VR and/or VL;
 *                    'S': compute selected eigenvectors (as indicated by select).
 * @param[in] select  Integer array, dimension (n).
 *                    If howmny = 'S', select specifies which eigenvectors to compute.
 *                    Nonzero = selected. Not referenced if howmny = 'A' or 'B'.
 * @param[in] n       The order of the matrix T. n >= 0.
 * @param[in,out] T   Single complex array, dimension (ldt, n).
 *                    The upper triangular matrix T. T is modified, but restored
 *                    on exit.
 * @param[in] ldt     The leading dimension of T. ldt >= max(1, n).
 * @param[in,out] VL  Single complex array, dimension (ldvl, mm).
 *                    On entry, if howmny = 'B', must contain an n-by-n matrix Q.
 *                    On exit, contains the left eigenvectors.
 *                    Not referenced if side = 'R'.
 * @param[in] ldvl    The leading dimension of VL. ldvl >= 1, and if
 *                    side = 'L' or 'B', ldvl >= n.
 * @param[in,out] VR  Single complex array, dimension (ldvr, mm).
 *                    On entry, if howmny = 'B', must contain an n-by-n matrix Q.
 *                    On exit, contains the right eigenvectors.
 *                    Not referenced if side = 'L'.
 * @param[in] ldvr    The leading dimension of VR. ldvr >= 1, and if
 *                    side = 'R' or 'B', ldvr >= n.
 * @param[in] mm      The number of columns in VL and/or VR. mm >= m.
 * @param[out] m      The number of columns actually used to store eigenvectors.
 * @param[out] work   Single complex array, dimension (2*n).
 * @param[out] rwork  Single precision array, dimension (n).
 * @param[out] info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value.
 */
void ctrevc(const char* side, const char* howmny, INT* select, const INT n,
            c64* T, const INT ldt, c64* VL, const INT ldvl,
            c64* VR, const INT ldvr, const INT mm, INT* m,
            c64* work, f32* rwork, INT* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CMZERO = CMPLXF(0.0f, 0.0f);
    const c64 CMONE = CMPLXF(1.0f, 0.0f);

    INT allv, bothv, leftv, over, rightv, somev;
    INT i, ii, is, j, k, ki;
    f32 remax, scale, smin, smlnum, ulp, unfl;

    /* Decode and test the input parameters */
    bothv = (side[0] == 'B' || side[0] == 'b');
    rightv = (side[0] == 'R' || side[0] == 'r') || bothv;
    leftv = (side[0] == 'L' || side[0] == 'l') || bothv;

    allv = (howmny[0] == 'A' || howmny[0] == 'a');
    over = (howmny[0] == 'B' || howmny[0] == 'b');
    somev = (howmny[0] == 'S' || howmny[0] == 's');

    /* Set M to the number of columns required to store the selected
     * eigenvectors. */
    if (somev) {
        *m = 0;
        for (j = 0; j < n; j++) {
            if (select[j])
                (*m)++;
        }
    } else {
        *m = n;
    }

    *info = 0;
    if (!rightv && !leftv) {
        *info = -1;
    } else if (!allv && !over && !somev) {
        *info = -2;
    } else if (n < 0) {
        *info = -4;
    } else if (ldt < (1 > n ? 1 : n)) {
        *info = -6;
    } else if (ldvl < 1 || (leftv && ldvl < n)) {
        *info = -8;
    } else if (ldvr < 1 || (rightv && ldvr < n)) {
        *info = -10;
    } else if (mm < *m) {
        *info = -11;
    }
    if (*info != 0) {
        xerbla("CTREVC", -(*info));
        return;
    }

    /* Quick return if possible. */
    if (n == 0)
        return;

    /* Set the constants to control overflow. */
    unfl = slamch("Safe minimum");
    ulp = slamch("Precision");
    smlnum = unfl * ((f32)n / ulp);

    /* Store the diagonal elements of T in working array WORK. */
    for (i = 0; i < n; i++) {
        work[i + n] = T[i + i * ldt];
    }

    /* Compute 1-norm of each column of strictly upper triangular
     * part of T to control overflow in triangular solver. */
    rwork[0] = ZERO;
    for (j = 1; j < n; j++) {
        rwork[j] = cblas_scasum(j, &T[j * ldt], 1);
    }

    if (rightv) {
        /* Compute right eigenvectors. */
        is = *m - 1;
        for (ki = n - 1; ki >= 0; ki--) {

            if (somev) {
                if (!select[ki])
                    goto L80;
            }
            smin = cabs1f(T[ki + ki * ldt]) * ulp;
            if (smin < smlnum) smin = smlnum;

            work[0] = CMONE;

            /* Form right-hand side. */
            for (k = 0; k < ki; k++) {
                work[k] = -T[k + ki * ldt];
            }

            /* Solve the triangular system:
             *    (T(1:KI-1,1:KI-1) - T(KI,KI))*X = SCALE*WORK. */
            for (k = 0; k < ki; k++) {
                T[k + k * ldt] = T[k + k * ldt] - T[ki + ki * ldt];
                if (cabs1f(T[k + k * ldt]) < smin)
                    T[k + k * ldt] = CMPLXF(smin, 0.0f);
            }

            if (ki > 0) {
                clatrs("Upper", "No transpose", "Non-unit", "Y",
                       ki, T, ldt, &work[0], &scale, rwork,
                       info);
                work[ki] = CMPLXF(scale, 0.0f);
            }

            /* Copy the vector x or Q*x to VR and normalize. */
            if (!over) {
                cblas_ccopy(ki + 1, &work[0], 1, &VR[is * ldvr], 1);

                ii = cblas_icamax(ki + 1, &VR[is * ldvr], 1);
                remax = ONE / cabs1f(VR[ii + is * ldvr]);
                cblas_csscal(ki + 1, remax, &VR[is * ldvr], 1);

                for (k = ki + 1; k < n; k++) {
                    VR[k + is * ldvr] = CMZERO;
                }
            } else {
                if (ki > 0) {
                    c64 scale_c = CMPLXF(scale, 0.0f);
                    cblas_cgemv(CblasColMajor, CblasNoTrans, n, ki,
                                &CMONE, VR, ldvr,
                                &work[0], 1, &scale_c, &VR[ki * ldvr], 1);
                }

                ii = cblas_icamax(n, &VR[ki * ldvr], 1);
                remax = ONE / cabs1f(VR[ii + ki * ldvr]);
                cblas_csscal(n, remax, &VR[ki * ldvr], 1);
            }

            /* Set back the original diagonal elements of T. */
            for (k = 0; k < ki; k++) {
                T[k + k * ldt] = work[k + n];
            }

            is = is - 1;
L80:        ;
        }
    }

    if (leftv) {
        /* Compute left eigenvectors. */
        is = 0;
        for (ki = 0; ki < n; ki++) {

            if (somev) {
                if (!select[ki])
                    goto L130;
            }
            smin = cabs1f(T[ki + ki * ldt]) * ulp;
            if (smin < smlnum) smin = smlnum;

            work[n - 1] = CMONE;

            /* Form right-hand side. */
            for (k = ki + 1; k < n; k++) {
                work[k] = -conjf(T[ki + k * ldt]);
            }

            /* Solve the triangular system:
             *    (T(KI+1:N,KI+1:N) - T(KI,KI))**H * X = SCALE*WORK. */
            for (k = ki + 1; k < n; k++) {
                T[k + k * ldt] = T[k + k * ldt] - T[ki + ki * ldt];
                if (cabs1f(T[k + k * ldt]) < smin)
                    T[k + k * ldt] = CMPLXF(smin, 0.0f);
            }

            if (ki < n - 1) {
                clatrs("Upper", "Conjugate transpose", "Non-unit",
                       "Y", n - ki - 1, &T[(ki + 1) + (ki + 1) * ldt], ldt,
                       &work[ki + 1], &scale, rwork, info);
                work[ki] = CMPLXF(scale, 0.0f);
            }

            /* Copy the vector x or Q*x to VL and normalize. */
            if (!over) {
                cblas_ccopy(n - ki, &work[ki], 1, &VL[ki + is * ldvl], 1);

                ii = cblas_icamax(n - ki, &VL[ki + is * ldvl], 1) + ki;
                remax = ONE / cabs1f(VL[ii + is * ldvl]);
                cblas_csscal(n - ki, remax, &VL[ki + is * ldvl], 1);

                for (k = 0; k < ki; k++) {
                    VL[k + is * ldvl] = CMZERO;
                }
            } else {
                if (ki < n - 1) {
                    c64 scale_c = CMPLXF(scale, 0.0f);
                    cblas_cgemv(CblasColMajor, CblasNoTrans, n, n - ki - 1,
                                &CMONE, &VL[(ki + 1) * ldvl], ldvl,
                                &work[ki + 1], 1, &scale_c,
                                &VL[ki * ldvl], 1);
                }

                ii = cblas_icamax(n, &VL[ki * ldvl], 1);
                remax = ONE / cabs1f(VL[ii + ki * ldvl]);
                cblas_csscal(n, remax, &VL[ki * ldvl], 1);
            }

            /* Set back the original diagonal elements of T. */
            for (k = ki + 1; k < n; k++) {
                T[k + k * ldt] = work[k + n];
            }

            is = is + 1;
L130:       ;
        }
    }
}
