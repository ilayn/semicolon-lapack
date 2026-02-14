/**
 * @file shsein.c
 * @brief SHSEIN uses inverse iteration to find specified right and/or left
 *        eigenvectors of a real upper Hessenberg matrix H.
 */

#include <math.h>
#include "semicolon_lapack_single.h"

/**
 * SHSEIN uses inverse iteration to find specified right and/or left
 * eigenvectors of a real upper Hessenberg matrix H.
 *
 * The right eigenvector x and the left eigenvector y of the matrix H
 * corresponding to an eigenvalue w are defined by:
 *
 *              H * x = w * x,     y**h * H = w * y**h
 *
 * where y**h denotes the conjugate transpose of the vector y.
 *
 * Each eigenvector is normalized so that the element of largest
 * magnitude has magnitude 1; here the magnitude of a complex number
 * (x,y) is taken to be |x|+|y|.
 *
 * @param[in]     side    'R': compute right eigenvectors only;
 *                        'L': compute left eigenvectors only;
 *                        'B': compute both right and left eigenvectors.
 * @param[in]     eigsrc  'Q': eigenvalues were found using SHSEQR, so matrix
 *                             splitting can be exploited;
 *                        'N': no assumptions on correspondence between
 *                             eigenvalues and diagonal blocks.
 * @param[in]     initv   'N': no initial vectors supplied;
 *                        'U': user-supplied initial vectors in VL and/or VR.
 * @param[in,out] select  Array of dimension n. Specifies which eigenvectors
 *                        to compute. select[j] = 1 (nonzero) selects the
 *                        eigenvalue wr[j] + i*wi[j].
 * @param[in]     n       The order of the matrix H (n >= 0).
 * @param[in]     H       Upper Hessenberg matrix H. Array of dimension (ldh, n).
 * @param[in]     ldh     The leading dimension of H (ldh >= max(1,n)).
 * @param[in,out] wr      Array of dimension n. Real parts of eigenvalues.
 *                        May be perturbed on exit.
 * @param[in]     wi      Array of dimension n. Imaginary parts of eigenvalues.
 * @param[in,out] VL      Left eigenvectors. Array of dimension (ldvl, mm).
 * @param[in]     ldvl    Leading dimension of VL (ldvl >= 1; ldvl >= n if
 *                        computing left eigenvectors).
 * @param[in,out] VR      Right eigenvectors. Array of dimension (ldvr, mm).
 * @param[in]     ldvr    Leading dimension of VR (ldvr >= 1; ldvr >= n if
 *                        computing right eigenvectors).
 * @param[in]     mm      Number of columns in VL and/or VR (mm >= m).
 * @param[out]    m       Number of columns required to store the eigenvectors.
 * @param[out]    work    Workspace array of dimension (n+2)*n.
 * @param[out]    ifaill  Array of dimension mm. Convergence status for left
 *                        eigenvectors (0 if converged, k if eigenvector
 *                        corresponding to eigenvalue k failed).
 * @param[out]    ifailr  Array of dimension mm. Convergence status for right
 *                        eigenvectors.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an illegal value
 *                         - > 0: info is the number of eigenvectors which failed
 *                           to converge.
 */
void shsein(
    const char* side,
    const char* eigsrc,
    const char* initv,
    int* const restrict select,
    const int n,
    const f32* const restrict H,
    const int ldh,
    f32* const restrict wr,
    const f32* const restrict wi,
    f32* const restrict VL,
    const int ldvl,
    f32* const restrict VR,
    const int ldvr,
    const int mm,
    int* m,
    f32* const restrict work,
    int* const restrict ifaill,
    int* const restrict ifailr,
    int* info)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;

    int bothv, fromqr, leftv, noinit_flag, pair, rightv;
    int i, iinfo, k, kl, kln, kr, ksi, ksr, ldwork;
    f32 bignum, eps3 = 0.0f, hnorm, smlnum, ulp, unfl, wki, wkr;

    /* Decode and test the input parameters */
    bothv = (side[0] == 'B' || side[0] == 'b');
    rightv = (side[0] == 'R' || side[0] == 'r') || bothv;
    leftv = (side[0] == 'L' || side[0] == 'l') || bothv;

    fromqr = (eigsrc[0] == 'Q' || eigsrc[0] == 'q');

    noinit_flag = (initv[0] == 'N' || initv[0] == 'n');

    /* Set m to the number of columns required to store the selected
     * eigenvectors, and standardize the array select. */
    *m = 0;
    pair = 0;
    for (k = 0; k < n; k++) {
        if (pair) {
            pair = 0;
            select[k] = 0;
        } else {
            if (wi[k] == ZERO) {
                if (select[k]) {
                    (*m)++;
                }
            } else {
                pair = 1;
                if (select[k] || (k + 1 < n && select[k + 1])) {
                    select[k] = 1;
                    *m += 2;
                }
            }
        }
    }

    *info = 0;
    if (!rightv && !leftv) {
        *info = -1;
    } else if (!fromqr && !(eigsrc[0] == 'N' || eigsrc[0] == 'n')) {
        *info = -2;
    } else if (!noinit_flag && !(initv[0] == 'U' || initv[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -5;
    } else if (ldh < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldvl < 1 || (leftv && ldvl < n)) {
        *info = -11;
    } else if (ldvr < 1 || (rightv && ldvr < n)) {
        *info = -13;
    } else if (mm < *m) {
        *info = -14;
    }

    if (*info != 0) {
        xerbla("SHSEIN", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0) {
        return;
    }

    /* Set machine-dependent constants */
    unfl = slamch("S");
    ulp = slamch("P");
    smlnum = unfl * ((f32)n / ulp);
    bignum = (ONE - ulp) / smlnum;

    ldwork = n + 1;

    kl = 0;
    kln = -1;
    if (fromqr) {
        kr = -1;
    } else {
        kr = n - 1;
    }
    ksr = 0;

    for (k = 0; k < n; k++) {
        if (select[k]) {
            /* Compute eigenvector(s) corresponding to w(k) */

            if (fromqr) {
                /* If affiliation of eigenvalues is known, check whether
                 * the matrix splits.
                 *
                 * Determine kl and kr such that 0 <= kl <= k <= kr <= n-1
                 * and H(kl,kl-1) and H(kr+1,kr) are zero (or kl = 0 or kr = n-1).
                 *
                 * Then inverse iteration can be performed with the
                 * submatrix H(kl:n-1,kl:n-1) for a left eigenvector, and with
                 * the submatrix H(0:kr,0:kr) for a right eigenvector. */
                for (i = k; i > kl; i--) {
                    if (H[i + (i - 1) * ldh] == ZERO) {
                        break;
                    }
                }
                kl = i;
                if (k > kr) {
                    for (i = k; i < n - 1; i++) {
                        if (H[i + 1 + i * ldh] == ZERO) {
                            break;
                        }
                    }
                    kr = i;
                }
            }

            if (kl != kln) {
                kln = kl;

                /* Compute infinity-norm of submatrix H(kl:kr,kl:kr) if it
                 * has not been computed before. */
                hnorm = slanhs("I", kr - kl + 1, &H[kl + kl * ldh], ldh, work);
                if (sisnan(hnorm)) {
                    *info = -6;
                    return;
                } else if (hnorm > ZERO) {
                    eps3 = hnorm * ulp;
                } else {
                    eps3 = smlnum;
                }
            }

            /* Perturb eigenvalue if it is close to any previous
             * selected eigenvalues affiliated to the submatrix
             * H(kl:kr,kl:kr). Close roots are modified by eps3. */
            wkr = wr[k];
            wki = wi[k];
        L60:
            for (i = k - 1; i >= kl; i--) {
                if (select[i] && fabsf(wr[i] - wkr) + fabsf(wi[i] - wki) < eps3) {
                    wkr = wkr + eps3;
                    goto L60;
                }
            }
            wr[k] = wkr;

            pair = (wki != ZERO);
            if (pair) {
                ksi = ksr + 1;
            } else {
                ksi = ksr;
            }

            if (leftv) {
                /* Compute left eigenvector */
                slaein(0, noinit_flag, n - kl, &H[kl + kl * ldh], ldh,
                       wkr, wki, &VL[kl + ksr * ldvl], &VL[kl + ksi * ldvl],
                       work, ldwork, &work[n * n + n], eps3, smlnum, bignum,
                       &iinfo);
                if (iinfo > 0) {
                    if (pair) {
                        *info += 2;
                    } else {
                        (*info)++;
                    }
                    ifaill[ksr] = k + 1;  /* 1-based index for user */
                    ifaill[ksi] = k + 1;
                } else {
                    ifaill[ksr] = 0;
                    ifaill[ksi] = 0;
                }
                for (i = 0; i < kl; i++) {
                    VL[i + ksr * ldvl] = ZERO;
                }
                if (pair) {
                    for (i = 0; i < kl; i++) {
                        VL[i + ksi * ldvl] = ZERO;
                    }
                }
            }

            if (rightv) {
                /* Compute right eigenvector */
                slaein(1, noinit_flag, kr + 1, H, ldh, wkr, wki,
                       &VR[ksr * ldvr], &VR[ksi * ldvr],
                       work, ldwork, &work[n * n + n], eps3, smlnum, bignum,
                       &iinfo);
                if (iinfo > 0) {
                    if (pair) {
                        *info += 2;
                    } else {
                        (*info)++;
                    }
                    ifailr[ksr] = k + 1;  /* 1-based index for user */
                    ifailr[ksi] = k + 1;
                } else {
                    ifailr[ksr] = 0;
                    ifailr[ksi] = 0;
                }
                for (i = kr + 1; i < n; i++) {
                    VR[i + ksr * ldvr] = ZERO;
                }
                if (pair) {
                    for (i = kr + 1; i < n; i++) {
                        VR[i + ksi * ldvr] = ZERO;
                    }
                }
            }

            if (pair) {
                ksr += 2;
            } else {
                ksr++;
            }
        }
    }
}
