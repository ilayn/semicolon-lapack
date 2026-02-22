/**
 * @file zhsein.c
 * @brief ZHSEIN uses inverse iteration to find specified right and/or left
 *        eigenvectors of a complex upper Hessenberg matrix H.
 */

#include <math.h>
#include <complex.h>
#include "semicolon_lapack_complex_double.h"

/**
 * ZHSEIN uses inverse iteration to find specified right and/or left
 * eigenvectors of a complex upper Hessenberg matrix H.
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
 * @param[in]     eigsrc  'Q': eigenvalues were found using ZHSEQR, so matrix
 *                             splitting can be exploited;
 *                        'N': no assumptions on correspondence between
 *                             eigenvalues and diagonal blocks.
 * @param[in]     initv   'N': no initial vectors supplied;
 *                        'U': user-supplied initial vectors in VL and/or VR.
 * @param[in]     select  Array of dimension n. Specifies which eigenvectors
 *                        to compute. select[j] nonzero selects the
 *                        eigenvector corresponding to W(j).
 * @param[in]     n       The order of the matrix H (n >= 0).
 * @param[in]     H       Upper Hessenberg matrix. Double complex array,
 *                        dimension (ldh, n).
 *                        If a NaN is detected in H, the routine will return
 *                        with info=-6.
 * @param[in]     ldh     The leading dimension of H (ldh >= max(1,n)).
 * @param[in,out] W       Double complex array, dimension (n).
 *                        On entry, the eigenvalues of H.
 *                        On exit, the real parts of W may have been altered
 *                        since close eigenvalues are perturbed slightly in
 *                        searching for independent eigenvectors.
 * @param[in,out] VL      Left eigenvectors. Double complex array,
 *                        dimension (ldvl, mm).
 * @param[in]     ldvl    Leading dimension of VL (ldvl >= 1; ldvl >= n if
 *                        computing left eigenvectors).
 * @param[in,out] VR      Right eigenvectors. Double complex array,
 *                        dimension (ldvr, mm).
 * @param[in]     ldvr    Leading dimension of VR (ldvr >= 1; ldvr >= n if
 *                        computing right eigenvectors).
 * @param[in]     mm      Number of columns in VL and/or VR (mm >= m).
 * @param[out]    m       Number of columns required to store the eigenvectors.
 * @param[out]    work    Double complex workspace array, dimension (n*n).
 * @param[out]    rwork   Double precision array, dimension (n).
 * @param[out]    ifaill  Integer array, dimension (mm). Convergence status
 *                        for left eigenvectors (-1 if converged, k (0-based)
 *                        if eigenvector corresponding to eigenvalue k failed).
 *                        Not referenced if side = 'R'.
 * @param[out]    ifailr  Integer array, dimension (mm). Convergence status
 *                        for right eigenvectors (-1 if converged, k (0-based)
 *                        if eigenvector corresponding to eigenvalue k failed).
 *                        Not referenced if side = 'L'.
 * @param[out]    info
 *                         - = 0: successful exit
 *                         - < 0: if info = -i, the i-th argument had an
 *                           illegal value
 *                         - > 0: info is the number of eigenvectors which
 *                           failed to converge.
 */
void zhsein(
    const char* side,
    const char* eigsrc,
    const char* initv,
    const INT* restrict select,
    const INT n,
    const c128* restrict H,
    const INT ldh,
    c128* restrict W,
    c128* restrict VL,
    const INT ldvl,
    c128* restrict VR,
    const INT ldvr,
    const INT mm,
    INT* m,
    c128* restrict work,
    f64* restrict rwork,
    INT* restrict ifaill,
    INT* restrict ifailr,
    INT* info)
{
    const c128 ZERO = CMPLX(0.0, 0.0);
    const f64 RZERO = 0.0;

    INT bothv, fromqr, leftv, noinit, rightv;
    INT i, iinfo, k, kl, kln, kr, ks, ldwork;
    f64 eps3 = 0.0, hnorm, smlnum, ulp, unfl;
    c128 wk;

    /* Decode and test the input parameters */
    bothv = (side[0] == 'B' || side[0] == 'b');
    rightv = (side[0] == 'R' || side[0] == 'r') || bothv;
    leftv = (side[0] == 'L' || side[0] == 'l') || bothv;

    fromqr = (eigsrc[0] == 'Q' || eigsrc[0] == 'q');

    noinit = (initv[0] == 'N' || initv[0] == 'n');

    /* Set m to the number of columns required to store the selected
     * eigenvectors. */
    *m = 0;
    for (k = 0; k < n; k++) {
        if (select[k])
            (*m)++;
    }

    *info = 0;
    if (!rightv && !leftv) {
        *info = -1;
    } else if (!fromqr && !(eigsrc[0] == 'N' || eigsrc[0] == 'n')) {
        *info = -2;
    } else if (!noinit && !(initv[0] == 'U' || initv[0] == 'u')) {
        *info = -3;
    } else if (n < 0) {
        *info = -5;
    } else if (ldh < (n > 1 ? n : 1)) {
        *info = -7;
    } else if (ldvl < 1 || (leftv && ldvl < n)) {
        *info = -10;
    } else if (ldvr < 1 || (rightv && ldvr < n)) {
        *info = -12;
    } else if (mm < *m) {
        *info = -13;
    }

    if (*info != 0) {
        xerbla("ZHSEIN", -(*info));
        return;
    }

    /* Quick return if possible */
    if (n == 0)
        return;

    /* Set machine-dependent constants */
    unfl = dlamch("S");
    ulp = dlamch("P");
    smlnum = unfl * ((f64)n / ulp);

    ldwork = n;

    kl = 0;
    kln = -1;
    if (fromqr) {
        kr = -1;
    } else {
        kr = n - 1;
    }
    ks = 0;

    for (k = 0; k < n; k++) {
        if (select[k]) {

            /* Compute eigenvector(s) corresponding to W(k) */

            if (fromqr) {

                /* If affiliation of eigenvalues is known, check whether
                 * the matrix splits.
                 *
                 * Determine kl and kr such that 0 <= kl <= k <= kr <= n-1
                 * and H(kl,kl-1) and H(kr+1,kr) are zero (or kl = 0 or
                 * kr = n-1).
                 *
                 * Then inverse iteration can be performed with the
                 * submatrix H(kl:n-1,kl:n-1) for a left eigenvector, and with
                 * the submatrix H(0:kr,0:kr) for a right eigenvector. */

                for (i = k; i > kl; i--) {
                    if (H[i + (i - 1) * ldh] == ZERO)
                        break;
                }
                kl = i;
                if (k > kr) {
                    for (i = k; i < n - 1; i++) {
                        if (H[i + 1 + i * ldh] == ZERO)
                            break;
                    }
                    kr = i;
                }
            }

            if (kl != kln) {
                kln = kl;

                /* Compute infinity-norm of submatrix H(kl:kr,kl:kr) if it
                 * has not been computed before. */
                hnorm = zlanhs("I", kr - kl + 1, &H[kl + kl * ldh], ldh,
                               rwork);
                if (disnan(hnorm)) {
                    *info = -6;
                    return;
                } else if (hnorm > RZERO) {
                    eps3 = hnorm * ulp;
                } else {
                    eps3 = smlnum;
                }
            }

            /* Perturb eigenvalue if it is close to any previous
             * selected eigenvalues affiliated to the submatrix
             * H(kl:kr,kl:kr). Close roots are modified by eps3. */
            wk = W[k];
        L60:
            for (i = k - 1; i >= kl; i--) {
                if (select[i] && cabs1(W[i] - wk) < eps3) {
                    wk = wk + eps3;
                    goto L60;
                }
            }
            W[k] = wk;

            if (leftv) {

                /* Compute left eigenvector */
                zlaein(0, noinit, n - kl, &H[kl + kl * ldh], ldh,
                       wk, &VL[kl + ks * ldvl], work, ldwork, rwork, eps3,
                       smlnum, &iinfo);
                if (iinfo > 0) {
                    *info = *info + 1;
                    ifaill[ks] = k;
                } else {
                    ifaill[ks] = -1;
                }
                for (i = 0; i < kl; i++) {
                    VL[i + ks * ldvl] = ZERO;
                }
            }
            if (rightv) {

                /* Compute right eigenvector */
                zlaein(1, noinit, kr + 1, H, ldh, wk, &VR[ks * ldvr],
                       work, ldwork, rwork, eps3, smlnum, &iinfo);
                if (iinfo > 0) {
                    *info = *info + 1;
                    ifailr[ks] = k;
                } else {
                    ifailr[ks] = -1;
                }
                for (i = kr + 1; i < n; i++) {
                    VR[i + ks * ldvr] = ZERO;
                }
            }
            ks++;
        }
    }
}
