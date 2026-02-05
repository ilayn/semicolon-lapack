/**
 * @file dlaqr5.c
 * @brief DLAQR5 performs a single small-bulge multi-shift QR sweep.
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>
#include <math.h>

/**
 * DLAQR5, called by DLAQR0, performs a single small-bulge multi-shift QR sweep.
 *
 * @param[in] wantt   If nonzero, the quasi-triangular Schur factor is being computed.
 * @param[in] wantz   If nonzero, the orthogonal Schur factor is being computed.
 * @param[in] kacc22  Specifies the computation mode of far-from-diagonal orthogonal updates.
 *                    = 0: Does not accumulate reflections and does not use matrix-matrix multiply.
 *                    = 1 or 2: Accumulates reflections and uses matrix-matrix multiply.
 * @param[in] n       The order of the Hessenberg matrix H. n >= 0.
 * @param[in] ktop    First row/column of isolated diagonal block (0-based).
 * @param[in] kbot    Last row/column of isolated diagonal block (0-based).
 * @param[in] nshfts  Number of simultaneous shifts. Must be positive and even.
 * @param[in,out] sr  Double precision array, dimension (nshfts). Real parts of shifts.
 * @param[in,out] si  Double precision array, dimension (nshfts). Imaginary parts of shifts.
 * @param[in,out] H   Double precision array, dimension (ldh, n). The Hessenberg matrix.
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[in] iloz    First row of Z to which transformations must be applied (0-based).
 * @param[in] ihiz    Last row of Z to which transformations must be applied (0-based).
 * @param[in,out] Z   Double precision array, dimension (ldz, n). The orthogonal matrix.
 * @param[in] ldz     Leading dimension of Z. ldz >= n.
 * @param[out] V      Double precision array, dimension (ldv, nshfts/2).
 * @param[in] ldv     Leading dimension of V. ldv >= 3.
 * @param[out] U      Double precision array, dimension (ldu, 2*nshfts).
 * @param[in] ldu     Leading dimension of U. ldu >= 2*nshfts.
 * @param[in] nv      Number of rows in WV available for workspace. nv >= 1.
 * @param[out] WV     Double precision array, dimension (ldwv, 2*nshfts).
 * @param[in] ldwv    Leading dimension of WV. ldwv >= nv.
 * @param[in] nh      Number of columns in WH available for workspace. nh >= 1.
 * @param[out] WH     Double precision array, dimension (ldwh, nh).
 * @param[in] ldwh    Leading dimension of WH. ldwh >= 2*nshfts.
 */
SEMICOLON_API void dlaqr5(const int wantt, const int wantz, const int kacc22,
                          const int n, const int ktop, const int kbot,
                          const int nshfts, double* sr, double* si,
                          double* H, const int ldh,
                          const int iloz, const int ihiz,
                          double* Z, const int ldz,
                          double* V, const int ldv,
                          double* U, const int ldu,
                          const int nv, double* WV, const int ldwv,
                          const int nh, double* WH, const int ldwh)
{
    /* Parameters */
    const double zero = 0.0;
    const double one = 1.0;

    /* Local scalars */
    double alpha, beta, h11, h12, h21, h22, refsum;
    double safmin, scl, smlnum, swap, t1, t2, t3, tst1, tst2, ulp;
    int i, i2, i4, incol, j, jbot, jcol, jlen, jrow, jtop;
    int k, k1, kdu, kms, krcol, m, m22, mbot, mtop, nbmps, ndcol, ns, nu;
    int accum, bmp22;

    /* Local array */
    double vt[3];

    /* If there are no shifts, then there is nothing to do */
    if (nshfts < 2)
        return;

    /* If the active block is empty or 1-by-1, then there is nothing to do */
    if (ktop >= kbot)
        return;

    /* Shuffle shifts into pairs of real shifts and pairs of complex
     * conjugate shifts assuming complex conjugate shifts are already
     * adjacent to one another */
    for (i = 0; i < nshfts - 2; i += 2) {
        if (si[i] != -si[i + 1]) {
            swap = sr[i];
            sr[i] = sr[i + 1];
            sr[i + 1] = sr[i + 2];
            sr[i + 2] = swap;

            swap = si[i];
            si[i] = si[i + 1];
            si[i + 1] = si[i + 2];
            si[i + 2] = swap;
        }
    }

    /* NSHFTS is supposed to be even, but if it is odd, then simply
     * reduce it by one */
    ns = nshfts - (nshfts % 2);

    /* Machine constants for deflation */
    safmin = dlamch("Safe minimum");
    ulp = dlamch("Precision");
    smlnum = safmin * ((double)n / ulp);

    /* Use accumulated reflections to update far-from-diagonal entries? */
    accum = (kacc22 == 1) || (kacc22 == 2);

    /* Clear trash */
    if (ktop + 2 <= kbot)
        H[(ktop + 2) + ktop * ldh] = zero;

    /* NBMPS = number of 2-shift bulges in the chain */
    nbmps = ns / 2;

    /* KDU = width of slab */
    kdu = 4 * nbmps;

    /* Create and chase chains of NBMPS bulges */
    for (incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps) {

        /* JTOP = Index from which updates from the right start */
        if (accum) {
            jtop = ktop > incol ? ktop : incol;
        } else if (wantt) {
            jtop = 0;
        } else {
            jtop = ktop;
        }

        ndcol = incol + kdu;
        if (accum)
            dlaset("ALL", kdu, kdu, zero, one, U, ldu);

        /* Near-the-diagonal bulge chase */
        for (krcol = incol; krcol <= (incol + 2 * nbmps - 1 < kbot - 2 ?
                                      incol + 2 * nbmps - 1 : kbot - 2); krcol++) {

            /* Bulges number MTOP to MBOT are active double implicit shift bulges */
            mtop = 1 > (ktop - krcol) / 2 + 1 ? 1 : (ktop - krcol) / 2 + 1;
            mbot = nbmps < (kbot - krcol - 1) / 2 ? nbmps : (kbot - krcol - 1) / 2;
            m22 = mbot + 1;
            bmp22 = (mbot < nbmps) && (krcol + 2 * (m22 - 1) == kbot - 2);

            /* Special case: 2-by-2 reflection at bottom treated separately */
            if (bmp22) {
                k = krcol + 2 * (m22 - 1);
                if (k == ktop - 1) {
                    dlaqr1(2, &H[(k + 1) + (k + 1) * ldh], ldh,
                           sr[2 * m22 - 2], si[2 * m22 - 2],
                           sr[2 * m22 - 1], si[2 * m22 - 1],
                           &V[(m22 - 1) * ldv]);
                    beta = V[(m22 - 1) * ldv];
                    dlarfg(2, &beta, &V[1 + (m22 - 1) * ldv], 1,
                           &V[(m22 - 1) * ldv]);
                } else {
                    beta = H[(k + 1) + k * ldh];
                    V[1 + (m22 - 1) * ldv] = H[(k + 2) + k * ldh];
                    dlarfg(2, &beta, &V[1 + (m22 - 1) * ldv], 1,
                           &V[(m22 - 1) * ldv]);
                    H[(k + 1) + k * ldh] = beta;
                    H[(k + 2) + k * ldh] = zero;
                }

                /* Perform update from right within computational window */
                t1 = V[(m22 - 1) * ldv];
                t2 = t1 * V[1 + (m22 - 1) * ldv];
                for (j = jtop; j <= (kbot < k + 3 ? kbot : k + 3); j++) {
                    refsum = H[j + (k + 1) * ldh] +
                             V[1 + (m22 - 1) * ldv] * H[j + (k + 2) * ldh];
                    H[j + (k + 1) * ldh] = H[j + (k + 1) * ldh] - refsum * t1;
                    H[j + (k + 2) * ldh] = H[j + (k + 2) * ldh] - refsum * t2;
                }

                /* Perform update from left within computational window */
                if (accum) {
                    jbot = ndcol < kbot ? ndcol : kbot;
                } else if (wantt) {
                    jbot = n - 1;
                } else {
                    jbot = kbot;
                }
                t1 = V[(m22 - 1) * ldv];
                t2 = t1 * V[1 + (m22 - 1) * ldv];
                for (j = k + 1; j <= jbot; j++) {
                    refsum = H[(k + 1) + j * ldh] +
                             V[1 + (m22 - 1) * ldv] * H[(k + 2) + j * ldh];
                    H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - refsum * t1;
                    H[(k + 2) + j * ldh] = H[(k + 2) + j * ldh] - refsum * t2;
                }

                /* Convergence test */
                if (k >= ktop) {
                    if (H[(k + 1) + k * ldh] != zero) {
                        tst1 = fabs(H[k + k * ldh]) + fabs(H[(k + 1) + (k + 1) * ldh]);
                        if (tst1 == zero) {
                            if (k >= ktop + 1)
                                tst1 = tst1 + fabs(H[k + (k - 1) * ldh]);
                            if (k >= ktop + 2)
                                tst1 = tst1 + fabs(H[k + (k - 2) * ldh]);
                            if (k >= ktop + 3)
                                tst1 = tst1 + fabs(H[k + (k - 3) * ldh]);
                            if (k <= kbot - 2)
                                tst1 = tst1 + fabs(H[(k + 2) + (k + 1) * ldh]);
                            if (k <= kbot - 3)
                                tst1 = tst1 + fabs(H[(k + 3) + (k + 1) * ldh]);
                            if (k <= kbot - 4)
                                tst1 = tst1 + fabs(H[(k + 4) + (k + 1) * ldh]);
                        }
                        if (fabs(H[(k + 1) + k * ldh]) <=
                            (smlnum > ulp * tst1 ? smlnum : ulp * tst1)) {
                            h12 = fabs(H[(k + 1) + k * ldh]) > fabs(H[k + (k + 1) * ldh]) ?
                                  fabs(H[(k + 1) + k * ldh]) : fabs(H[k + (k + 1) * ldh]);
                            h21 = fabs(H[(k + 1) + k * ldh]) < fabs(H[k + (k + 1) * ldh]) ?
                                  fabs(H[(k + 1) + k * ldh]) : fabs(H[k + (k + 1) * ldh]);
                            h11 = fabs(H[(k + 1) + (k + 1) * ldh]) >
                                  fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                                  fabs(H[(k + 1) + (k + 1) * ldh]) :
                                  fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                            h22 = fabs(H[(k + 1) + (k + 1) * ldh]) <
                                  fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                                  fabs(H[(k + 1) + (k + 1) * ldh]) :
                                  fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                            scl = h11 + h12;
                            tst2 = h22 * (h11 / scl);
                            if (tst2 == zero || h21 * (h12 / scl) <=
                                (smlnum > ulp * tst2 ? smlnum : ulp * tst2)) {
                                H[(k + 1) + k * ldh] = zero;
                            }
                        }
                    }
                }

                /* Accumulate orthogonal transformations */
                if (accum) {
                    kms = k - incol;
                    t1 = V[(m22 - 1) * ldv];
                    t2 = t1 * V[1 + (m22 - 1) * ldv];
                    for (j = (1 > ktop - incol ? 1 : ktop - incol) - 1; j < kdu; j++) {
                        refsum = U[j + kms * ldu] +
                                 V[1 + (m22 - 1) * ldv] * U[j + (kms + 1) * ldu];
                        U[j + kms * ldu] = U[j + kms * ldu] - refsum * t1;
                        U[j + (kms + 1) * ldu] = U[j + (kms + 1) * ldu] - refsum * t2;
                    }
                } else if (wantz) {
                    t1 = V[(m22 - 1) * ldv];
                    t2 = t1 * V[1 + (m22 - 1) * ldv];
                    for (j = iloz; j <= ihiz; j++) {
                        refsum = Z[j + (k + 1) * ldz] +
                                 V[1 + (m22 - 1) * ldv] * Z[j + (k + 2) * ldz];
                        Z[j + (k + 1) * ldz] = Z[j + (k + 1) * ldz] - refsum * t1;
                        Z[j + (k + 2) * ldz] = Z[j + (k + 2) * ldz] - refsum * t2;
                    }
                }
            }

            /* Normal case: Chain of 3-by-3 reflections */
            for (m = mbot; m >= mtop; m--) {
                k = krcol + 2 * (m - 1);
                if (k == ktop - 1) {
                    dlaqr1(3, &H[ktop + ktop * ldh], ldh,
                           sr[2 * m - 2], si[2 * m - 2],
                           sr[2 * m - 1], si[2 * m - 1],
                           &V[(m - 1) * ldv]);
                    alpha = V[(m - 1) * ldv];
                    dlarfg(3, &alpha, &V[1 + (m - 1) * ldv], 1,
                           &V[(m - 1) * ldv]);
                } else {
                    /* Perform delayed transformation of row below Mth bulge.
                     * Exploit fact that first two elements of row are zero. */
                    t1 = V[(m - 1) * ldv];
                    t2 = t1 * V[1 + (m - 1) * ldv];
                    t3 = t1 * V[2 + (m - 1) * ldv];
                    refsum = V[2 + (m - 1) * ldv] * H[(k + 3) + (k + 2) * ldh];
                    H[(k + 3) + k * ldh] = -refsum * t1;
                    H[(k + 3) + (k + 1) * ldh] = -refsum * t2;
                    H[(k + 3) + (k + 2) * ldh] = H[(k + 3) + (k + 2) * ldh] - refsum * t3;

                    /* Calculate reflection to move Mth bulge one step */
                    beta = H[(k + 1) + k * ldh];
                    V[1 + (m - 1) * ldv] = H[(k + 2) + k * ldh];
                    V[2 + (m - 1) * ldv] = H[(k + 3) + k * ldh];
                    dlarfg(3, &beta, &V[1 + (m - 1) * ldv], 1,
                           &V[(m - 1) * ldv]);

                    /* A Bulge may collapse because of vigilant deflation or
                     * destructive underflow */
                    if (H[(k + 3) + k * ldh] != zero ||
                        H[(k + 3) + (k + 1) * ldh] != zero ||
                        H[(k + 3) + (k + 2) * ldh] == zero) {
                        /* Typical case: not collapsed (yet) */
                        H[(k + 1) + k * ldh] = beta;
                        H[(k + 2) + k * ldh] = zero;
                        H[(k + 3) + k * ldh] = zero;
                    } else {
                        /* Atypical case: collapsed. Attempt to reintroduce
                         * ignoring H(k+1,k) and H(k+2,k) */
                        dlaqr1(3, &H[(k + 1) + (k + 1) * ldh], ldh,
                               sr[2 * m - 2], si[2 * m - 2],
                               sr[2 * m - 1], si[2 * m - 1], vt);
                        alpha = vt[0];
                        dlarfg(3, &alpha, &vt[1], 1, &vt[0]);
                        t1 = vt[0];
                        t2 = t1 * vt[1];
                        t3 = t1 * vt[2];
                        refsum = H[(k + 1) + k * ldh] + vt[1] * H[(k + 2) + k * ldh];

                        if (fabs(H[(k + 2) + k * ldh] - refsum * t2) +
                            fabs(refsum * t3) > ulp * (fabs(H[k + k * ldh]) +
                            fabs(H[(k + 1) + (k + 1) * ldh]) +
                            fabs(H[(k + 2) + (k + 2) * ldh]))) {
                            /* Starting a new bulge here would create
                             * non-negligible fill */
                            H[(k + 1) + k * ldh] = beta;
                            H[(k + 2) + k * ldh] = zero;
                            H[(k + 3) + k * ldh] = zero;
                        } else {
                            /* Replace the old reflector with the new one */
                            H[(k + 1) + k * ldh] = H[(k + 1) + k * ldh] - refsum * t1;
                            H[(k + 2) + k * ldh] = zero;
                            H[(k + 3) + k * ldh] = zero;
                            V[(m - 1) * ldv] = vt[0];
                            V[1 + (m - 1) * ldv] = vt[1];
                            V[2 + (m - 1) * ldv] = vt[2];
                        }
                    }
                }

                /* Apply reflection from the right and first column of
                 * update from the left */
                t1 = V[(m - 1) * ldv];
                t2 = t1 * V[1 + (m - 1) * ldv];
                t3 = t1 * V[2 + (m - 1) * ldv];
                for (j = jtop; j <= (kbot < k + 3 ? kbot : k + 3); j++) {
                    refsum = H[j + (k + 1) * ldh] +
                             V[1 + (m - 1) * ldv] * H[j + (k + 2) * ldh] +
                             V[2 + (m - 1) * ldv] * H[j + (k + 3) * ldh];
                    H[j + (k + 1) * ldh] = H[j + (k + 1) * ldh] - refsum * t1;
                    H[j + (k + 2) * ldh] = H[j + (k + 2) * ldh] - refsum * t2;
                    H[j + (k + 3) * ldh] = H[j + (k + 3) * ldh] - refsum * t3;
                }

                /* Perform update from left for subsequent column */
                refsum = H[(k + 1) + (k + 1) * ldh] +
                         V[1 + (m - 1) * ldv] * H[(k + 2) + (k + 1) * ldh] +
                         V[2 + (m - 1) * ldv] * H[(k + 3) + (k + 1) * ldh];
                H[(k + 1) + (k + 1) * ldh] = H[(k + 1) + (k + 1) * ldh] - refsum * t1;
                H[(k + 2) + (k + 1) * ldh] = H[(k + 2) + (k + 1) * ldh] - refsum * t2;
                H[(k + 3) + (k + 1) * ldh] = H[(k + 3) + (k + 1) * ldh] - refsum * t3;

                /* Convergence test */
                if (k < ktop)
                    continue;

                if (H[(k + 1) + k * ldh] != zero) {
                    tst1 = fabs(H[k + k * ldh]) + fabs(H[(k + 1) + (k + 1) * ldh]);
                    if (tst1 == zero) {
                        if (k >= ktop + 1)
                            tst1 = tst1 + fabs(H[k + (k - 1) * ldh]);
                        if (k >= ktop + 2)
                            tst1 = tst1 + fabs(H[k + (k - 2) * ldh]);
                        if (k >= ktop + 3)
                            tst1 = tst1 + fabs(H[k + (k - 3) * ldh]);
                        if (k <= kbot - 2)
                            tst1 = tst1 + fabs(H[(k + 2) + (k + 1) * ldh]);
                        if (k <= kbot - 3)
                            tst1 = tst1 + fabs(H[(k + 3) + (k + 1) * ldh]);
                        if (k <= kbot - 4)
                            tst1 = tst1 + fabs(H[(k + 4) + (k + 1) * ldh]);
                    }
                    if (fabs(H[(k + 1) + k * ldh]) <=
                        (smlnum > ulp * tst1 ? smlnum : ulp * tst1)) {
                        h12 = fabs(H[(k + 1) + k * ldh]) > fabs(H[k + (k + 1) * ldh]) ?
                              fabs(H[(k + 1) + k * ldh]) : fabs(H[k + (k + 1) * ldh]);
                        h21 = fabs(H[(k + 1) + k * ldh]) < fabs(H[k + (k + 1) * ldh]) ?
                              fabs(H[(k + 1) + k * ldh]) : fabs(H[k + (k + 1) * ldh]);
                        h11 = fabs(H[(k + 1) + (k + 1) * ldh]) >
                              fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                              fabs(H[(k + 1) + (k + 1) * ldh]) :
                              fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                        h22 = fabs(H[(k + 1) + (k + 1) * ldh]) <
                              fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                              fabs(H[(k + 1) + (k + 1) * ldh]) :
                              fabs(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                        scl = h11 + h12;
                        tst2 = h22 * (h11 / scl);
                        if (tst2 == zero || h21 * (h12 / scl) <=
                            (smlnum > ulp * tst2 ? smlnum : ulp * tst2)) {
                            H[(k + 1) + k * ldh] = zero;
                        }
                    }
                }
            }

            /* Multiply H by reflections from the left */
            if (accum) {
                jbot = ndcol < kbot ? ndcol : kbot;
            } else if (wantt) {
                jbot = n - 1;
            } else {
                jbot = kbot;
            }

            for (m = mbot; m >= mtop; m--) {
                k = krcol + 2 * (m - 1);
                t1 = V[(m - 1) * ldv];
                t2 = t1 * V[1 + (m - 1) * ldv];
                t3 = t1 * V[2 + (m - 1) * ldv];
                for (j = (ktop > krcol + 2 * m ? ktop : krcol + 2 * m); j <= jbot; j++) {
                    refsum = H[(k + 1) + j * ldh] +
                             V[1 + (m - 1) * ldv] * H[(k + 2) + j * ldh] +
                             V[2 + (m - 1) * ldv] * H[(k + 3) + j * ldh];
                    H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - refsum * t1;
                    H[(k + 2) + j * ldh] = H[(k + 2) + j * ldh] - refsum * t2;
                    H[(k + 3) + j * ldh] = H[(k + 3) + j * ldh] - refsum * t3;
                }
            }

            /* Accumulate orthogonal transformations */
            if (accum) {
                /* Accumulate U */
                for (m = mbot; m >= mtop; m--) {
                    k = krcol + 2 * (m - 1);
                    kms = k - incol;
                    i2 = 1 > ktop - incol ? 1 : ktop - incol;
                    i2 = i2 > kms - (krcol - incol) + 1 ? i2 : kms - (krcol - incol) + 1;
                    i4 = kdu < krcol + 2 * (mbot - 1) - incol + 5 ?
                         kdu : krcol + 2 * (mbot - 1) - incol + 5;
                    t1 = V[(m - 1) * ldv];
                    t2 = t1 * V[1 + (m - 1) * ldv];
                    t3 = t1 * V[2 + (m - 1) * ldv];
                    for (j = i2 - 1; j < i4; j++) {
                        refsum = U[j + kms * ldu] +
                                 V[1 + (m - 1) * ldv] * U[j + (kms + 1) * ldu] +
                                 V[2 + (m - 1) * ldv] * U[j + (kms + 2) * ldu];
                        U[j + kms * ldu] = U[j + kms * ldu] - refsum * t1;
                        U[j + (kms + 1) * ldu] = U[j + (kms + 1) * ldu] - refsum * t2;
                        U[j + (kms + 2) * ldu] = U[j + (kms + 2) * ldu] - refsum * t3;
                    }
                }
            } else if (wantz) {
                /* Update Z now by multiplying by reflections from the right */
                for (m = mbot; m >= mtop; m--) {
                    k = krcol + 2 * (m - 1);
                    t1 = V[(m - 1) * ldv];
                    t2 = t1 * V[1 + (m - 1) * ldv];
                    t3 = t1 * V[2 + (m - 1) * ldv];
                    for (j = iloz; j <= ihiz; j++) {
                        refsum = Z[j + (k + 1) * ldz] +
                                 V[1 + (m - 1) * ldv] * Z[j + (k + 2) * ldz] +
                                 V[2 + (m - 1) * ldv] * Z[j + (k + 3) * ldz];
                        Z[j + (k + 1) * ldz] = Z[j + (k + 1) * ldz] - refsum * t1;
                        Z[j + (k + 2) * ldz] = Z[j + (k + 2) * ldz] - refsum * t2;
                        Z[j + (k + 3) * ldz] = Z[j + (k + 3) * ldz] - refsum * t3;
                    }
                }
            }
        }

        /* Use U (if accumulated) to update far-from-diagonal entries in H */
        if (accum) {
            if (wantt) {
                jtop = 0;
                jbot = n - 1;
            } else {
                jtop = ktop;
                jbot = kbot;
            }
            k1 = 1 > ktop - incol ? 1 : ktop - incol;
            nu = (kdu - (0 > ndcol - kbot ? 0 : ndcol - kbot)) - k1 + 1;

            /* Horizontal Multiply */
            for (jcol = (ndcol < kbot ? ndcol : kbot) + 1; jcol <= jbot; jcol += nh) {
                jlen = nh < jbot - jcol + 1 ? nh : jbot - jcol + 1;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                           nu, jlen, nu, one, &U[(k1 - 1) + (k1 - 1) * ldu], ldu,
                           &H[(incol + k1) + jcol * ldh], ldh, zero, WH, ldwh);
                dlacpy("ALL", nu, jlen, WH, ldwh, &H[(incol + k1) + jcol * ldh], ldh);
            }

            /* Vertical multiply */
            for (jrow = jtop; jrow <= (ktop > incol ? ktop : incol) - 1; jrow += nv) {
                jlen = nv < (ktop > incol ? ktop : incol) - jrow ?
                       nv : (ktop > incol ? ktop : incol) - jrow;
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                           jlen, nu, nu, one, &H[jrow + (incol + k1) * ldh], ldh,
                           &U[(k1 - 1) + (k1 - 1) * ldu], ldu, zero, WV, ldwv);
                dlacpy("ALL", jlen, nu, WV, ldwv, &H[jrow + (incol + k1) * ldh], ldh);
            }

            /* Z multiply (also vertical) */
            if (wantz) {
                for (jrow = iloz; jrow <= ihiz; jrow += nv) {
                    jlen = nv < ihiz - jrow + 1 ? nv : ihiz - jrow + 1;
                    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                               jlen, nu, nu, one, &Z[jrow + (incol + k1) * ldz], ldz,
                               &U[(k1 - 1) + (k1 - 1) * ldu], ldu, zero, WV, ldwv);
                    dlacpy("ALL", jlen, nu, WV, ldwv, &Z[jrow + (incol + k1) * ldz], ldz);
                }
            }
        }
    }
}
