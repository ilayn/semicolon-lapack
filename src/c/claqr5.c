/**
 * @file claqr5.c
 * @brief CLAQR5 performs a single small-bulge multi-shift QR sweep.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include "semicolon_cblas.h"
#include <math.h>

/**
 * CLAQR5, called by CLAQR0, performs a single small-bulge multi-shift QR sweep.
 *
 * @param[in] wantt   If nonzero, the triangular Schur factor is being computed.
 * @param[in] wantz   If nonzero, the unitary Schur factor is being computed.
 * @param[in] kacc22  Specifies the computation mode of far-from-diagonal
 *                    unitary updates (0, 1, or 2).
 * @param[in] n       The order of the Hessenberg matrix H. n >= 0.
 * @param[in] ktop    First row/column of isolated diagonal block (0-based).
 * @param[in] kbot    Last row/column of isolated diagonal block (0-based).
 * @param[in] nshfts  Number of simultaneous shifts. Must be positive and even.
 * @param[in,out] S   Complex array, dimension (nshfts). The shifts.
 * @param[in,out] H   Complex array, dimension (ldh, n). The Hessenberg matrix.
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[in] iloz    First row of Z to which transformations must be applied (0-based).
 * @param[in] ihiz    Last row of Z to which transformations must be applied (0-based).
 * @param[in,out] Z   Complex array, dimension (ldz, n). The unitary matrix.
 * @param[in] ldz     Leading dimension of Z. ldz >= n.
 * @param[out] V      Complex array, dimension (ldv, nshfts/2).
 * @param[in] ldv     Leading dimension of V. ldv >= 3.
 * @param[out] U      Complex array, dimension (ldu, 2*nshfts).
 * @param[in] ldu     Leading dimension of U. ldu >= 2*nshfts.
 * @param[in] nv      Number of rows in WV available for workspace. nv >= 1.
 * @param[out] WV     Complex array, dimension (ldwv, 2*nshfts).
 * @param[in] ldwv    Leading dimension of WV. ldwv >= nv.
 * @param[in] nh      Number of columns in WH available for workspace. nh >= 1.
 * @param[out] WH     Complex array, dimension (ldwh, nh).
 * @param[in] ldwh    Leading dimension of WH. ldwh >= 2*nshfts.
 */
void claqr5(const INT wantt, const INT wantz, const INT kacc22,
            const INT n, const INT ktop, const INT kbot,
            const INT nshfts, c64* S,
            c64* H, const INT ldh,
            const INT iloz, const INT ihiz,
            c64* Z, const INT ldz,
            c64* V, const INT ldv,
            c64* U, const INT ldu,
            const INT nv, c64* WV, const INT ldwv,
            const INT nh, c64* WH, const INT ldwh)
{
    const c64 czero = 0.0f;
    const c64 cone = 1.0f;
    const f32 rzero = 0.0f;

    c64 alpha, beta, refsum, t1, t2, t3;
    f32 h11, h12, h21, h22, safmin, scl, smlnum, tst1, tst2, ulp;
    INT i2, i4, incol, j, jbot, jcol, jlen, jrow, jtop;
    INT k, k1, kdu, kms, krcol, m, m22, mbot, mtop, nbmps, ndcol, ns, nu;
    INT accum, bmp22;

    c64 vt[3];

    if (nshfts < 2)
        return;

    if (ktop >= kbot)
        return;

    ns = nshfts - (nshfts % 2);

    safmin = slamch("Safe minimum");
    ulp = slamch("Precision");
    smlnum = safmin * ((f32)n / ulp);

    accum = (kacc22 == 1) || (kacc22 == 2);

    if (ktop + 2 <= kbot)
        H[(ktop + 2) + ktop * ldh] = czero;

    nbmps = ns / 2;

    kdu = 4 * nbmps;

    for (incol = ktop - 2 * nbmps + 1; incol <= kbot - 2; incol += 2 * nbmps) {

        if (accum) {
            jtop = ktop > incol ? ktop : incol;
        } else if (wantt) {
            jtop = 0;
        } else {
            jtop = ktop;
        }

        ndcol = incol + kdu;
        if (accum)
            claset("ALL", kdu, kdu, czero, cone, U, ldu);

        for (krcol = incol; krcol <= (incol + 2 * nbmps - 1 < kbot - 2 ?
                                      incol + 2 * nbmps - 1 : kbot - 2); krcol++) {

            mtop = 1 > (ktop - krcol) / 2 + 1 ? 1 : (ktop - krcol) / 2 + 1;
            mbot = nbmps < (kbot - krcol - 1) / 2 ? nbmps : (kbot - krcol - 1) / 2;
            m22 = mbot + 1;
            bmp22 = (mbot < nbmps) && (krcol + 2 * (m22 - 1) == kbot - 2);

            if (bmp22) {
                k = krcol + 2 * (m22 - 1);
                if (k == ktop - 1) {
                    claqr1(2, &H[(k + 1) + (k + 1) * ldh], ldh,
                           S[2 * m22 - 2], S[2 * m22 - 1],
                           &V[(m22 - 1) * ldv]);
                    beta = V[(m22 - 1) * ldv];
                    clarfg(2, &beta, &V[1 + (m22 - 1) * ldv], 1,
                           &V[(m22 - 1) * ldv]);
                } else {
                    beta = H[(k + 1) + k * ldh];
                    V[1 + (m22 - 1) * ldv] = H[(k + 2) + k * ldh];
                    clarfg(2, &beta, &V[1 + (m22 - 1) * ldv], 1,
                           &V[(m22 - 1) * ldv]);
                    H[(k + 1) + k * ldh] = beta;
                    H[(k + 2) + k * ldh] = czero;
                }

                /* Perform update from right within computational window */
                t1 = V[(m22 - 1) * ldv];
                t2 = t1 * conjf(V[1 + (m22 - 1) * ldv]);
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
                t1 = conjf(V[(m22 - 1) * ldv]);
                t2 = t1 * V[1 + (m22 - 1) * ldv];
                for (j = k + 1; j <= jbot; j++) {
                    refsum = H[(k + 1) + j * ldh] +
                             conjf(V[1 + (m22 - 1) * ldv]) * H[(k + 2) + j * ldh];
                    H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - refsum * t1;
                    H[(k + 2) + j * ldh] = H[(k + 2) + j * ldh] - refsum * t2;
                }

                /* Convergence test */
                if (k >= ktop) {
                    if (H[(k + 1) + k * ldh] != czero) {
                        tst1 = cabs1f(H[k + k * ldh]) + cabs1f(H[(k + 1) + (k + 1) * ldh]);
                        if (tst1 == rzero) {
                            if (k >= ktop + 1)
                                tst1 = tst1 + cabs1f(H[k + (k - 1) * ldh]);
                            if (k >= ktop + 2)
                                tst1 = tst1 + cabs1f(H[k + (k - 2) * ldh]);
                            if (k >= ktop + 3)
                                tst1 = tst1 + cabs1f(H[k + (k - 3) * ldh]);
                            if (k <= kbot - 2)
                                tst1 = tst1 + cabs1f(H[(k + 2) + (k + 1) * ldh]);
                            if (k <= kbot - 3)
                                tst1 = tst1 + cabs1f(H[(k + 3) + (k + 1) * ldh]);
                            if (k <= kbot - 4)
                                tst1 = tst1 + cabs1f(H[(k + 4) + (k + 1) * ldh]);
                        }
                        if (cabs1f(H[(k + 1) + k * ldh]) <=
                            (smlnum > ulp * tst1 ? smlnum : ulp * tst1)) {
                            h12 = cabs1f(H[(k + 1) + k * ldh]) > cabs1f(H[k + (k + 1) * ldh]) ?
                                  cabs1f(H[(k + 1) + k * ldh]) : cabs1f(H[k + (k + 1) * ldh]);
                            h21 = cabs1f(H[(k + 1) + k * ldh]) < cabs1f(H[k + (k + 1) * ldh]) ?
                                  cabs1f(H[(k + 1) + k * ldh]) : cabs1f(H[k + (k + 1) * ldh]);
                            h11 = cabs1f(H[(k + 1) + (k + 1) * ldh]) >
                                  cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                                  cabs1f(H[(k + 1) + (k + 1) * ldh]) :
                                  cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                            h22 = cabs1f(H[(k + 1) + (k + 1) * ldh]) <
                                  cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                                  cabs1f(H[(k + 1) + (k + 1) * ldh]) :
                                  cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                            scl = h11 + h12;
                            tst2 = h22 * (h11 / scl);
                            if (tst2 == rzero || h21 * (h12 / scl) <=
                                (smlnum > ulp * tst2 ? smlnum : ulp * tst2)) {
                                H[(k + 1) + k * ldh] = czero;
                            }
                        }
                    }
                }

                /* Accumulate unitary transformations */
                if (accum) {
                    kms = k - incol;
                    t1 = V[(m22 - 1) * ldv];
                    t2 = t1 * conjf(V[1 + (m22 - 1) * ldv]);
                    for (j = (1 > ktop - incol ? 1 : ktop - incol) - 1; j < kdu; j++) {
                        refsum = U[j + kms * ldu] +
                                 V[1 + (m22 - 1) * ldv] * U[j + (kms + 1) * ldu];
                        U[j + kms * ldu] = U[j + kms * ldu] - refsum * t1;
                        U[j + (kms + 1) * ldu] = U[j + (kms + 1) * ldu] - refsum * t2;
                    }
                } else if (wantz) {
                    t1 = V[(m22 - 1) * ldv];
                    t2 = t1 * conjf(V[1 + (m22 - 1) * ldv]);
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
                    claqr1(3, &H[ktop + ktop * ldh], ldh,
                           S[2 * m - 2], S[2 * m - 1],
                           &V[(m - 1) * ldv]);
                    alpha = V[(m - 1) * ldv];
                    clarfg(3, &alpha, &V[1 + (m - 1) * ldv], 1,
                           &V[(m - 1) * ldv]);
                } else {
                    /* Perform delayed transformation of row below Mth bulge.
                     * Exploit fact that first two elements of row are zero. */
                    t1 = V[(m - 1) * ldv];
                    t2 = t1 * conjf(V[1 + (m - 1) * ldv]);
                    t3 = t1 * conjf(V[2 + (m - 1) * ldv]);
                    refsum = V[2 + (m - 1) * ldv] * H[(k + 3) + (k + 2) * ldh];
                    H[(k + 3) + k * ldh] = -refsum * t1;
                    H[(k + 3) + (k + 1) * ldh] = -refsum * t2;
                    H[(k + 3) + (k + 2) * ldh] = H[(k + 3) + (k + 2) * ldh] - refsum * t3;

                    /* Calculate reflection to move Mth bulge one step */
                    beta = H[(k + 1) + k * ldh];
                    V[1 + (m - 1) * ldv] = H[(k + 2) + k * ldh];
                    V[2 + (m - 1) * ldv] = H[(k + 3) + k * ldh];
                    clarfg(3, &beta, &V[1 + (m - 1) * ldv], 1,
                           &V[(m - 1) * ldv]);

                    if (H[(k + 3) + k * ldh] != czero ||
                        H[(k + 3) + (k + 1) * ldh] != czero ||
                        H[(k + 3) + (k + 2) * ldh] == czero) {
                        H[(k + 1) + k * ldh] = beta;
                        H[(k + 2) + k * ldh] = czero;
                        H[(k + 3) + k * ldh] = czero;
                    } else {
                        claqr1(3, &H[(k + 1) + (k + 1) * ldh], ldh,
                               S[2 * m - 2], S[2 * m - 1], vt);
                        alpha = vt[0];
                        clarfg(3, &alpha, &vt[1], 1, &vt[0]);
                        t1 = conjf(vt[0]);
                        t2 = t1 * vt[1];
                        t3 = t1 * vt[2];
                        refsum = H[(k + 1) + k * ldh] + conjf(vt[1]) * H[(k + 2) + k * ldh];

                        if (cabs1f(H[(k + 2) + k * ldh] - refsum * t2) +
                            cabs1f(refsum * t3) > ulp * (cabs1f(H[k + k * ldh]) +
                            cabs1f(H[(k + 1) + (k + 1) * ldh]) +
                            cabs1f(H[(k + 2) + (k + 2) * ldh]))) {
                            H[(k + 1) + k * ldh] = beta;
                            H[(k + 2) + k * ldh] = czero;
                            H[(k + 3) + k * ldh] = czero;
                        } else {
                            H[(k + 1) + k * ldh] = H[(k + 1) + k * ldh] - refsum * conjf(vt[0]);
                            H[(k + 2) + k * ldh] = czero;
                            H[(k + 3) + k * ldh] = czero;
                            V[(m - 1) * ldv] = vt[0];
                            V[1 + (m - 1) * ldv] = vt[1];
                            V[2 + (m - 1) * ldv] = vt[2];
                        }
                    }
                }

                /* Apply reflection from the right and first column of
                 * update from the left */
                t1 = V[(m - 1) * ldv];
                t2 = t1 * conjf(V[1 + (m - 1) * ldv]);
                t3 = t1 * conjf(V[2 + (m - 1) * ldv]);
                for (j = jtop; j <= (kbot < k + 3 ? kbot : k + 3); j++) {
                    refsum = H[j + (k + 1) * ldh] +
                             V[1 + (m - 1) * ldv] * H[j + (k + 2) * ldh] +
                             V[2 + (m - 1) * ldv] * H[j + (k + 3) * ldh];
                    H[j + (k + 1) * ldh] = H[j + (k + 1) * ldh] - refsum * t1;
                    H[j + (k + 2) * ldh] = H[j + (k + 2) * ldh] - refsum * t2;
                    H[j + (k + 3) * ldh] = H[j + (k + 3) * ldh] - refsum * t3;
                }

                /* Perform update from left for subsequent column */
                t1 = conjf(V[(m - 1) * ldv]);
                t2 = t1 * V[1 + (m - 1) * ldv];
                t3 = t1 * V[2 + (m - 1) * ldv];
                refsum = H[(k + 1) + (k + 1) * ldh] +
                         conjf(V[1 + (m - 1) * ldv]) * H[(k + 2) + (k + 1) * ldh] +
                         conjf(V[2 + (m - 1) * ldv]) * H[(k + 3) + (k + 1) * ldh];
                H[(k + 1) + (k + 1) * ldh] = H[(k + 1) + (k + 1) * ldh] - refsum * t1;
                H[(k + 2) + (k + 1) * ldh] = H[(k + 2) + (k + 1) * ldh] - refsum * t2;
                H[(k + 3) + (k + 1) * ldh] = H[(k + 3) + (k + 1) * ldh] - refsum * t3;

                /* Convergence test */
                if (k < ktop)
                    continue;

                if (H[(k + 1) + k * ldh] != czero) {
                    tst1 = cabs1f(H[k + k * ldh]) + cabs1f(H[(k + 1) + (k + 1) * ldh]);
                    if (tst1 == rzero) {
                        if (k >= ktop + 1)
                            tst1 = tst1 + cabs1f(H[k + (k - 1) * ldh]);
                        if (k >= ktop + 2)
                            tst1 = tst1 + cabs1f(H[k + (k - 2) * ldh]);
                        if (k >= ktop + 3)
                            tst1 = tst1 + cabs1f(H[k + (k - 3) * ldh]);
                        if (k <= kbot - 2)
                            tst1 = tst1 + cabs1f(H[(k + 2) + (k + 1) * ldh]);
                        if (k <= kbot - 3)
                            tst1 = tst1 + cabs1f(H[(k + 3) + (k + 1) * ldh]);
                        if (k <= kbot - 4)
                            tst1 = tst1 + cabs1f(H[(k + 4) + (k + 1) * ldh]);
                    }
                    if (cabs1f(H[(k + 1) + k * ldh]) <=
                        (smlnum > ulp * tst1 ? smlnum : ulp * tst1)) {
                        h12 = cabs1f(H[(k + 1) + k * ldh]) > cabs1f(H[k + (k + 1) * ldh]) ?
                              cabs1f(H[(k + 1) + k * ldh]) : cabs1f(H[k + (k + 1) * ldh]);
                        h21 = cabs1f(H[(k + 1) + k * ldh]) < cabs1f(H[k + (k + 1) * ldh]) ?
                              cabs1f(H[(k + 1) + k * ldh]) : cabs1f(H[k + (k + 1) * ldh]);
                        h11 = cabs1f(H[(k + 1) + (k + 1) * ldh]) >
                              cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                              cabs1f(H[(k + 1) + (k + 1) * ldh]) :
                              cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                        h22 = cabs1f(H[(k + 1) + (k + 1) * ldh]) <
                              cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]) ?
                              cabs1f(H[(k + 1) + (k + 1) * ldh]) :
                              cabs1f(H[k + k * ldh] - H[(k + 1) + (k + 1) * ldh]);
                        scl = h11 + h12;
                        tst2 = h22 * (h11 / scl);
                        if (tst2 == rzero || h21 * (h12 / scl) <=
                            (smlnum > ulp * tst2 ? smlnum : ulp * tst2)) {
                            H[(k + 1) + k * ldh] = czero;
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
                t1 = conjf(V[(m - 1) * ldv]);
                t2 = t1 * V[1 + (m - 1) * ldv];
                t3 = t1 * V[2 + (m - 1) * ldv];
                for (j = (ktop > krcol + 2 * m ? ktop : krcol + 2 * m); j <= jbot; j++) {
                    refsum = H[(k + 1) + j * ldh] +
                             conjf(V[1 + (m - 1) * ldv]) * H[(k + 2) + j * ldh] +
                             conjf(V[2 + (m - 1) * ldv]) * H[(k + 3) + j * ldh];
                    H[(k + 1) + j * ldh] = H[(k + 1) + j * ldh] - refsum * t1;
                    H[(k + 2) + j * ldh] = H[(k + 2) + j * ldh] - refsum * t2;
                    H[(k + 3) + j * ldh] = H[(k + 3) + j * ldh] - refsum * t3;
                }
            }

            /* Accumulate unitary transformations */
            if (accum) {
                for (m = mbot; m >= mtop; m--) {
                    k = krcol + 2 * (m - 1);
                    kms = k - incol;
                    i2 = 1 > ktop - incol ? 1 : ktop - incol;
                    i2 = i2 > kms - (krcol - incol) + 1 ? i2 : kms - (krcol - incol) + 1;
                    i4 = kdu < krcol + 2 * (mbot - 1) - incol + 5 ?
                         kdu : krcol + 2 * (mbot - 1) - incol + 5;
                    t1 = V[(m - 1) * ldv];
                    t2 = t1 * conjf(V[1 + (m - 1) * ldv]);
                    t3 = t1 * conjf(V[2 + (m - 1) * ldv]);
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
                for (m = mbot; m >= mtop; m--) {
                    k = krcol + 2 * (m - 1);
                    t1 = V[(m - 1) * ldv];
                    t2 = t1 * conjf(V[1 + (m - 1) * ldv]);
                    t3 = t1 * conjf(V[2 + (m - 1) * ldv]);
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
                cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                           nu, jlen, nu, &cone, &U[(k1 - 1) + (k1 - 1) * ldu], ldu,
                           &H[(incol + k1) + jcol * ldh], ldh, &czero, WH, ldwh);
                clacpy("ALL", nu, jlen, WH, ldwh, &H[(incol + k1) + jcol * ldh], ldh);
            }

            /* Vertical multiply */
            for (jrow = jtop; jrow <= (ktop > incol ? ktop : incol) - 1; jrow += nv) {
                jlen = nv < (ktop > incol ? ktop : incol) - jrow ?
                       nv : (ktop > incol ? ktop : incol) - jrow;
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                           jlen, nu, nu, &cone, &H[jrow + (incol + k1) * ldh], ldh,
                           &U[(k1 - 1) + (k1 - 1) * ldu], ldu, &czero, WV, ldwv);
                clacpy("ALL", jlen, nu, WV, ldwv, &H[jrow + (incol + k1) * ldh], ldh);
            }

            /* Z multiply (also vertical) */
            if (wantz) {
                for (jrow = iloz; jrow <= ihiz; jrow += nv) {
                    jlen = nv < ihiz - jrow + 1 ? nv : ihiz - jrow + 1;
                    cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                               jlen, nu, nu, &cone, &Z[jrow + (incol + k1) * ldz], ldz,
                               &U[(k1 - 1) + (k1 - 1) * ldu], ldu, &czero, WV, ldwv);
                    clacpy("ALL", jlen, nu, WV, ldwv, &Z[jrow + (incol + k1) * ldz], ldz);
                }
            }
        }
    }
}
