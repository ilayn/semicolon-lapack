/**
 * @file claqr0.c
 * @brief CLAQR0 computes eigenvalues of a Hessenberg matrix using small-bulge
 *        multi-shift QR with aggressive early deflation.
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <math.h>

/** @cond */
static INT iparmq_nmin(void)
{
    return 75;
}

static INT iparmq_nshfts(INT nh)
{
    INT ns = 2;
    if (nh >= 30) ns = 4;
    if (nh >= 60) ns = 10;
    if (nh >= 150) {
        f32 lognh = logf((f32)nh) / logf(2.0f);
        INT div = (INT)(lognh + 0.5f);
        if (div < 1) div = 1;
        ns = nh / div;
        if (ns < 10) ns = 10;
    }
    if (nh >= 590) ns = 64;
    if (nh >= 3000) ns = 128;
    if (nh >= 6000) ns = 256;
    ns = ns - (ns % 2);
    if (ns < 2) ns = 2;
    return ns;
}

static INT iparmq_nwr(INT nh)
{
    INT ns = iparmq_nshfts(nh);
    if (nh <= 500) {
        return ns;
    } else {
        return 3 * ns / 2;
    }
}

static INT iparmq_nibble(void)
{
    return 14;
}

static INT iparmq_kacc22(INT ns)
{
    const INT kacmin = 14;
    const INT k22min = 14;
    INT kacc = 0;
    if (ns >= kacmin) kacc = 1;
    if (ns >= k22min) kacc = 2;
    return kacc;
}
/** @endcond */

/**
 * CLAQR0 computes the eigenvalues of a Hessenberg matrix H
 * and, optionally, the matrices T and Z from the Schur decomposition
 * H = Z T Z^H, where T is an upper triangular matrix (the
 * Schur form), and Z is the unitary matrix of Schur vectors.
 *
 * Optionally Z may be postmultiplied into an input unitary
 * matrix Q so that this routine can give the Schur factorization
 * of a matrix A which has been reduced to the Hessenberg form H
 * by the unitary matrix Q:  A = Q*H*Q^H = (QZ)*T*(QZ)^H.
 *
 * CLAQR0 is identical to CLAQR4 except that it calls CLAQR3
 * instead of CLAQR2.
 *
 * @param[in] wantt   If nonzero, the full Schur form T is required.
 * @param[in] wantz   If nonzero, the matrix of Schur vectors Z is required.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ilo     First index of isolated block (0-based).
 * @param[in] ihi     Last index of isolated block (0-based).
 * @param[in,out] H   Complex array, dimension (ldh, n).
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[out] W      Complex array, dimension (n). The computed eigenvalues.
 * @param[in] iloz    First row of Z to update (0-based).
 * @param[in] ihiz    Last row of Z to update (0-based).
 * @param[in,out] Z   Complex array, dimension (ldz, n).
 * @param[in] ldz     Leading dimension of Z.
 * @param[out] work   Complex array, dimension (lwork).
 * @param[in] lwork   Dimension of work array.
 *                    If lwork = -1, workspace query is assumed.
 * @param[out] info   = 0: successful exit.
 *                    > 0: if info = i (1-based), CLAQR0 failed to compute
 *                         all eigenvalues.
 */
void claqr0(const INT wantt, const INT wantz, const INT n,
            const INT ilo, const INT ihi,
            c64* H, const INT ldh,
            c64* W,
            const INT iloz, const INT ihiz,
            c64* Z, const INT ldz,
            c64* work, const INT lwork, INT* info)
{
    const INT ntiny = 15;
    const INT kexnw = 5;
    const INT kexsh = 6;
    const f32 wilk1 = 0.75f;
    const c64 czero = 0.0f;
    const c64 cone = 1.0f;
    const f32 two = 2.0f;

    c64 aa, bb, cc, dd, det, rtdisc, swap, tr2;
    f32 s;
    INT i, inf, it, itmax, k, kacc22, kbot, kdu, ks;
    INT kt, ktop, ku, kv, kwh, kwtop, kwv, ld, ls;
    INT lwkopt, ndec = -1, ndfl, nh, nho, nibble, nmin, ns;
    INT nsmax, nsr, nve, nw, nwmax, nwr, nwupbd;
    INT sorted;

    c64 zdum[1];

    *info = 0;

    if (n == 0) {
        work[0] = cone;
        return;
    }

    if (n <= ntiny) {
        lwkopt = 1;
        if (lwork != -1) {
            clahqr(wantt, wantz, n, ilo, ihi, H, ldh, W,
                   iloz, ihiz, Z, ldz, info);
        }
    } else {
        *info = 0;

        nh = ihi - ilo + 1;

        nwr = iparmq_nwr(nh);
        if (nwr < 2) nwr = 2;
        if (nwr > (n - 1) / 3) nwr = (n - 1) / 3;
        if (nwr > ihi - ilo + 1) nwr = ihi - ilo + 1;

        nsr = iparmq_nshfts(nh);
        if (nsr > (n - 3) / 6) nsr = (n - 3) / 6;
        if (nsr > ihi - ilo) nsr = ihi - ilo;
        nsr = nsr - (nsr % 2);
        if (nsr < 2) nsr = 2;

        /* Workspace query call to CLAQR3 */
        claqr3(wantt, wantz, n, ilo, ihi, nwr + 1, H, ldh, iloz,
               ihiz, Z, ldz, &ls, &ld, W, H, ldh, n, H, ldh,
               n, H, ldh, work, -1);

        lwkopt = 3 * nsr / 2;
        if ((INT)crealf(work[0]) > lwkopt) lwkopt = (INT)crealf(work[0]);

        if (lwork == -1) {
            work[0] = (f32)lwkopt;
            return;
        }

        nmin = iparmq_nmin();
        if (nmin < ntiny) nmin = ntiny;

        nibble = iparmq_nibble();
        if (nibble < 0) nibble = 0;

        kacc22 = iparmq_kacc22(nsr);
        if (kacc22 < 0) kacc22 = 0;
        if (kacc22 > 2) kacc22 = 2;

        nwmax = (n - 1) / 3;
        if (nwmax > lwork / 2) nwmax = lwork / 2;
        nw = nwmax;

        nsmax = (n - 3) / 6;
        if (nsmax > 2 * lwork / 3) nsmax = 2 * lwork / 3;
        nsmax = nsmax - (nsmax % 2);

        ndfl = 1;

        itmax = 30 > 2 * kexsh ? 30 : 2 * kexsh;
        itmax *= (10 > ihi - ilo + 1 ? 10 : ihi - ilo + 1);

        kbot = ihi;

        for (it = 1; it <= itmax; it++) {
            if (kbot < ilo)
                goto L90;

            for (k = kbot; k >= ilo + 1; k--) {
                if (H[k + (k - 1) * ldh] == czero)
                    break;
            }
            if (k < ilo + 1) k = ilo;
            ktop = k;

            nh = kbot - ktop + 1;
            nwupbd = nh < nwmax ? nh : nwmax;
            if (ndfl < kexnw) {
                nw = nwupbd < nwr ? nwupbd : nwr;
            } else {
                nw = nwupbd < 2 * nw ? nwupbd : 2 * nw;
            }
            if (nw < nwmax) {
                if (nw >= nh - 1) {
                    nw = nh;
                } else {
                    kwtop = kbot - nw + 1;
                    if (cabs1f(H[kwtop + (kwtop - 1) * ldh]) >
                        cabs1f(H[(kwtop - 1) + (kwtop - 2) * ldh]))
                        nw = nw + 1;
                }
            }
            if (ndfl < kexnw) {
                ndec = -1;
            } else if (ndec >= 0 || nw >= nwupbd) {
                ndec = ndec + 1;
                if (nw - ndec < 2)
                    ndec = 0;
                nw = nw - ndec;
            }

            kv = n - nw;
            kt = nw;
            nho = (n - nw - 1) - kt;
            kwv = nw + 1;
            nve = (n - nw) - kwv;

            /* Aggressive early deflation - uses CLAQR3 (not CLAQR2) */
            claqr3(wantt, wantz, n, ktop, kbot, nw, H, ldh, iloz,
                   ihiz, Z, ldz, &ls, &ld, W, &H[kv], ldh,
                   nho, &H[kv + kt * ldh], ldh, nve, &H[kwv], ldh,
                   work, lwork);

            kbot = kbot - ld;

            ks = kbot - ls + 1;

            if ((ld == 0) || ((100 * ld <= nw * nibble) &&
                              (kbot - ktop + 1 > (nmin < nwmax ? nmin : nwmax)))) {

                ns = nsmax < nsr ? nsmax : nsr;
                ns = ns < kbot - ktop ? ns : kbot - ktop;
                if (ns < 2) ns = 2;
                ns = ns - (ns % 2);

                if ((ndfl % kexsh) == 0) {
                    ks = kbot - ns + 1;
                    for (i = kbot; i >= ks + 1; i -= 2) {
                        W[i] = H[i + i * ldh] + wilk1 * cabs1f(H[i + (i - 1) * ldh]);
                        W[i - 1] = W[i];
                    }
                } else {
                    if (kbot - ks + 1 <= ns / 2) {
                        ks = kbot - ns + 1;
                        kt = n - ns;
                        clacpy("A", ns, ns, &H[ks + ks * ldh], ldh,
                               &H[kt], ldh);
                        if (ns > nmin) {
                            claqr4(0, 0, ns, 0, ns - 1, &H[kt], ldh, &W[ks],
                                   0, 0, zdum, 1, work, lwork, &inf);
                        } else {
                            clahqr(0, 0, ns, 0, ns - 1, &H[kt], ldh, &W[ks],
                                   0, 0, zdum, 1, &inf);
                        }
                        ks = ks + inf;

                        if (ks >= kbot) {
                            s = cabs1f(H[(kbot - 1) + (kbot - 1) * ldh]) +
                                cabs1f(H[kbot + (kbot - 1) * ldh]) +
                                cabs1f(H[(kbot - 1) + kbot * ldh]) +
                                cabs1f(H[kbot + kbot * ldh]);
                            aa = H[(kbot - 1) + (kbot - 1) * ldh] / s;
                            cc = H[kbot + (kbot - 1) * ldh] / s;
                            bb = H[(kbot - 1) + kbot * ldh] / s;
                            dd = H[kbot + kbot * ldh] / s;
                            tr2 = (aa + dd) / two;
                            det = (aa - tr2) * (dd - tr2) - bb * cc;
                            rtdisc = csqrtf(-det);
                            W[kbot - 1] = (tr2 + rtdisc) * s;
                            W[kbot] = (tr2 - rtdisc) * s;

                            ks = kbot - 1;
                        }
                    }

                    if (kbot - ks + 1 > ns) {
                        sorted = 0;
                        for (k = kbot; k >= ks + 1; k--) {
                            if (sorted)
                                break;
                            sorted = 1;
                            for (i = ks; i <= k - 1; i++) {
                                if (cabs1f(W[i]) < cabs1f(W[i + 1])) {
                                    sorted = 0;
                                    swap = W[i];
                                    W[i] = W[i + 1];
                                    W[i + 1] = swap;
                                }
                            }
                        }
                    }
                }

                if (kbot - ks + 1 == 2) {
                    if (cabs1f(W[kbot] - H[kbot + kbot * ldh]) <
                        cabs1f(W[kbot - 1] - H[kbot + kbot * ldh])) {
                        W[kbot - 1] = W[kbot];
                    } else {
                        W[kbot] = W[kbot - 1];
                    }
                }

                ns = ns < kbot - ks + 1 ? ns : kbot - ks + 1;
                ns = ns - (ns % 2);
                ks = kbot - ns + 1;

                kdu = 2 * ns;
                ku = n - kdu;
                kwh = kdu;
                nho = (n - kdu + 1 - 4) - kdu;
                kwv = kdu + 3;
                nve = n - kdu - kwv;

                claqr5(wantt, wantz, kacc22, n, ktop, kbot, ns,
                       &W[ks], H, ldh, iloz, ihiz, Z, ldz,
                       work, 3, &H[ku], ldh, nve, &H[kwv], ldh,
                       nho, &H[ku + kwh * ldh], ldh);
            }

            if (ld > 0) {
                ndfl = 1;
            } else {
                ndfl = ndfl + 1;
            }
        }

        *info = kbot + 1;

L90:;
    }

    work[0] = (f32)lwkopt;
}
