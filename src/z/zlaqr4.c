/**
 * @file zlaqr4.c
 * @brief ZLAQR4 computes eigenvalues of a Hessenberg matrix using small-bulge
 *        multi-shift QR with aggressive early deflation.
 */

#include "semicolon_lapack_complex_double.h"
#include <complex.h>
#include <math.h>

static int iparmq_nmin(void)
{
    return 75;
}

static int iparmq_nshfts(int nh)
{
    int ns = 2;
    if (nh >= 30) ns = 4;
    if (nh >= 60) ns = 10;
    if (nh >= 150) {
        double lognh = log((double)nh) / log(2.0);
        int div = (int)(lognh + 0.5);
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

static int iparmq_nwr(int nh)
{
    int ns = iparmq_nshfts(nh);
    if (nh <= 500) {
        return ns;
    } else {
        return 3 * ns / 2;
    }
}

static int iparmq_nibble(void)
{
    return 14;
}

static int iparmq_kacc22(int ns)
{
    const int kacmin = 14;
    const int k22min = 14;
    int kacc = 0;
    if (ns >= kacmin) kacc = 1;
    if (ns >= k22min) kacc = 2;
    return kacc;
}

/**
 * ZLAQR4 implements one level of recursion for ZLAQR0.
 * It is a complete implementation of the small bulge multi-shift
 * QR algorithm. It is identical to ZLAQR0 except that it calls ZLAQR2
 * instead of ZLAQR3.
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
 *                    > 0: if info = i (1-based), ZLAQR4 failed to compute
 *                         all eigenvalues.
 */
void zlaqr4(const int wantt, const int wantz, const int n,
            const int ilo, const int ihi,
            double complex* H, const int ldh,
            double complex* W,
            const int iloz, const int ihiz,
            double complex* Z, const int ldz,
            double complex* work, const int lwork, int* info)
{
    const int ntiny = 15;
    const int kexnw = 5;
    const int kexsh = 6;
    const double wilk1 = 0.75;
    const double complex czero = 0.0;
    const double complex cone = 1.0;
    const double two = 2.0;

    double complex aa, bb, cc, dd, det, rtdisc, swap, tr2;
    double s;
    int i, inf, it, itmax, k, kacc22, kbot, kdu, ks;
    int kt, ktop, ku, kv, kwh, kwtop, kwv, ld, ls;
    int lwkopt, ndec = -1, ndfl, nh, nho, nibble, nmin, ns;
    int nsmax, nsr, nve, nw, nwmax, nwr, nwupbd;
    int sorted;

    double complex zdum[1];

    *info = 0;

    if (n == 0) {
        work[0] = cone;
        return;
    }

    if (n <= ntiny) {
        lwkopt = 1;
        if (lwork != -1) {
            zlahqr(wantt, wantz, n, ilo, ihi, H, ldh, W,
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

        zlaqr2(wantt, wantz, n, ilo, ihi, nwr + 1, H, ldh, iloz,
               ihiz, Z, ldz, &ls, &ld, W, H, ldh, n, H, ldh,
               n, H, ldh, work, -1);

        lwkopt = 3 * nsr / 2;
        if ((int)creal(work[0]) > lwkopt) lwkopt = (int)creal(work[0]);

        if (lwork == -1) {
            work[0] = (double)lwkopt;
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
                    if (cabs1(H[kwtop + (kwtop - 1) * ldh]) >
                        cabs1(H[(kwtop - 1) + (kwtop - 2) * ldh]))
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

            zlaqr2(wantt, wantz, n, ktop, kbot, nw, H, ldh, iloz,
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
                        W[i] = H[i + i * ldh] + wilk1 * cabs1(H[i + (i - 1) * ldh]);
                        W[i - 1] = W[i];
                    }
                } else {
                    if (kbot - ks + 1 <= ns / 2) {
                        ks = kbot - ns + 1;
                        kt = n - ns;
                        zlacpy("A", ns, ns, &H[ks + ks * ldh], ldh,
                               &H[kt], ldh);
                        zlahqr(0, 0, ns, 0, ns - 1, &H[kt], ldh, &W[ks],
                               0, 0, zdum, 1, &inf);
                        ks = ks + inf;

                        if (ks >= kbot) {
                            s = cabs1(H[(kbot - 1) + (kbot - 1) * ldh]) +
                                cabs1(H[kbot + (kbot - 1) * ldh]) +
                                cabs1(H[(kbot - 1) + kbot * ldh]) +
                                cabs1(H[kbot + kbot * ldh]);
                            aa = H[(kbot - 1) + (kbot - 1) * ldh] / s;
                            cc = H[kbot + (kbot - 1) * ldh] / s;
                            bb = H[(kbot - 1) + kbot * ldh] / s;
                            dd = H[kbot + kbot * ldh] / s;
                            tr2 = (aa + dd) / two;
                            det = (aa - tr2) * (dd - tr2) - bb * cc;
                            rtdisc = csqrt(-det);
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
                                if (cabs1(W[i]) < cabs1(W[i + 1])) {
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
                    if (cabs1(W[kbot] - H[kbot + kbot * ldh]) <
                        cabs1(W[kbot - 1] - H[kbot + kbot * ldh])) {
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

                zlaqr5(wantt, wantz, kacc22, n, ktop, kbot, ns,
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

    work[0] = (double)lwkopt;
}
