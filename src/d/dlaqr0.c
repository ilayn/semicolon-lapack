/**
 * @file dlaqr0.c
 * @brief DLAQR0 computes eigenvalues of a Hessenberg matrix using small-bulge
 *        multi-shift QR with aggressive early deflation.
 */

#include "semicolon_lapack_double.h"
#include <math.h>

/**
 * Compute IPARMQ-style tuning parameters for DLAQR0.
 * These are hardcoded from LAPACK's iparmq.f defaults.
 */

/* ISPEC=12: NMIN - crossover to DLAHQR (must be at least 11) */
static int iparmq_nmin(void)
{
    return 75;
}

/* ISPEC=15: number of simultaneous shifts */
static int iparmq_nshfts(int nh)
{
    int ns = 2;
    if (nh >= 30) ns = 4;
    if (nh >= 60) ns = 10;
    if (nh >= 150) {
        /* ns = max(10, nh / nint(log(nh)/log(2))) */
        f64 lognh = log((f64)nh) / log(2.0);
        int div = (int)(lognh + 0.5);  /* nint */
        if (div < 1) div = 1;
        ns = nh / div;
        if (ns < 10) ns = 10;
    }
    if (nh >= 590) ns = 64;
    if (nh >= 3000) ns = 128;
    if (nh >= 6000) ns = 256;
    /* Make even and at least 2 */
    ns = ns - (ns % 2);
    if (ns < 2) ns = 2;
    return ns;
}

/* ISPEC=13: deflation window size */
static int iparmq_nwr(int nh)
{
    int ns = iparmq_nshfts(nh);
    if (nh <= 500) {
        return ns;
    } else {
        return 3 * ns / 2;
    }
}

/* ISPEC=14: nibble crossover point */
static int iparmq_nibble(void)
{
    return 14;
}

/* ISPEC=16: accumulation mode (0, 1, or 2) */
static int iparmq_kacc22(int ns)
{
    /* For DLAQR0: use NS-based threshold */
    const int kacmin = 14;
    const int k22min = 14;
    int kacc = 0;
    if (ns >= kacmin) kacc = 1;
    if (ns >= k22min) kacc = 2;
    return kacc;
}

/**
 * DLAQR0 computes the eigenvalues of a Hessenberg matrix H
 * and, optionally, the matrices T and Z from the Schur decomposition
 * H = Z T Z^T, where T is an upper quasi-triangular matrix (the
 * Schur form), and Z is the orthogonal matrix of Schur vectors.
 *
 * Optionally Z may be postmultiplied into an input orthogonal
 * matrix Q so that this routine can give the Schur factorization
 * of a matrix A which has been reduced to the Hessenberg form H
 * by the orthogonal matrix Q:  A = Q*H*Q^T = (QZ)*T*(QZ)^T.
 *
 * DLAQR0 is identical to DLAQR4 except that it calls DLAQR3
 * instead of DLAQR2.
 *
 * @param[in] wantt   If nonzero, the full Schur form T is required.
 *                    If zero, only eigenvalues are required.
 * @param[in] wantz   If nonzero, the matrix of Schur vectors Z is required.
 *                    If zero, Schur vectors are not required.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ilo     First index of isolated block (0-based).
 * @param[in] ihi     Last index of isolated block (0-based).
 *                    It is assumed that H is already upper triangular in rows
 *                    and columns 0:ilo-1 and ihi+1:n-1.
 * @param[in,out] H   Double precision array, dimension (ldh, n).
 *                    On entry, the upper Hessenberg matrix H.
 *                    On exit, if info = 0 and wantt is nonzero, then H
 *                    contains the upper quasi-triangular matrix T from the
 *                    Schur decomposition (the Schur form).
 * @param[in] ldh     Leading dimension of H. ldh >= max(1, n).
 * @param[out] wr     Double precision array, dimension (n).
 *                    Real parts of the computed eigenvalues.
 * @param[out] wi     Double precision array, dimension (n).
 *                    Imaginary parts of the computed eigenvalues.
 * @param[in] iloz    First row of Z to update (0-based).
 * @param[in] ihiz    Last row of Z to update (0-based).
 * @param[in,out] Z   Double precision array, dimension (ldz, n).
 *                    If wantz is nonzero, Z is updated with the orthogonal
 *                    Schur factor.
 * @param[in] ldz     Leading dimension of Z. ldz >= 1; if wantz, ldz >= ihiz+1.
 * @param[out] work   Double precision array, dimension (lwork).
 * @param[in] lwork   Dimension of work array. lwork >= max(1, n).
 *                    If lwork = -1, workspace query is assumed.
 * @param[out] info
 *                         - = 0: successful exit
 *                         - > 0: if info = i, DLAQR0 failed to compute all of the
 *                           eigenvalues. Elements ilo:info contain those
 *                           eigenvalues which have been successfully computed.
 */
SEMICOLON_API void dlaqr0(const int wantt, const int wantz, const int n,
                          const int ilo, const int ihi,
                          f64* H, const int ldh,
                          f64* wr, f64* wi,
                          const int iloz, const int ihiz,
                          f64* Z, const int ldz,
                          f64* work, const int lwork, int* info)
{
    /* Parameters */
    const int ntiny = 15;   /* Matrices of order NTINY or smaller use DLAHQR */
    const int kexnw = 5;    /* Exceptional deflation window frequency */
    const int kexsh = 6;    /* Exceptional shifts frequency */
    const f64 wilk1 = 0.75;
    const f64 wilk2 = -0.4375;
    const f64 zero = 0.0;
    const f64 one = 1.0;

    /* Local scalars */
    f64 aa, bb, cc, cs, dd, sn, ss, swap;
    int i, inf, it, itmax, k, kacc22, kbot, kdu, ks;
    int kt, ktop, ku, kv, kwh, kwtop, kwv, ld, ls;
    int lwkopt, ndec = -1, ndfl, nh, nho, nibble, nmin, ns;
    int nsmax, nsr, nve, nw, nwmax, nwr, nwupbd;
    int sorted;

    /* Local array for dummy Z in DLAHQR/DLAQR4 call */
    f64 zdum[1];

    *info = 0;

    /* Quick return for N = 0: nothing to do */
    if (n == 0) {
        work[0] = one;
        return;
    }

    if (n <= ntiny) {
        /* Tiny matrices must use DLAHQR */
        lwkopt = 1;
        if (lwork != -1) {
            dlahqr(wantt, wantz, n, ilo, ihi, H, ldh, wr, wi,
                   iloz, ihiz, Z, ldz, info);
        }
    } else {
        /* Use small bulge multi-shift QR with aggressive early
         * deflation on larger-than-tiny matrices. */

        *info = 0;

        /* Get tuning parameters */
        nh = ihi - ilo + 1;

        /* NWR = recommended deflation window size */
        nwr = iparmq_nwr(nh);
        if (nwr < 2) nwr = 2;
        if (nwr > (n - 1) / 3) nwr = (n - 1) / 3;
        if (nwr > ihi - ilo + 1) nwr = ihi - ilo + 1;

        /* NSR = recommended number of simultaneous shifts */
        nsr = iparmq_nshfts(nh);
        if (nsr > (n - 3) / 6) nsr = (n - 3) / 6;
        if (nsr > ihi - ilo) nsr = ihi - ilo;
        nsr = nsr - (nsr % 2);
        if (nsr < 2) nsr = 2;

        /* Estimate optimal workspace */
        /* Workspace query call to DLAQR3 */
        dlaqr3(wantt, wantz, n, ilo, ihi, nwr + 1, H, ldh, iloz,
               ihiz, Z, ldz, &ls, &ld, wr, wi, H, ldh, n, H, ldh,
               n, H, ldh, work, -1);

        /* Optimal workspace = MAX(3*NSR/2, DLAQR3 workspace) */
        lwkopt = 3 * nsr / 2;
        if ((int)work[0] > lwkopt) lwkopt = (int)work[0];

        /* Quick return in case of workspace query */
        if (lwork == -1) {
            work[0] = (f64)lwkopt;
            return;
        }

        /* NMIN = DLAHQR/DLAQR0 crossover point */
        nmin = iparmq_nmin();
        if (nmin < ntiny) nmin = ntiny;

        /* NIBBLE crossover point */
        nibble = iparmq_nibble();
        if (nibble < 0) nibble = 0;

        /* KACC22 = accumulate reflections? Use block 2-by-2 structure? */
        kacc22 = iparmq_kacc22(nsr);
        if (kacc22 < 0) kacc22 = 0;
        if (kacc22 > 2) kacc22 = 2;

        /* NWMAX = largest possible deflation window for which there
         * is sufficient workspace */
        nwmax = (n - 1) / 3;
        if (nwmax > lwork / 2) nwmax = lwork / 2;
        nw = nwmax;

        /* NSMAX = largest number of simultaneous shifts for which
         * there is sufficient workspace */
        nsmax = (n - 3) / 6;
        if (nsmax > 2 * lwork / 3) nsmax = 2 * lwork / 3;
        nsmax = nsmax - (nsmax % 2);

        /* NDFL: an iteration count restarted at deflation */
        ndfl = 1;

        /* ITMAX = iteration limit */
        itmax = 30 > 2 * kexsh ? 30 : 2 * kexsh;
        itmax *= (10 > ihi - ilo + 1 ? 10 : ihi - ilo + 1);

        /* Last row and column in the active block */
        kbot = ihi;

        /* Main Loop */
        for (it = 1; it <= itmax; it++) {
            /* Done when KBOT falls below ILO */
            if (kbot < ilo)
                goto L90;

            /* Locate active block */
            for (k = kbot; k >= ilo + 1; k--) {
                if (H[k + (k - 1) * ldh] == zero)
                    break;
            }
            if (k < ilo + 1) k = ilo;
            ktop = k;

            /* Select deflation window size:
             * Typical Case: If possible and advisable, nibble the entire
             * active block. If not, use size MIN(NWR, NWMAX) or
             * MIN(NWR+1, NWMAX) depending upon which has the smaller
             * corresponding subdiagonal entry (a heuristic).
             *
             * Exceptional Case: If there have been no deflations in KEXNW
             * or more iterations, then vary the deflation window size. */

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
                    if (fabs(H[kwtop + (kwtop - 1) * ldh]) >
                        fabs(H[(kwtop - 1) + (kwtop - 2) * ldh]))
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

            /* Aggressive early deflation:
             * split workspace under the subdiagonal into
             * - an nw-by-nw work array V in the lower left-hand-corner,
             * - an NW-by-at-least-NW-but-more-is-better (NW-by-NHO)
             *   horizontal work array along the bottom edge,
             * - an at-least-NW-but-more-is-better (NVE-by-NW)
             *   vertical work array along the left-hand-edge. */

            kv = n - nw;       /* 0-based: Fortran KV = N - NW + 1 */
            kt = nw;           /* 0-based: Fortran KT = NW + 1 */
            nho = (n - nw - 1) - kt;
            kwv = nw + 1;      /* 0-based: Fortran KWV = NW + 2 */
            nve = (n - nw) - kwv;

            /* Aggressive early deflation - uses DLAQR3 (not DLAQR2) */
            dlaqr3(wantt, wantz, n, ktop, kbot, nw, H, ldh, iloz,
                   ihiz, Z, ldz, &ls, &ld, wr, wi, &H[kv], ldh,
                   nho, &H[kv + kt * ldh], ldh, nve, &H[kwv], ldh,
                   work, lwork);

            /* Adjust KBOT accounting for new deflations */
            kbot = kbot - ld;

            /* KS points to the shifts */
            ks = kbot - ls + 1;

            /* Skip an expensive QR sweep if there is a (partly
             * heuristic) reason to expect that many eigenvalues
             * will deflate without it. Here, the QR sweep is
             * skipped if many eigenvalues have just been deflated
             * or if the remaining active block is small. */

            if ((ld == 0) || ((100 * ld <= nw * nibble) &&
                              (kbot - ktop + 1 > (nmin < nwmax ? nmin : nwmax)))) {

                /* NS = nominal number of simultaneous shifts.
                 * This may be lowered (slightly) if DLAQR3
                 * did not provide that many shifts. */
                ns = nsmax < nsr ? nsmax : nsr;
                ns = ns < kbot - ktop ? ns : kbot - ktop;
                if (ns < 2) ns = 2;
                ns = ns - (ns % 2);

                /* If there have been no deflations in a multiple of KEXSH
                 * iterations, then try exceptional shifts. Otherwise use
                 * shifts provided by DLAQR3 above or from the eigenvalues
                 * of a trailing principal submatrix. */

                if ((ndfl % kexsh) == 0) {
                    ks = kbot - ns + 1;
                    for (i = kbot; i >= (ks + 1 > ktop + 2 ? ks + 1 : ktop + 2); i -= 2) {
                        ss = fabs(H[i + (i - 1) * ldh]) + fabs(H[(i - 1) + (i - 2) * ldh]);
                        aa = wilk1 * ss + H[i + i * ldh];
                        bb = ss;
                        cc = wilk2 * ss;
                        dd = aa;
                        dlanv2(&aa, &bb, &cc, &dd, &wr[i - 1], &wi[i - 1],
                               &wr[i], &wi[i], &cs, &sn);
                    }
                    if (ks == ktop) {
                        wr[ks + 1] = H[(ks + 1) + (ks + 1) * ldh];
                        wi[ks + 1] = zero;
                        wr[ks] = wr[ks + 1];
                        wi[ks] = wi[ks + 1];
                    }
                } else {
                    /* Got NS/2 or fewer shifts? Use DLAQR4 or DLAHQR on a
                     * trailing principal submatrix to get more. (Since NS
                     * <= NSMAX <= (N-3)/6, there is enough space below the
                     * subdiagonal to fit an NS-by-NS scratch array.) */

                    if (kbot - ks + 1 <= ns / 2) {
                        ks = kbot - ns + 1;
                        kt = n - ns;   /* 0-based: Fortran KT = N - NS + 1 */
                        dlacpy("A", ns, ns, &H[ks + ks * ldh], ldh,
                               &H[kt], ldh);
                        if (ns > nmin) {
                            /* Use DLAQR4 for larger submatrices */
                            dlaqr4(0, 0, ns, 0, ns - 1, &H[kt], ldh, &wr[ks],
                                   &wi[ks], 0, 0, zdum, 1, work, lwork, &inf);
                        } else {
                            /* Use DLAHQR for smaller submatrices */
                            dlahqr(0, 0, ns, 0, ns - 1, &H[kt], ldh, &wr[ks],
                                   &wi[ks], 0, 0, zdum, 1, &inf);
                        }
                        ks = ks + inf;

                        /* In case of a rare QR failure use eigenvalues
                         * of the trailing 2-by-2 principal submatrix. */
                        if (ks >= kbot) {
                            aa = H[(kbot - 1) + (kbot - 1) * ldh];
                            cc = H[kbot + (kbot - 1) * ldh];
                            bb = H[(kbot - 1) + kbot * ldh];
                            dd = H[kbot + kbot * ldh];
                            dlanv2(&aa, &bb, &cc, &dd, &wr[kbot - 1],
                                   &wi[kbot - 1], &wr[kbot], &wi[kbot],
                                   &cs, &sn);
                            ks = kbot - 1;
                        }
                    }

                    if (kbot - ks + 1 > ns) {
                        /* Sort the shifts (Helps a little)
                         * Bubble sort keeps complex conjugate pairs together. */
                        sorted = 0;
                        for (k = kbot; k >= ks + 1; k--) {
                            if (sorted)
                                break;
                            sorted = 1;
                            for (i = ks; i <= k - 1; i++) {
                                if (fabs(wr[i]) + fabs(wi[i]) <
                                    fabs(wr[i + 1]) + fabs(wi[i + 1])) {
                                    sorted = 0;

                                    swap = wr[i];
                                    wr[i] = wr[i + 1];
                                    wr[i + 1] = swap;

                                    swap = wi[i];
                                    wi[i] = wi[i + 1];
                                    wi[i + 1] = swap;
                                }
                            }
                        }
                    }

                    /* Shuffle shifts into pairs of real shifts and pairs
                     * of complex conjugate shifts assuming complex conjugate
                     * shifts are already adjacent to one another. (Yes, they are.) */

                    for (i = kbot; i >= ks + 2; i -= 2) {
                        if (wi[i] != -wi[i - 1]) {
                            swap = wr[i];
                            wr[i] = wr[i - 1];
                            wr[i - 1] = wr[i - 2];
                            wr[i - 2] = swap;

                            swap = wi[i];
                            wi[i] = wi[i - 1];
                            wi[i - 1] = wi[i - 2];
                            wi[i - 2] = swap;
                        }
                    }
                }

                /* If there are only two shifts and both are real,
                 * then use only one. */
                if (kbot - ks + 1 == 2) {
                    if (wi[kbot] == zero) {
                        if (fabs(wr[kbot] - H[kbot + kbot * ldh]) <
                            fabs(wr[kbot - 1] - H[kbot + kbot * ldh])) {
                            wr[kbot - 1] = wr[kbot];
                        } else {
                            wr[kbot] = wr[kbot - 1];
                        }
                    }
                }

                /* Use up to NS of the smallest magnitude shifts.
                 * If there aren't NS shifts available, then use them all,
                 * possibly dropping one to make the number of shifts even. */
                ns = ns < kbot - ks + 1 ? ns : kbot - ks + 1;
                ns = ns - (ns % 2);
                ks = kbot - ns + 1;

                /* Small-bulge multi-shift QR sweep:
                 * split workspace under the subdiagonal into
                 * - a KDU-by-KDU work array U in the lower left-hand-corner,
                 * - a KDU-by-at-least-KDU-but-more-is-better (KDU-by-NHo)
                 *   horizontal work array WH along the bottom edge,
                 * - and an at-least-KDU-but-more-is-better-by-KDU
                 *   (NVE-by-KDU) vertical work WV along the left-hand-edge. */

                kdu = 2 * ns;
                ku = n - kdu;      /* 0-based: Fortran KU = N - KDU + 1 */
                kwh = kdu;         /* 0-based: Fortran KWH = KDU + 1 */
                nho = (n - kdu + 1 - 4) - kdu;
                kwv = kdu + 3;     /* 0-based: Fortran KWV = KDU + 4 */
                nve = n - kdu - kwv;

                /* Small-bulge multi-shift QR sweep */
                dlaqr5(wantt, wantz, kacc22, n, ktop, kbot, ns,
                       &wr[ks], &wi[ks], H, ldh, iloz, ihiz, Z, ldz,
                       work, 3, &H[ku], ldh, nve, &H[kwv], ldh,
                       nho, &H[ku + kwh * ldh], ldh);
            }

            /* Note progress (or the lack of it) */
            if (ld > 0) {
                ndfl = 1;
            } else {
                ndfl = ndfl + 1;
            }
        }

        /* Iteration limit exceeded. Set INFO to show where
         * the problem occurred and exit. */
        /* Convert KBOT to 1-based for INFO (matching Fortran convention) */
        *info = kbot + 1;

L90:;
    }

    /* Return the optimal value of LWORK */
    work[0] = (f64)lwkopt;
}
