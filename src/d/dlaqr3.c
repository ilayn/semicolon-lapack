/**
 * @file dlaqr3.c
 * @brief DLAQR3 performs aggressive early deflation (recursive version).
 */

#include "semicolon_lapack_double.h"
#include <cblas.h>
#include <math.h>

/** @cond */
/* ISPEC=12: NMIN - crossover to DLAHQR (from iparmq.f) */
static int iparmq_nmin(void)
{
    return 75;
}
/** @endcond */

/**
 * DLAQR3 accepts as input an upper Hessenberg matrix H and performs an
 * orthogonal similarity transformation designed to detect and deflate
 * fully converged eigenvalues from a trailing principal submatrix
 * (aggressive early deflation).
 *
 * DLAQR3 is identical to DLAQR2 except that it calls DLAQR4 instead of
 * DLAHQR for larger deflation windows (when JW > NMIN).
 *
 * @param[in] wantt   If nonzero, the Hessenberg matrix H is fully updated.
 * @param[in] wantz   If nonzero, the orthogonal matrix Z is updated.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ktop    First row/column of isolated block (0-based).
 * @param[in] kbot    Last row/column of isolated block (0-based).
 * @param[in] nw      Deflation window size. 1 <= nw <= (kbot - ktop + 1).
 * @param[in,out] H   Double precision array, dimension (ldh, n).
 * @param[in] ldh     Leading dimension of H. ldh >= n.
 * @param[in] iloz    First row of Z to update (0-based).
 * @param[in] ihiz    Last row of Z to update (0-based).
 * @param[in,out] Z   Double precision array, dimension (ldz, n).
 * @param[in] ldz     Leading dimension of Z. ldz >= 1.
 * @param[out] ns     Number of unconverged eigenvalues (shifts).
 * @param[out] nd     Number of converged (deflated) eigenvalues.
 * @param[out] sr     Double precision array, dimension (kbot+1). Real parts.
 * @param[out] si     Double precision array, dimension (kbot+1). Imaginary parts.
 * @param[out] V      Double precision array, dimension (ldv, nw).
 * @param[in] ldv     Leading dimension of V. ldv >= nw.
 * @param[in] nh      Number of columns of T. nh >= nw.
 * @param[out] T      Double precision array, dimension (ldt, nw).
 * @param[in] ldt     Leading dimension of T. ldt >= nw.
 * @param[in] nv      Number of rows of WV. nv >= nw.
 * @param[out] WV     Double precision array, dimension (ldwv, nw).
 * @param[in] ldwv    Leading dimension of WV. ldwv >= nv.
 * @param[out] work   Double precision array, dimension (lwork).
 * @param[in] lwork   Dimension of work array. lwork >= 2*nw.
 *                    If lwork = -1, workspace query is assumed.
 */
SEMICOLON_API void dlaqr3(const int wantt, const int wantz, const int n,
                          const int ktop, const int kbot, const int nw,
                          f64* H, const int ldh,
                          const int iloz, const int ihiz,
                          f64* Z, const int ldz,
                          int* ns, int* nd,
                          f64* sr, f64* si,
                          f64* V, const int ldv,
                          const int nh, f64* T, const int ldt,
                          const int nv, f64* WV, const int ldwv,
                          f64* work, const int lwork)
{
    /* Parameters */
    const f64 zero = 0.0;
    const f64 one = 1.0;

    /* Local scalars */
    f64 aa, bb, beta, cc, cs, dd, evi, evk, foo, s;
    f64 safmin, smlnum, sn, tau, ulp;
    int i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, kln;
    int krow, kwtop, ltop, lwk1, lwk2, lwk3, lwkopt, nmin;
    int bulge, sorted;

    /* Estimate optimal workspace */
    jw = nw < kbot - ktop + 1 ? nw : kbot - ktop + 1;
    if (jw <= 2) {
        lwkopt = 1;
    } else {
        /* Workspace query call to DGEHRD */
        dgehrd(jw, 0, jw - 2, T, ldt, work, work, -1, &info);
        lwk1 = (int)work[0];

        /* Workspace query call to DORMHR */
        dormhr("R", "N", jw, jw, 0, jw - 2, T, ldt, work, V, ldv,
               work, -1, &info);
        lwk2 = (int)work[0];

        /* Workspace query call to DLAQR4 */
        dlaqr4(1, 1, jw, 0, jw - 1, T, ldt, sr, si, 0, jw - 1,
               V, ldv, work, -1, &infqr);
        lwk3 = (int)work[0];

        /* Optimal workspace = MAX(JW + MAX(LWK1, LWK2), LWK3) */
        lwkopt = lwk1 > lwk2 ? lwk1 : lwk2;
        lwkopt = jw + lwkopt;
        if (lwk3 > lwkopt) lwkopt = lwk3;
    }

    /* Quick return in case of workspace query */
    if (lwork == -1) {
        work[0] = (f64)lwkopt;
        return;
    }

    /* Nothing to do for an empty active block */
    *ns = 0;
    *nd = 0;
    work[0] = one;
    if (ktop > kbot)
        return;
    /* Nor for an empty deflation window */
    if (nw < 1)
        return;

    /* Machine constants */
    safmin = dlamch("Safe minimum");
    ulp = dlamch("Precision");
    smlnum = safmin * ((f64)n / ulp);

    /* Setup deflation window */
    jw = nw < kbot - ktop + 1 ? nw : kbot - ktop + 1;
    kwtop = kbot - jw + 1;
    if (kwtop == ktop) {
        s = zero;
    } else {
        s = H[kwtop + (kwtop - 1) * ldh];
    }

    if (kbot == kwtop) {
        /* 1-by-1 deflation window: not much to do */
        sr[kwtop] = H[kwtop + kwtop * ldh];
        si[kwtop] = zero;
        *ns = 1;
        *nd = 0;
        if (fabs(s) <= (smlnum > ulp * fabs(H[kwtop + kwtop * ldh]) ?
                        smlnum : ulp * fabs(H[kwtop + kwtop * ldh]))) {
            *ns = 0;
            *nd = 1;
            if (kwtop > ktop)
                H[kwtop + (kwtop - 1) * ldh] = zero;
        }
        work[0] = one;
        return;
    }

    /* Convert to spike-triangular form. (In case of a rare QR failure,
     * this routine continues to do aggressive early deflation using that
     * part of the deflation window that converged using INFQR here and
     * there to keep track.) */
    dlacpy("U", jw, jw, &H[kwtop + kwtop * ldh], ldh, T, ldt);
    cblas_dcopy(jw - 1, &H[(kwtop + 1) + kwtop * ldh], ldh + 1, &T[1], ldt + 1);

    dlaset("A", jw, jw, zero, one, V, ldv);

    /* Get crossover point: use DLAQR4 for jw > nmin, else DLAHQR */
    nmin = iparmq_nmin();
    if (jw > nmin) {
        /* 0-based: Fortran calls DLAQR4(.true., .true., JW, 1, JW, ...)
         * which becomes (JW, 0, JW-1) in 0-based indexing */
        dlaqr4(1, 1, jw, 0, jw - 1, T, ldt, &sr[kwtop], &si[kwtop],
               0, jw - 1, V, ldv, work, lwork, &infqr);
    } else {
        dlahqr(1, 1, jw, 0, jw - 1, T, ldt, &sr[kwtop], &si[kwtop],
               0, jw - 1, V, ldv, &infqr);
    }

    /* DTREXC needs a clean margin near the diagonal */
    for (j = 0; j < jw - 3; j++) {
        T[(j + 2) + j * ldt] = zero;
        T[(j + 3) + j * ldt] = zero;
    }
    if (jw > 2)
        T[(jw - 1) + (jw - 3) * ldt] = zero;

    /* Deflation detection loop */
    *ns = jw;
    ilst = infqr;  /* 0-based now */

    while (ilst < *ns) {
        if (*ns == 1) {
            bulge = 0;
        } else {
            bulge = T[(*ns - 1) + (*ns - 2) * ldt] != zero;
        }

        /* Small spike tip test for deflation */
        if (!bulge) {
            /* Real eigenvalue */
            foo = fabs(T[(*ns - 1) + (*ns - 1) * ldt]);
            if (foo == zero)
                foo = fabs(s);
            if (fabs(s * V[(*ns - 1) * ldv]) <=
                (smlnum > ulp * foo ? smlnum : ulp * foo)) {
                /* Deflatable */
                (*ns)--;
            } else {
                /* Undeflatable. Move it up out of the way. */
                ifst = *ns - 1;  /* 0-based */
                dtrexc("V", jw, T, ldt, V, ldv, &ifst, &ilst, work, &info);
                ilst++;
            }
        } else {
            /* Complex conjugate pair */
            foo = fabs(T[(*ns - 1) + (*ns - 1) * ldt]) +
                  sqrt(fabs(T[(*ns - 1) + (*ns - 2) * ldt])) *
                  sqrt(fabs(T[(*ns - 2) + (*ns - 1) * ldt]));
            if (foo == zero)
                foo = fabs(s);
            if ((fabs(s * V[(*ns - 1) * ldv]) > fabs(s * V[(*ns - 2) * ldv]) ?
                 fabs(s * V[(*ns - 1) * ldv]) : fabs(s * V[(*ns - 2) * ldv])) <=
                (smlnum > ulp * foo ? smlnum : ulp * foo)) {
                /* Deflatable */
                *ns -= 2;
            } else {
                /* Undeflatable */
                ifst = *ns - 1;
                dtrexc("V", jw, T, ldt, V, ldv, &ifst, &ilst, work, &info);
                ilst += 2;
            }
        }
    }

    /* Return to Hessenberg form */
    if (*ns == 0)
        s = zero;

    if (*ns < jw) {
        /* Sorting diagonal blocks of T improves accuracy for graded matrices.
         * Bubble sort deals well with exchange failures. */
        sorted = 0;
        i = *ns;

        while (!sorted) {
            sorted = 1;
            kend = i;
            i = infqr;
            if (i == *ns - 1) {
                k = i + 1;
            } else if (T[(i + 1) + i * ldt] == zero) {
                k = i + 1;
            } else {
                k = i + 2;
            }

            while (k <= kend) {
                if (k == i + 1) {
                    evi = fabs(T[i + i * ldt]);
                } else {
                    evi = fabs(T[i + i * ldt]) +
                          sqrt(fabs(T[(i + 1) + i * ldt])) *
                          sqrt(fabs(T[i + (i + 1) * ldt]));
                }

                if (k == kend) {
                    evk = fabs(T[k + k * ldt]);
                } else if (T[(k + 1) + k * ldt] == zero) {
                    evk = fabs(T[k + k * ldt]);
                } else {
                    evk = fabs(T[k + k * ldt]) +
                          sqrt(fabs(T[(k + 1) + k * ldt])) *
                          sqrt(fabs(T[k + (k + 1) * ldt]));
                }

                if (evi >= evk) {
                    i = k;
                } else {
                    sorted = 0;
                    ifst = i;
                    ilst = k;
                    dtrexc("V", jw, T, ldt, V, ldv, &ifst, &ilst, work, &info);
                    if (info == 0) {
                        i = ilst;
                    } else {
                        i = k;
                    }
                }

                if (i == kend) {
                    k = i + 1;
                } else if (T[(i + 1) + i * ldt] == zero) {
                    k = i + 1;
                } else {
                    k = i + 2;
                }
            }
        }
    }

    /* Restore shift/eigenvalue array from T */
    i = jw - 1;
    while (i >= infqr) {
        if (i == infqr) {
            sr[kwtop + i] = T[i + i * ldt];
            si[kwtop + i] = zero;
            i--;
        } else if (T[i + (i - 1) * ldt] == zero) {
            sr[kwtop + i] = T[i + i * ldt];
            si[kwtop + i] = zero;
            i--;
        } else {
            aa = T[(i - 1) + (i - 1) * ldt];
            cc = T[i + (i - 1) * ldt];
            bb = T[(i - 1) + i * ldt];
            dd = T[i + i * ldt];
            dlanv2(&aa, &bb, &cc, &dd, &sr[kwtop + i - 1], &si[kwtop + i - 1],
                   &sr[kwtop + i], &si[kwtop + i], &cs, &sn);
            i -= 2;
        }
    }

    if (*ns < jw || s == zero) {
        if (*ns > 1 && s != zero) {
            /* Reflect spike back into lower triangle */
            cblas_dcopy(*ns, V, ldv, work, 1);
            beta = work[0];
            dlarfg(*ns, &beta, &work[1], 1, &tau);

            dlaset("L", jw - 2, jw - 2, zero, zero, &T[2], ldt);

            dlarf1f("L", *ns, jw, work, 1, tau, T, ldt, &work[jw]);
            dlarf1f("R", *ns, *ns, work, 1, tau, T, ldt, &work[jw]);
            dlarf1f("R", jw, *ns, work, 1, tau, V, ldv, &work[jw]);

            dgehrd(jw, 0, *ns - 1, T, ldt, work, &work[jw], lwork - jw, &info);
        }

        /* Copy updated reduced window into place */
        if (kwtop > 0)
            H[kwtop + (kwtop - 1) * ldh] = s * V[0];
        dlacpy("U", jw, jw, T, ldt, &H[kwtop + kwtop * ldh], ldh);
        cblas_dcopy(jw - 1, &T[1], ldt + 1, &H[(kwtop + 1) + kwtop * ldh], ldh + 1);

        /* Accumulate orthogonal matrix in order to update H and Z */
        if (*ns > 1 && s != zero)
            dormhr("R", "N", jw, *ns, 0, *ns - 1, T, ldt, work, V, ldv,
                   &work[jw], lwork - jw, &info);

        /* Update vertical slab in H */
        if (wantt) {
            ltop = 0;
        } else {
            ltop = ktop;
        }
        for (krow = ltop; krow < kwtop; krow += nv) {
            kln = nv < kwtop - krow ? nv : kwtop - krow;
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       kln, jw, jw, one, &H[krow + kwtop * ldh], ldh,
                       V, ldv, zero, WV, ldwv);
            dlacpy("A", kln, jw, WV, ldwv, &H[krow + kwtop * ldh], ldh);
        }

        /* Update horizontal slab in H */
        if (wantt) {
            for (kcol = kbot + 1; kcol < n; kcol += nh) {
                kln = nh < n - kcol ? nh : n - kcol;
                cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                           jw, kln, jw, one, V, ldv,
                           &H[kwtop + kcol * ldh], ldh, zero, T, ldt);
                dlacpy("A", jw, kln, T, ldt, &H[kwtop + kcol * ldh], ldh);
            }
        }

        /* Update vertical slab in Z */
        if (wantz) {
            for (krow = iloz; krow <= ihiz; krow += nv) {
                kln = nv < ihiz - krow + 1 ? nv : ihiz - krow + 1;
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                           kln, jw, jw, one, &Z[krow + kwtop * ldz], ldz,
                           V, ldv, zero, WV, ldwv);
                dlacpy("A", kln, jw, WV, ldwv, &Z[krow + kwtop * ldz], ldz);
            }
        }
    }

    /* Return the number of deflations */
    *nd = jw - *ns;

    /* Return the number of shifts (subtracting infqr takes care of
     * rare QR failure while calculating eigenvalues of the deflation window) */
    *ns = *ns - infqr;

    /* Return optimal workspace */
    work[0] = (f64)lwkopt;
}
