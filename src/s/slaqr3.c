/**
 * @file slaqr3.c
 * @brief SLAQR3 performs aggressive early deflation (recursive version).
 */

#include "semicolon_lapack_single.h"
#include "semicolon_cblas.h"
#include <math.h>

/** @cond */
/* ISPEC=12: NMIN - crossover to SLAHQR (from iparmq.f) */
static INT iparmq_nmin(void)
{
    return 75;
}
/** @endcond */

/**
 * SLAQR3 accepts as input an upper Hessenberg matrix H and performs an
 * orthogonal similarity transformation designed to detect and deflate
 * fully converged eigenvalues from a trailing principal submatrix
 * (aggressive early deflation).
 *
 * SLAQR3 is identical to SLAQR2 except that it calls SLAQR4 instead of
 * SLAHQR for larger deflation windows (when JW > NMIN).
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
SEMICOLON_API void slaqr3(const INT wantt, const INT wantz, const INT n,
                          const INT ktop, const INT kbot, const INT nw,
                          f32* H, const INT ldh,
                          const INT iloz, const INT ihiz,
                          f32* Z, const INT ldz,
                          INT* ns, INT* nd,
                          f32* sr, f32* si,
                          f32* V, const INT ldv,
                          const INT nh, f32* T, const INT ldt,
                          const INT nv, f32* WV, const INT ldwv,
                          f32* work, const INT lwork)
{
    /* Parameters */
    const f32 zero = 0.0f;
    const f32 one = 1.0f;

    /* Local scalars */
    f32 aa, bb, beta, cc, cs, dd, evi, evk, foo, s;
    f32 safmin, smlnum, sn, tau, ulp;
    INT i, ifst, ilst, info, infqr, j, jw, k, kcol, kend, kln;
    INT krow, kwtop, ltop, lwk1, lwk2, lwk3, lwkopt, nmin;
    INT bulge, sorted;

    /* Estimate optimal workspace */
    jw = nw < kbot - ktop + 1 ? nw : kbot - ktop + 1;
    if (jw <= 2) {
        lwkopt = 1;
    } else {
        /* Workspace query call to SGEHRD */
        sgehrd(jw, 0, jw - 2, T, ldt, work, work, -1, &info);
        lwk1 = (INT)work[0];

        /* Workspace query call to SORMHR */
        sormhr("R", "N", jw, jw, 0, jw - 2, T, ldt, work, V, ldv,
               work, -1, &info);
        lwk2 = (INT)work[0];

        /* Workspace query call to SLAQR4 */
        slaqr4(1, 1, jw, 0, jw - 1, T, ldt, sr, si, 0, jw - 1,
               V, ldv, work, -1, &infqr);
        lwk3 = (INT)work[0];

        /* Optimal workspace = MAX(JW + MAX(LWK1, LWK2), LWK3) */
        lwkopt = lwk1 > lwk2 ? lwk1 : lwk2;
        lwkopt = jw + lwkopt;
        if (lwk3 > lwkopt) lwkopt = lwk3;
    }

    /* Quick return in case of workspace query */
    if (lwork == -1) {
        work[0] = (f32)lwkopt;
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
    safmin = slamch("Safe minimum");
    ulp = slamch("Precision");
    smlnum = safmin * ((f32)n / ulp);

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
        if (fabsf(s) <= (smlnum > ulp * fabsf(H[kwtop + kwtop * ldh]) ?
                        smlnum : ulp * fabsf(H[kwtop + kwtop * ldh]))) {
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
    slacpy("U", jw, jw, &H[kwtop + kwtop * ldh], ldh, T, ldt);
    cblas_scopy(jw - 1, &H[(kwtop + 1) + kwtop * ldh], ldh + 1, &T[1], ldt + 1);

    slaset("A", jw, jw, zero, one, V, ldv);

    /* Get crossover point: use SLAQR4 for jw > nmin, else SLAHQR */
    nmin = iparmq_nmin();
    if (jw > nmin) {
        /* 0-based: Fortran calls SLAQR4(.true., .true., JW, 1, JW, ...)
         * which becomes (JW, 0, JW-1) in 0-based indexing */
        slaqr4(1, 1, jw, 0, jw - 1, T, ldt, &sr[kwtop], &si[kwtop],
               0, jw - 1, V, ldv, work, lwork, &infqr);
    } else {
        slahqr(1, 1, jw, 0, jw - 1, T, ldt, &sr[kwtop], &si[kwtop],
               0, jw - 1, V, ldv, &infqr);
    }

    /* STREXC needs a clean margin near the diagonal */
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
            foo = fabsf(T[(*ns - 1) + (*ns - 1) * ldt]);
            if (foo == zero)
                foo = fabsf(s);
            if (fabsf(s * V[(*ns - 1) * ldv]) <=
                (smlnum > ulp * foo ? smlnum : ulp * foo)) {
                /* Deflatable */
                (*ns)--;
            } else {
                /* Undeflatable. Move it up out of the way. */
                ifst = *ns - 1;  /* 0-based */
                strexc("V", jw, T, ldt, V, ldv, &ifst, &ilst, work, &info);
                ilst++;
            }
        } else {
            /* Complex conjugate pair */
            foo = fabsf(T[(*ns - 1) + (*ns - 1) * ldt]) +
                  sqrtf(fabsf(T[(*ns - 1) + (*ns - 2) * ldt])) *
                  sqrtf(fabsf(T[(*ns - 2) + (*ns - 1) * ldt]));
            if (foo == zero)
                foo = fabsf(s);
            if ((fabsf(s * V[(*ns - 1) * ldv]) > fabsf(s * V[(*ns - 2) * ldv]) ?
                 fabsf(s * V[(*ns - 1) * ldv]) : fabsf(s * V[(*ns - 2) * ldv])) <=
                (smlnum > ulp * foo ? smlnum : ulp * foo)) {
                /* Deflatable */
                *ns -= 2;
            } else {
                /* Undeflatable */
                ifst = *ns - 1;
                strexc("V", jw, T, ldt, V, ldv, &ifst, &ilst, work, &info);
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
            kend = i - 1;
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
                    evi = fabsf(T[i + i * ldt]);
                } else {
                    evi = fabsf(T[i + i * ldt]) +
                          sqrtf(fabsf(T[(i + 1) + i * ldt])) *
                          sqrtf(fabsf(T[i + (i + 1) * ldt]));
                }

                if (k == kend) {
                    evk = fabsf(T[k + k * ldt]);
                } else if (T[(k + 1) + k * ldt] == zero) {
                    evk = fabsf(T[k + k * ldt]);
                } else {
                    evk = fabsf(T[k + k * ldt]) +
                          sqrtf(fabsf(T[(k + 1) + k * ldt])) *
                          sqrtf(fabsf(T[k + (k + 1) * ldt]));
                }

                if (evi >= evk) {
                    i = k;
                } else {
                    sorted = 0;
                    ifst = i;
                    ilst = k;
                    strexc("V", jw, T, ldt, V, ldv, &ifst, &ilst, work, &info);
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
            slanv2(&aa, &bb, &cc, &dd, &sr[kwtop + i - 1], &si[kwtop + i - 1],
                   &sr[kwtop + i], &si[kwtop + i], &cs, &sn);
            i -= 2;
        }
    }

    if (*ns < jw || s == zero) {
        if (*ns > 1 && s != zero) {
            /* Reflect spike back into lower triangle */
            cblas_scopy(*ns, V, ldv, work, 1);
            beta = work[0];
            slarfg(*ns, &beta, &work[1], 1, &tau);

            slaset("L", jw - 2, jw - 2, zero, zero, &T[2], ldt);

            slarf1f("L", *ns, jw, work, 1, tau, T, ldt, &work[jw]);
            slarf1f("R", *ns, *ns, work, 1, tau, T, ldt, &work[jw]);
            slarf1f("R", jw, *ns, work, 1, tau, V, ldv, &work[jw]);

            sgehrd(jw, 0, *ns - 1, T, ldt, work, &work[jw], lwork - jw, &info);
        }

        /* Copy updated reduced window into place */
        if (kwtop > 0)
            H[kwtop + (kwtop - 1) * ldh] = s * V[0];
        slacpy("U", jw, jw, T, ldt, &H[kwtop + kwtop * ldh], ldh);
        cblas_scopy(jw - 1, &T[1], ldt + 1, &H[(kwtop + 1) + kwtop * ldh], ldh + 1);

        /* Accumulate orthogonal matrix in order to update H and Z */
        if (*ns > 1 && s != zero)
            sormhr("R", "N", jw, *ns, 0, *ns - 1, T, ldt, work, V, ldv,
                   &work[jw], lwork - jw, &info);

        /* Update vertical slab in H */
        if (wantt) {
            ltop = 0;
        } else {
            ltop = ktop;
        }
        for (krow = ltop; krow < kwtop; krow += nv) {
            kln = nv < kwtop - krow ? nv : kwtop - krow;
            cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       kln, jw, jw, one, &H[krow + kwtop * ldh], ldh,
                       V, ldv, zero, WV, ldwv);
            slacpy("A", kln, jw, WV, ldwv, &H[krow + kwtop * ldh], ldh);
        }

        /* Update horizontal slab in H */
        if (wantt) {
            for (kcol = kbot + 1; kcol < n; kcol += nh) {
                kln = nh < n - kcol ? nh : n - kcol;
                cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                           jw, kln, jw, one, V, ldv,
                           &H[kwtop + kcol * ldh], ldh, zero, T, ldt);
                slacpy("A", jw, kln, T, ldt, &H[kwtop + kcol * ldh], ldh);
            }
        }

        /* Update vertical slab in Z */
        if (wantz) {
            for (krow = iloz; krow <= ihiz; krow += nv) {
                kln = nv < ihiz - krow + 1 ? nv : ihiz - krow + 1;
                cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                           kln, jw, jw, one, &Z[krow + kwtop * ldz], ldz,
                           V, ldv, zero, WV, ldwv);
                slacpy("A", kln, jw, WV, ldwv, &Z[krow + kwtop * ldz], ldz);
            }
        }
    }

    /* Return the number of deflations */
    *nd = jw - *ns;

    /* Return the number of shifts (subtracting infqr takes care of
     * rare QR failure while calculating eigenvalues of the deflation window) */
    *ns = *ns - infqr;

    /* Return optimal workspace */
    work[0] = (f32)lwkopt;
}
