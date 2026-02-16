/**
 * @file claqr3.c
 * @brief CLAQR3 performs aggressive early deflation (recursive version).
 */

#include "semicolon_lapack_complex_single.h"
#include <complex.h>
#include <cblas.h>
#include <math.h>

/** @cond */
static int iparmq_nmin(void)
{
    return 75;
}
/** @endcond */

/**
 * CLAQR3 performs the unitary similarity transformation of a Hessenberg
 * matrix to detect and deflate fully converged eigenvalues from a trailing
 * principal submatrix (aggressive early deflation).
 *
 * CLAQR3 is identical to CLAQR2 except that it calls CLAQR4 instead of
 * CLAHQR for larger deflation windows (when JW > NMIN).
 *
 * @param[in] wantt   If nonzero, the Hessenberg matrix H is fully updated.
 * @param[in] wantz   If nonzero, the unitary matrix Z is updated.
 * @param[in] n       The order of the matrix H. n >= 0.
 * @param[in] ktop    First row/column of isolated block (0-based).
 * @param[in] kbot    Last row/column of isolated block (0-based).
 * @param[in] nw      Deflation window size. 1 <= nw <= (kbot - ktop + 1).
 * @param[in,out] H   Complex array, dimension (ldh, n).
 * @param[in] ldh     Leading dimension of H. ldh >= n.
 * @param[in] iloz    First row of Z to update (0-based).
 * @param[in] ihiz    Last row of Z to update (0-based).
 * @param[in,out] Z   Complex array, dimension (ldz, n).
 * @param[in] ldz     Leading dimension of Z. ldz >= 1.
 * @param[out] ns     Number of unconverged eigenvalues (shifts).
 * @param[out] nd     Number of converged (deflated) eigenvalues.
 * @param[out] SH     Complex array, dimension (kbot+1). Eigenvalues/shifts.
 * @param[out] V      Complex array, dimension (ldv, nw).
 * @param[in] ldv     Leading dimension of V. ldv >= nw.
 * @param[in] nh      Number of columns of T. nh >= nw.
 * @param[out] T      Complex array, dimension (ldt, nw).
 * @param[in] ldt     Leading dimension of T. ldt >= nw.
 * @param[in] nv      Number of rows of WV. nv >= nw.
 * @param[out] WV     Complex array, dimension (ldwv, nw).
 * @param[in] ldwv    Leading dimension of WV. ldwv >= nv.
 * @param[out] work   Complex array, dimension (lwork).
 * @param[in] lwork   Dimension of work array. lwork >= 2*nw.
 *                    If lwork = -1, workspace query is assumed.
 */
void claqr3(const int wantt, const int wantz, const int n,
            const int ktop, const int kbot, const int nw,
            c64* H, const int ldh,
            const int iloz, const int ihiz,
            c64* Z, const int ldz,
            int* ns, int* nd,
            c64* SH,
            c64* V, const int ldv,
            const int nh, c64* T, const int ldt,
            const int nv, c64* WV, const int ldwv,
            c64* work, const int lwork)
{
    const c64 czero = 0.0f;
    const c64 cone = 1.0f;
    const f32 rzero = 0.0f;

    c64 s, tau;
    f32 foo, safmin, smlnum, ulp;
    int i, ifst, ilst, info, infqr, j, jw, kcol, kln, knt;
    int krow, kwtop, ltop, lwk1, lwk2, lwk3, lwkopt, nmin;

    /* Estimate optimal workspace */
    jw = nw < kbot - ktop + 1 ? nw : kbot - ktop + 1;
    if (jw <= 2) {
        lwkopt = 1;
    } else {
        cgehrd(jw, 0, jw - 2, T, ldt, work, work, -1, &info);
        lwk1 = (int)crealf(work[0]);

        cunmhr("R", "N", jw, jw, 0, jw - 2, T, ldt, work, V, ldv,
               work, -1, &info);
        lwk2 = (int)crealf(work[0]);

        claqr4(1, 1, jw, 0, jw - 1, T, ldt, SH, 0, jw - 1,
               V, ldv, work, -1, &infqr);
        lwk3 = (int)crealf(work[0]);

        lwkopt = lwk1 > lwk2 ? lwk1 : lwk2;
        lwkopt = jw + lwkopt;
        if (lwk3 > lwkopt) lwkopt = lwk3;
    }

    if (lwork == -1) {
        work[0] = (f32)lwkopt;
        return;
    }

    *ns = 0;
    *nd = 0;
    work[0] = cone;
    if (ktop > kbot)
        return;
    if (nw < 1)
        return;

    safmin = slamch("Safe minimum");
    ulp = slamch("Precision");
    smlnum = safmin * ((f32)n / ulp);

    jw = nw < kbot - ktop + 1 ? nw : kbot - ktop + 1;
    kwtop = kbot - jw + 1;
    if (kwtop == ktop) {
        s = czero;
    } else {
        s = H[kwtop + (kwtop - 1) * ldh];
    }

    if (kbot == kwtop) {
        SH[kwtop] = H[kwtop + kwtop * ldh];
        *ns = 1;
        *nd = 0;
        if (cabs1f(s) <= (smlnum > ulp * cabs1f(H[kwtop + kwtop * ldh]) ?
                         smlnum : ulp * cabs1f(H[kwtop + kwtop * ldh]))) {
            *ns = 0;
            *nd = 1;
            if (kwtop > ktop)
                H[kwtop + (kwtop - 1) * ldh] = czero;
        }
        work[0] = cone;
        return;
    }

    /* Convert to spike-triangular form */
    clacpy("U", jw, jw, &H[kwtop + kwtop * ldh], ldh, T, ldt);
    cblas_ccopy(jw - 1, &H[(kwtop + 1) + kwtop * ldh], ldh + 1, &T[1], ldt + 1);

    claset("A", jw, jw, czero, cone, V, ldv);

    nmin = iparmq_nmin();
    if (jw > nmin) {
        claqr4(1, 1, jw, 0, jw - 1, T, ldt, &SH[kwtop],
               0, jw - 1, V, ldv, work, lwork, &infqr);
    } else {
        clahqr(1, 1, jw, 0, jw - 1, T, ldt, &SH[kwtop],
               0, jw - 1, V, ldv, &infqr);
    }

    /* Deflation detection loop */
    *ns = jw;
    ilst = infqr;
    for (knt = infqr; knt < jw; knt++) {

        foo = cabs1f(T[(*ns - 1) + (*ns - 1) * ldt]);
        if (foo == rzero)
            foo = cabs1f(s);
        if (cabs1f(s) * cabs1f(V[(*ns - 1) * ldv]) <=
            (smlnum > ulp * foo ? smlnum : ulp * foo)) {
            (*ns)--;
        } else {
            ifst = *ns - 1;
            ctrexc("V", jw, T, ldt, V, ldv, ifst, ilst, &info);
            ilst++;
        }
    }

    if (*ns == 0)
        s = czero;

    if (*ns < jw) {
        /* Sorting the diagonal of T improves accuracy for graded matrices */
        for (i = infqr; i < *ns; i++) {
            ifst = i;
            for (j = i + 1; j < *ns; j++) {
                if (cabs1f(T[j + j * ldt]) > cabs1f(T[ifst + ifst * ldt]))
                    ifst = j;
            }
            ilst = i;
            if (ifst != ilst)
                ctrexc("V", jw, T, ldt, V, ldv, ifst, ilst, &info);
        }
    }

    /* Restore shift/eigenvalue array from T */
    for (i = infqr; i < jw; i++)
        SH[kwtop + i] = T[i + i * ldt];

    if (*ns < jw || s == czero) {
        if (*ns > 1 && s != czero) {
            /* Reflect spike back into lower triangle */
            cblas_ccopy(*ns, V, ldv, work, 1);
            for (i = 0; i < *ns; i++)
                work[i] = conjf(work[i]);
            clarfg(*ns, &work[0], &work[1], 1, &tau);

            claset("L", jw - 2, jw - 2, czero, czero, &T[2], ldt);

            clarf1f("L", *ns, jw, work, 1, conjf(tau), T, ldt, &work[jw]);
            clarf1f("R", *ns, *ns, work, 1, tau, T, ldt, &work[jw]);
            clarf1f("R", jw, *ns, work, 1, tau, V, ldv, &work[jw]);

            cgehrd(jw, 0, *ns - 1, T, ldt, work, &work[jw], lwork - jw, &info);
        }

        /* Copy updated reduced window into place */
        if (kwtop > 0)
            H[kwtop + (kwtop - 1) * ldh] = s * conjf(V[0]);
        clacpy("U", jw, jw, T, ldt, &H[kwtop + kwtop * ldh], ldh);
        cblas_ccopy(jw - 1, &T[1], ldt + 1, &H[(kwtop + 1) + kwtop * ldh], ldh + 1);

        /* Accumulate unitary matrix in order to update H and Z */
        if (*ns > 1 && s != czero)
            cunmhr("R", "N", jw, *ns, 0, *ns - 1, T, ldt, work, V, ldv,
                   &work[jw], lwork - jw, &info);

        /* Update vertical slab in H */
        if (wantt) {
            ltop = 0;
        } else {
            ltop = ktop;
        }
        for (krow = ltop; krow < kwtop; krow += nv) {
            kln = nv < kwtop - krow ? nv : kwtop - krow;
            cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                       kln, jw, jw, &cone, &H[krow + kwtop * ldh], ldh,
                       V, ldv, &czero, WV, ldwv);
            clacpy("A", kln, jw, WV, ldwv, &H[krow + kwtop * ldh], ldh);
        }

        /* Update horizontal slab in H */
        if (wantt) {
            for (kcol = kbot + 1; kcol < n; kcol += nh) {
                kln = nh < n - kcol ? nh : n - kcol;
                cblas_cgemm(CblasColMajor, CblasConjTrans, CblasNoTrans,
                           jw, kln, jw, &cone, V, ldv,
                           &H[kwtop + kcol * ldh], ldh, &czero, T, ldt);
                clacpy("A", jw, kln, T, ldt, &H[kwtop + kcol * ldh], ldh);
            }
        }

        /* Update vertical slab in Z */
        if (wantz) {
            for (krow = iloz; krow <= ihiz; krow += nv) {
                kln = nv < ihiz - krow + 1 ? nv : ihiz - krow + 1;
                cblas_cgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                           kln, jw, jw, &cone, &Z[krow + kwtop * ldz], ldz,
                           V, ldv, &czero, WV, ldwv);
                clacpy("A", kln, jw, WV, ldwv, &Z[krow + kwtop * ldz], ldz);
            }
        }
    }

    *nd = jw - *ns;

    *ns = *ns - infqr;

    work[0] = (f32)lwkopt;
}
