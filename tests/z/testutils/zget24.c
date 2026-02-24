/**
 * @file zget24.c
 * @brief ZGET24 checks the nonsymmetric eigenvalue (Schur form) problem
 *        expert driver ZGEESX.
 *
 * Port of LAPACK's TESTING/EIG/zget24.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"

/* File-static globals for SSLCT COMMON block */
static INT g_selopt;
static INT g_seldim;
static INT g_selval[20];
static f64 g_selwr[20];
static f64 g_selwi[20];

/**
 * ZSLECT returns .TRUE. if the eigenvalue Z is to be selected,
 * and otherwise it returns .FALSE.
 * It is used by ZGEESX to test whether the j-th eigenvalue is to be
 * reordered to the top left corner of the Schur form.
 */
static INT zslect(const c128* z)
{
    INT i;
    f64 rmin, x;

    if (g_selopt == 0) {
        return (creal(*z) < 0.0);
    } else {
        rmin = dlapy2(creal(*z) - g_selwr[0], cimag(*z) - g_selwi[0]);
        INT val = g_selval[0];
        for (i = 1; i < g_seldim; i++) {
            x = dlapy2(creal(*z) - g_selwr[i], cimag(*z) - g_selwi[i]);
            if (x <= rmin) {
                rmin = x;
                val = g_selval[i];
            }
        }
        return val;
    }
}

/**
 * ZGET24 checks the nonsymmetric eigenvalue (Schur form) problem
 * expert driver ZGEESX.
 *
 * If COMP = 0, the first 13 of the following tests will be performed
 * on the input matrix A, and also tests 14 and 15 if LWORK is
 * sufficiently large.
 * If COMP = 1, all 17 tests will be performed.
 *
 *    (1)     0 if T is in Schur form, 1/ulp otherwise (no sorting)
 *    (2)     | A - VS T VS' | / ( n |A| ulp ) (no sorting)
 *    (3)     | I - VS VS' | / ( n ulp ) (no sorting)
 *    (4)     0 if W are eigenvalues of T, 1/ulp otherwise (no sorting)
 *    (5)     0 if T(with VS) = T(without VS), 1/ulp otherwise
 *    (6)     0 if eigenvalues(with VS) = eigenvalues(without VS)
 *    (7)     0 if T is in Schur form (with sorting), 1/ulp otherwise
 *    (8)     | A - VS T VS' | / ( n |A| ulp ) (with sorting)
 *    (9)     | I - VS VS' | / ( n ulp ) (with sorting)
 *   (10)     0 if W are eigenvalues of T (with sorting)
 *   (11)     0 if T(with VS) = T(without VS) (with sorting)
 *   (12)     0 if eigenvalues(with VS) = eigenvalues(without VS)
 *   (13)     if sorting worked and SDIM is the number of eigenvalues selected
 *   (14)     if RCONDE the same no matter if VS and/or RCONDV computed
 *   (15)     if RCONDV the same no matter if VS and/or RCONDE computed
 *   (16)     |RCONDE - RCDEIN| / cond(RCONDE)
 *   (17)     |RCONDV - RCDVIN| / cond(RCONDV)
 */
void zget24(const INT comp, const INT jtype, const f64 thresh,
            const INT n, c128* A, const INT lda,
            c128* H, c128* HT,
            c128* W, c128* WT, c128* WTMP,
            c128* VS, const INT ldvs, c128* VS1,
            const f64 rcdein, const f64 rcdvin,
            const INT nslct, const INT* islct, const INT isrt,
            f64* result, c128* work, const INT lwork,
            f64* rwork, INT* bwork, INT* info)
{
    (void)jtype;
    (void)thresh;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 EPSIN = 5.9605e-8;
    const c128 CONE = CMPLX(1.0, 0.0);
    const c128 CZERO = CMPLX(0.0, 0.0);
    const c128 CNEGONE = CMPLX(-1.0, 0.0);

    INT i, j, kmin, knteig, rsub, sdim, sdim1;
    INT iinfo, isort, itmp;
    f64 anorm, eps, rcnde1, rcndv1, rconde, rcondv;
    f64 smlnum, tol, tolin, ulp, ulpinv, v, vricmp, vrimin, wnorm;
    INT ipnt[20];
    c128 ctmp;

    /* Check for errors */
    *info = 0;
    if (thresh < ZERO) {
        *info = -3;
    } else if (n < 0) {
        *info = -6;
    } else if (lda < 1 || lda < n) {
        *info = -8;
    } else if (ldvs < 1 || ldvs < n) {
        *info = -15;
    } else if (lwork < 2 * n) {
        *info = -24;
    }

    if (*info != 0)
        return;

    /* Quick return if nothing to do */
    for (i = 0; i < 17; i++)
        result[i] = -ONE;

    if (n == 0)
        return;

    /* Important constants */
    smlnum = dlamch("S");
    ulp = dlamch("P");
    ulpinv = ONE / ulp;

    /* Perform tests (1)-(13) */
    g_selopt = 0;
    for (isort = 0; isort <= 1; isort++) {
        if (isort == 0) {
            rsub = 0;
        } else {
            rsub = 6;
        }

        /* Compute Schur form and Schur vectors, and test them */
        zlacpy("F", n, n, A, lda, H, lda);
        zgeesx("V", isort == 0 ? "N" : "S", zslect, "N", n, H, lda, &sdim,
               W, VS, ldvs, &rconde, &rcondv, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[0 + rsub] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            return;
        }
        if (isort == 0) {
            cblas_zcopy(n, W, 1, WTMP, 1);
        }

        /* Do Test (1) or Test (7) */
        result[0 + rsub] = ZERO;
        for (j = 0; j < n - 1; j++) {
            for (i = j + 1; i < n; i++) {
                if (creal(H[i + j * lda]) != 0.0 || cimag(H[i + j * lda]) != 0.0)
                    result[0 + rsub] = ulpinv;
            }
        }

        /* Test (2) or (8): Compute norm(A - Q*H*Q') / (norm(A) * N * ULP) */

        /* Copy A to VS1, used as workspace */
        zlacpy(" ", n, n, A, lda, VS1, ldvs);

        /* Compute Q*H and store in HT */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                    n, n, n, &CONE, VS, ldvs, H, lda, &CZERO, HT, lda);

        /* Compute A - Q*H*Q' */
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasConjTrans,
                    n, n, n, &CNEGONE, HT, lda, VS, ldvs, &CONE, VS1, ldvs);

        anorm = fmax(zlange("1", n, n, A, lda, rwork), smlnum);
        wnorm = zlange("1", n, n, VS1, ldvs, rwork);

        if (anorm > wnorm) {
            result[1 + rsub] = (wnorm / anorm) / (n * ulp);
        } else {
            if (anorm < ONE) {
                result[1 + rsub] = (fmin(wnorm, n * anorm) / anorm) / (n * ulp);
            } else {
                result[1 + rsub] = fmin(wnorm / anorm, (f64)n) / (n * ulp);
            }
        }

        /* Test (3) or (9):  Compute norm( I - Q'*Q ) / ( N * ULP ) */
        zunt01("C", n, n, VS, ldvs, work, lwork, rwork, &result[2 + rsub]);

        /* Do Test (4) or Test (10) */
        result[3 + rsub] = ZERO;
        for (i = 0; i < n; i++) {
            if (creal(H[i + i * lda]) != creal(W[i]) ||
                cimag(H[i + i * lda]) != cimag(W[i]))
                result[3 + rsub] = ulpinv;
        }

        /* Do Test (5) or Test (11) */
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("N", isort == 0 ? "N" : "S", zslect, "N", n, HT, lda, &sdim,
               WT, VS, ldvs, &rconde, &rcondv, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0) {
            result[4 + rsub] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_220;
        }

        result[4 + rsub] = ZERO;
        for (j = 0; j < n; j++) {
            for (i = 0; i < n; i++) {
                if (creal(H[i + j * lda]) != creal(HT[i + j * lda]) ||
                    cimag(H[i + j * lda]) != cimag(HT[i + j * lda]))
                    result[4 + rsub] = ulpinv;
            }
        }

        /* Do Test (6) or Test (12) */
        result[5 + rsub] = ZERO;
        for (i = 0; i < n; i++) {
            if (creal(W[i]) != creal(WT[i]) || cimag(W[i]) != cimag(WT[i]))
                result[5 + rsub] = ulpinv;
        }

        /* Do Test (13) */
        if (isort == 1) {
            result[12] = ZERO;
            knteig = 0;
            for (i = 0; i < n; i++) {
                if (zslect(&W[i]))
                    knteig = knteig + 1;
                if (i < n - 1) {
                    if (zslect(&W[i + 1]) && (!zslect(&W[i])))
                        result[12] = ulpinv;
                }
            }
            if (sdim != knteig)
                result[12] = ulpinv;
        }
    }

    /* If there is enough workspace, perform tests (14) and (15)
     * as well as (10) through (13) */
    if (lwork >= (n * (n + 1)) / 2) {

        /* Compute both RCONDE and RCONDV with VS */
        result[13] = ZERO;
        result[14] = ZERO;
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("V", "S", zslect, "B", n, HT, lda, &sdim1, WT,
               VS1, ldvs, &rconde, &rcondv, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_220;
        }

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (creal(W[i]) != creal(WT[i]) || cimag(W[i]) != cimag(WT[i]))
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (creal(H[i + j * lda]) != creal(HT[i + j * lda]) ||
                    cimag(H[i + j * lda]) != cimag(HT[i + j * lda]))
                    result[10] = ulpinv;
                if (creal(VS[i + j * ldvs]) != creal(VS1[i + j * ldvs]) ||
                    cimag(VS[i + j * ldvs]) != cimag(VS1[i + j * ldvs]))
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute both RCONDE and RCONDV without VS, and compare */
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("N", "S", zslect, "B", n, HT, lda, &sdim1, WT,
               VS1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_220;
        }

        /* Perform tests (14) and (15) */
        if (rcnde1 != rconde)
            result[13] = ulpinv;
        if (rcndv1 != rcondv)
            result[14] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (creal(W[i]) != creal(WT[i]) || cimag(W[i]) != cimag(WT[i]))
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (creal(H[i + j * lda]) != creal(HT[i + j * lda]) ||
                    cimag(H[i + j * lda]) != cimag(HT[i + j * lda]))
                    result[10] = ulpinv;
                if (creal(VS[i + j * ldvs]) != creal(VS1[i + j * ldvs]) ||
                    cimag(VS[i + j * ldvs]) != cimag(VS1[i + j * ldvs]))
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDE with VS, and compare */
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("V", "S", zslect, "E", n, HT, lda, &sdim1, WT,
               VS1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_220;
        }

        /* Perform test (14) */
        if (rcnde1 != rconde)
            result[13] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (creal(W[i]) != creal(WT[i]) || cimag(W[i]) != cimag(WT[i]))
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (creal(H[i + j * lda]) != creal(HT[i + j * lda]) ||
                    cimag(H[i + j * lda]) != cimag(HT[i + j * lda]))
                    result[10] = ulpinv;
                if (creal(VS[i + j * ldvs]) != creal(VS1[i + j * ldvs]) ||
                    cimag(VS[i + j * ldvs]) != cimag(VS1[i + j * ldvs]))
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDE without VS, and compare */
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("N", "S", zslect, "E", n, HT, lda, &sdim1, WT,
               VS1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[13] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_220;
        }

        /* Perform test (14) */
        if (rcnde1 != rconde)
            result[13] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (creal(W[i]) != creal(WT[i]) || cimag(W[i]) != cimag(WT[i]))
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (creal(H[i + j * lda]) != creal(HT[i + j * lda]) ||
                    cimag(H[i + j * lda]) != cimag(HT[i + j * lda]))
                    result[10] = ulpinv;
                if (creal(VS[i + j * ldvs]) != creal(VS1[i + j * ldvs]) ||
                    cimag(VS[i + j * ldvs]) != cimag(VS1[i + j * ldvs]))
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDV with VS, and compare */
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("V", "S", zslect, "V", n, HT, lda, &sdim1, WT,
               VS1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_220;
        }

        /* Perform test (15) */
        if (rcndv1 != rcondv)
            result[14] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (creal(W[i]) != creal(WT[i]) || cimag(W[i]) != cimag(WT[i]))
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (creal(H[i + j * lda]) != creal(HT[i + j * lda]) ||
                    cimag(H[i + j * lda]) != cimag(HT[i + j * lda]))
                    result[10] = ulpinv;
                if (creal(VS[i + j * ldvs]) != creal(VS1[i + j * ldvs]) ||
                    cimag(VS[i + j * ldvs]) != cimag(VS1[i + j * ldvs]))
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;

        /* Compute RCONDV without VS, and compare */
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("N", "S", zslect, "V", n, HT, lda, &sdim1, WT,
               VS1, ldvs, &rcnde1, &rcndv1, work, lwork, rwork,
               bwork, &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[14] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_220;
        }

        /* Perform test (15) */
        if (rcndv1 != rcondv)
            result[14] = ulpinv;

        /* Perform tests (10), (11), (12), and (13) */
        for (i = 0; i < n; i++) {
            if (creal(W[i]) != creal(WT[i]) || cimag(W[i]) != cimag(WT[i]))
                result[9] = ulpinv;
            for (j = 0; j < n; j++) {
                if (creal(H[i + j * lda]) != creal(HT[i + j * lda]) ||
                    cimag(H[i + j * lda]) != cimag(HT[i + j * lda]))
                    result[10] = ulpinv;
                if (creal(VS[i + j * ldvs]) != creal(VS1[i + j * ldvs]) ||
                    cimag(VS[i + j * ldvs]) != cimag(VS1[i + j * ldvs]))
                    result[11] = ulpinv;
            }
        }
        if (sdim != sdim1)
            result[12] = ulpinv;
    }

label_220:

    /* If there are precomputed reciprocal condition numbers, compare
     * computed values with them. */
    if (comp) {

        /* First set up SELOPT, SELDIM, SELVAL, SELWR and SELWI so that
         * the logical function ZSLECT selects the eigenvalues specified
         * by NSLCT, ISLCT and ISRT. */
        g_seldim = n;
        g_selopt = 1;
        eps = fmax(ulp, EPSIN);
        for (i = 0; i < n; i++) {
            ipnt[i] = i;
            g_selval[i] = 0;
            g_selwr[i] = creal(WTMP[i]);
            g_selwi[i] = cimag(WTMP[i]);
        }
        for (i = 0; i < n - 1; i++) {
            kmin = i;
            if (isrt == 0) {
                vrimin = creal(WTMP[i]);
            } else {
                vrimin = cimag(WTMP[i]);
            }
            for (j = i + 1; j < n; j++) {
                if (isrt == 0) {
                    vricmp = creal(WTMP[j]);
                } else {
                    vricmp = cimag(WTMP[j]);
                }
                if (vricmp < vrimin) {
                    kmin = j;
                    vrimin = vricmp;
                }
            }
            ctmp = WTMP[kmin];
            WTMP[kmin] = WTMP[i];
            WTMP[i] = ctmp;
            itmp = ipnt[i];
            ipnt[i] = ipnt[kmin];
            ipnt[kmin] = itmp;
        }
        for (i = 0; i < nslct; i++) {
            g_selval[ipnt[islct[i]]] = 1;
        }

        /* Compute condition numbers */
        zlacpy("F", n, n, A, lda, HT, lda);
        zgeesx("N", "S", zslect, "B", n, HT, lda, &sdim1, WT, VS1,
               ldvs, &rconde, &rcondv, work, lwork, rwork, bwork,
               &iinfo);
        if (iinfo != 0 && iinfo != n + 2) {
            result[15] = ulpinv;
            result[16] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_270;
        }

        /* Compare condition number for average of selected eigenvalues
         * taking its condition number into account */
        anorm = zlange("1", n, n, A, lda, rwork);
        v = fmax((f64)n * eps * anorm, smlnum);
        if (anorm == ZERO)
            v = ONE;
        if (v > rcondv) {
            tol = ONE;
        } else {
            tol = v / rcondv;
        }
        if (v > rcdvin) {
            tolin = ONE;
        } else {
            tolin = v / rcdvin;
        }
        tol = fmax(tol, smlnum / eps);
        tolin = fmax(tolin, smlnum / eps);
        if (eps * (rcdein - tolin) > rconde + tol) {
            result[15] = ulpinv;
        } else if (rcdein - tolin > rconde + tol) {
            result[15] = (rcdein - tolin) / (rconde + tol);
        } else if (rcdein + tolin < eps * (rconde - tol)) {
            result[15] = ulpinv;
        } else if (rcdein + tolin < rconde - tol) {
            result[15] = (rconde - tol) / (rcdein + tolin);
        } else {
            result[15] = ONE;
        }

        /* Compare condition numbers for right invariant subspace
         * taking its condition number into account */
        if (v > rcondv * rconde) {
            tol = rcondv;
        } else {
            tol = v / rconde;
        }
        if (v > rcdvin * rcdein) {
            tolin = rcdvin;
        } else {
            tolin = v / rcdein;
        }
        tol = fmax(tol, smlnum / eps);
        tolin = fmax(tolin, smlnum / eps);
        if (eps * (rcdvin - tolin) > rcondv + tol) {
            result[16] = ulpinv;
        } else if (rcdvin - tolin > rcondv + tol) {
            result[16] = (rcdvin - tolin) / (rcondv + tol);
        } else if (rcdvin + tolin < eps * (rcondv - tol)) {
            result[16] = ulpinv;
        } else if (rcdvin + tolin < rcondv - tol) {
            result[16] = (rcondv - tol) / (rcdvin + tolin);
        } else {
            result[16] = ONE;
        }

label_270:
        ;
    }
}
