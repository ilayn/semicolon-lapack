/**
 * @file dget23.c
 * @brief DGET23 checks the nonsymmetric eigenvalue problem driver DGEEVX.
 *
 * Port of LAPACK's TESTING/EIG/dget23.f to C.
 */

#include <math.h>
#include "semicolon_cblas.h"
#include "verify.h"
/**
 * DGET23 checks the nonsymmetric eigenvalue problem driver DGEEVX.
 *
 * If COMP = 0, the first 8 of the following tests will be
 * performed on the input matrix A, and also test 9 if LWORK is
 * sufficiently large.
 * If COMP = 1 all 11 tests will be performed.
 *
 *    (1)     | A * VR - VR * W | / ( n |A| ulp )
 *    (2)     | A**H * VL - VL * W**H | / ( n |A| ulp )
 *    (3)     | |VR(i)| - 1 | / ulp and largest component real
 *    (4)     | |VL(i)| - 1 | / ulp and largest component real
 *    (5)     0 if W(full) = W(partial), 1/ulp otherwise
 *    (6)     0 if VR(full) = VR(partial), 1/ulp otherwise
 *    (7)     0 if VL(full) = VL(partial), 1/ulp otherwise
 *    (8)     0 if SCALE, ILO, IHI, ABNRM (full) =
 *                 SCALE, ILO, IHI, ABNRM (partial), 1/ulp otherwise
 *    (9)     0 if RCONDV(full) = RCONDV(partial), 1/ulp otherwise
 *   (10)     |RCONDV - RCDVIN| / cond(RCONDV)
 *   (11)     |RCONDE - RCDEIN| / cond(RCONDE)
 */
void dget23(const INT comp, const char* balanc, const INT jtype,
            const f64 thresh, const INT n,
            f64* A, const INT lda, f64* H,
            f64* wr, f64* wi, f64* wr1, f64* wi1,
            f64* VL, const INT ldvl, f64* VR, const INT ldvr,
            f64* LRE, const INT ldlre,
            f64* rcondv, f64* rcndv1, const f64* rcdvin,
            f64* rconde, f64* rcnde1, const f64* rcdein,
            f64* scale, f64* scale1, f64* result,
            f64* work, const INT lwork, INT* iwork, INT* info)
{
    (void)jtype;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const f64 TWO = 2.0;
    const f64 EPSIN = 5.9605e-8;

    INT nobal, balok;
    INT i, j, jj, kmin;
    INT ihi, ihi1, iinfo, ilo, ilo1, isens, isensm;
    f64 abnrm, abnrm1, eps, smlnum, tnrm, tol, tolin;
    f64 ulp, ulpinv, v, vimin, vmax, vmx, vrmin, vrmx, vtst;
    const char* sens[2] = {"N", "V"};
    f64 dum[1], res[2];
    char sense_str[2];

    /* Check for errors */
    nobal = (balanc[0] == 'N' || balanc[0] == 'n');
    balok = nobal ||
            (balanc[0] == 'P' || balanc[0] == 'p') ||
            (balanc[0] == 'S' || balanc[0] == 's') ||
            (balanc[0] == 'B' || balanc[0] == 'b');
    *info = 0;
    if (!balok) {
        *info = -2;
    } else if (thresh < ZERO) {
        *info = -4;
    } else if (n < 0) {
        *info = -5;
    } else if (lda < 1 || lda < n) {
        *info = -7;
    } else if (ldvl < 1 || ldvl < n) {
        *info = -13;
    } else if (ldvr < 1 || ldvr < n) {
        *info = -15;
    } else if (ldlre < 1 || ldlre < n) {
        *info = -17;
    } else if (lwork < 3 * n || (comp && lwork < 6 * n + n * n)) {
        *info = -28;
    }

    if (*info != 0)
        return;

    /* Quick return if nothing to do */
    for (i = 0; i < 11; i++)
        result[i] = -ONE;

    if (n == 0)
        return;

    /* More Important constants */
    ulp = dlamch("P");
    smlnum = dlamch("S");
    ulpinv = ONE / ulp;

    /* Compute eigenvalues and eigenvectors, and test them */
    if (lwork >= 6 * n + n * n) {
        sense_str[0] = 'B';
        sense_str[1] = '\0';
        isensm = 2;
    } else {
        sense_str[0] = 'E';
        sense_str[1] = '\0';
        isensm = 1;
    }
    dlacpy("F", n, n, A, lda, H, lda);
    dgeevx(balanc, "V", "V", sense_str, n, H, lda, wr, wi, VL, ldvl,
           VR, ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv,
           work, lwork, iwork, &iinfo);
    if (iinfo != 0) {
        result[0] = ulpinv;
        *info = (iinfo < 0) ? -iinfo : iinfo;
        return;
    }

    /* Do Test (1) */
    dget22("N", "N", "N", n, A, lda, VR, ldvr, wr, wi, work, res);
    result[0] = res[0];

    /* Do Test (2) */
    dget22("T", "N", "T", n, A, lda, VL, ldvl, wr, wi, work, res);
    result[1] = res[0];

    /* Do Test (3) */
    for (j = 0; j < n; j++) {
        tnrm = ONE;
        if (wi[j] == ZERO) {
            tnrm = cblas_dnrm2(n, &VR[j * ldvr], 1);
        } else if (wi[j] > ZERO) {
            tnrm = dlapy2(cblas_dnrm2(n, &VR[j * ldvr], 1),
                          cblas_dnrm2(n, &VR[(j + 1) * ldvr], 1));
        }
        result[2] = fmax(result[2],
                         fmin(ulpinv, fabs(tnrm - ONE) / ulp));
        if (wi[j] > ZERO) {
            vmx = ZERO;
            vrmx = ZERO;
            for (jj = 0; jj < n; jj++) {
                vtst = dlapy2(VR[jj + j * ldvr], VR[jj + (j + 1) * ldvr]);
                if (vtst > vmx)
                    vmx = vtst;
                if (VR[jj + (j + 1) * ldvr] == ZERO &&
                    fabs(VR[jj + j * ldvr]) > vrmx)
                    vrmx = fabs(VR[jj + j * ldvr]);
            }
            if (vrmx / vmx < ONE - TWO * ulp)
                result[2] = ulpinv;
        }
    }

    /* Do Test (4) */
    for (j = 0; j < n; j++) {
        tnrm = ONE;
        if (wi[j] == ZERO) {
            tnrm = cblas_dnrm2(n, &VL[j * ldvl], 1);
        } else if (wi[j] > ZERO) {
            tnrm = dlapy2(cblas_dnrm2(n, &VL[j * ldvl], 1),
                          cblas_dnrm2(n, &VL[(j + 1) * ldvl], 1));
        }
        result[3] = fmax(result[3],
                         fmin(ulpinv, fabs(tnrm - ONE) / ulp));
        if (wi[j] > ZERO) {
            vmx = ZERO;
            vrmx = ZERO;
            for (jj = 0; jj < n; jj++) {
                vtst = dlapy2(VL[jj + j * ldvl], VL[jj + (j + 1) * ldvl]);
                if (vtst > vmx)
                    vmx = vtst;
                if (VL[jj + (j + 1) * ldvl] == ZERO &&
                    fabs(VL[jj + j * ldvl]) > vrmx)
                    vrmx = fabs(VL[jj + j * ldvl]);
            }
            if (vrmx / vmx < ONE - TWO * ulp)
                result[3] = ulpinv;
        }
    }

    /* Test for all options of computing condition numbers */
    for (isens = 0; isens < isensm; isens++) {

        /* Compute eigenvalues only, and test them */
        dlacpy("F", n, n, A, lda, H, lda);
        dgeevx(balanc, "N", "N", sens[isens], n, H, lda, wr1, wi1, dum,
               1, dum, 1, &ilo1, &ihi1, scale1, &abnrm1, rcnde1,
               rcndv1, work, lwork, iwork, &iinfo);
        if (iinfo != 0) {
            result[0] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_190;
        }

        /* Do Test (5) */
        for (j = 0; j < n; j++) {
            if (wr[j] != wr1[j] || wi[j] != wi1[j])
                result[4] = ulpinv;
        }

        /* Do Test (8) */
        if (!nobal) {
            for (j = 0; j < n; j++) {
                if (scale[j] != scale1[j])
                    result[7] = ulpinv;
            }
            if (ilo != ilo1)
                result[7] = ulpinv;
            if (ihi != ihi1)
                result[7] = ulpinv;
            if (abnrm != abnrm1)
                result[7] = ulpinv;
        }

        /* Do Test (9) */
        if (isens == 1 && n > 1) {
            for (j = 0; j < n; j++) {
                if (rcondv[j] != rcndv1[j])
                    result[8] = ulpinv;
            }
        }

        /* Compute eigenvalues and right eigenvectors, and test them */
        dlacpy("F", n, n, A, lda, H, lda);
        dgeevx(balanc, "N", "V", sens[isens], n, H, lda, wr1, wi1, dum,
               1, LRE, ldlre, &ilo1, &ihi1, scale1, &abnrm1, rcnde1,
               rcndv1, work, lwork, iwork, &iinfo);
        if (iinfo != 0) {
            result[0] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_190;
        }

        /* Do Test (5) again */
        for (j = 0; j < n; j++) {
            if (wr[j] != wr1[j] || wi[j] != wi1[j])
                result[4] = ulpinv;
        }

        /* Do Test (6) */
        for (j = 0; j < n; j++) {
            for (jj = 0; jj < n; jj++) {
                if (VR[j + jj * ldvr] != LRE[j + jj * ldlre])
                    result[5] = ulpinv;
            }
        }

        /* Do Test (8) again */
        if (!nobal) {
            for (j = 0; j < n; j++) {
                if (scale[j] != scale1[j])
                    result[7] = ulpinv;
            }
            if (ilo != ilo1)
                result[7] = ulpinv;
            if (ihi != ihi1)
                result[7] = ulpinv;
            if (abnrm != abnrm1)
                result[7] = ulpinv;
        }

        /* Do Test (9) again */
        if (isens == 1 && n > 1) {
            for (j = 0; j < n; j++) {
                if (rcondv[j] != rcndv1[j])
                    result[8] = ulpinv;
            }
        }

        /* Compute eigenvalues and left eigenvectors, and test them */
        dlacpy("F", n, n, A, lda, H, lda);
        dgeevx(balanc, "V", "N", sens[isens], n, H, lda, wr1, wi1, LRE,
               ldlre, dum, 1, &ilo1, &ihi1, scale1, &abnrm1, rcnde1,
               rcndv1, work, lwork, iwork, &iinfo);
        if (iinfo != 0) {
            result[0] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_190;
        }

        /* Do Test (5) again */
        for (j = 0; j < n; j++) {
            if (wr[j] != wr1[j] || wi[j] != wi1[j])
                result[4] = ulpinv;
        }

        /* Do Test (7) */
        for (j = 0; j < n; j++) {
            for (jj = 0; jj < n; jj++) {
                if (VL[j + jj * ldvl] != LRE[j + jj * ldlre])
                    result[6] = ulpinv;
            }
        }

        /* Do Test (8) again */
        if (!nobal) {
            for (j = 0; j < n; j++) {
                if (scale[j] != scale1[j])
                    result[7] = ulpinv;
            }
            if (ilo != ilo1)
                result[7] = ulpinv;
            if (ihi != ihi1)
                result[7] = ulpinv;
            if (abnrm != abnrm1)
                result[7] = ulpinv;
        }

        /* Do Test (9) again */
        if (isens == 1 && n > 1) {
            for (j = 0; j < n; j++) {
                if (rcondv[j] != rcndv1[j])
                    result[8] = ulpinv;
            }
        }

label_190:
        ;
    }

    /* If COMP, compare condition numbers to precomputed ones */
    if (comp) {
        dlacpy("F", n, n, A, lda, H, lda);
        dgeevx("N", "V", "V", "B", n, H, lda, wr, wi, VL, ldvl,
               VR, ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv,
               work, lwork, iwork, &iinfo);
        if (iinfo != 0) {
            result[0] = ulpinv;
            *info = (iinfo < 0) ? -iinfo : iinfo;
            goto label_250;
        }

        /* Sort eigenvalues and condition numbers lexicographically
         * to compare with inputs */
        for (i = 0; i < n - 1; i++) {
            kmin = i;
            vrmin = wr[i];
            vimin = wi[i];
            for (j = i + 1; j < n; j++) {
                if (wr[j] < vrmin) {
                    kmin = j;
                    vrmin = wr[j];
                    vimin = wi[j];
                }
            }
            wr[kmin] = wr[i];
            wi[kmin] = wi[i];
            wr[i] = vrmin;
            wi[i] = vimin;
            vrmin = rconde[kmin];
            rconde[kmin] = rconde[i];
            rconde[i] = vrmin;
            vrmin = rcondv[kmin];
            rcondv[kmin] = rcondv[i];
            rcondv[i] = vrmin;
        }

        /* Compare condition numbers for eigenvectors
         * taking their condition numbers into account */
        result[9] = ZERO;
        eps = fmax(EPSIN, ulp);
        v = fmax((f64)n * eps * abnrm, smlnum);
        if (abnrm == ZERO)
            v = ONE;
        for (i = 0; i < n; i++) {
            if (v > rcondv[i] * rconde[i]) {
                tol = rcondv[i];
            } else {
                tol = v / rconde[i];
            }
            if (v > rcdvin[i] * rcdein[i]) {
                tolin = rcdvin[i];
            } else {
                tolin = v / rcdein[i];
            }
            tol = fmax(tol, smlnum / eps);
            tolin = fmax(tolin, smlnum / eps);
            if (eps * (rcdvin[i] - tolin) > rcondv[i] + tol) {
                vmax = ONE / eps;
            } else if (rcdvin[i] - tolin > rcondv[i] + tol) {
                vmax = (rcdvin[i] - tolin) / (rcondv[i] + tol);
            } else if (rcdvin[i] + tolin < eps * (rcondv[i] - tol)) {
                vmax = ONE / eps;
            } else if (rcdvin[i] + tolin < rcondv[i] - tol) {
                vmax = (rcondv[i] - tol) / (rcdvin[i] + tolin);
            } else {
                vmax = ONE;
            }
            result[9] = fmax(result[9], vmax);
        }

        /* Compare condition numbers for eigenvalues
         * taking their condition numbers into account */
        result[10] = ZERO;
        for (i = 0; i < n; i++) {
            if (v > rcondv[i]) {
                tol = ONE;
            } else {
                tol = v / rcondv[i];
            }
            if (v > rcdvin[i]) {
                tolin = ONE;
            } else {
                tolin = v / rcdvin[i];
            }
            tol = fmax(tol, smlnum / eps);
            tolin = fmax(tolin, smlnum / eps);
            if (eps * (rcdein[i] - tolin) > rconde[i] + tol) {
                vmax = ONE / eps;
            } else if (rcdein[i] - tolin > rconde[i] + tol) {
                vmax = (rcdein[i] - tolin) / (rconde[i] + tol);
            } else if (rcdein[i] + tolin < eps * (rconde[i] - tol)) {
                vmax = ONE / eps;
            } else if (rcdein[i] + tolin < rconde[i] - tol) {
                vmax = (rconde[i] - tol) / (rcdein[i] + tolin);
            } else {
                vmax = ONE;
            }
            result[10] = fmax(result[10], vmax);
        }
label_250:
        ;
    }
}
