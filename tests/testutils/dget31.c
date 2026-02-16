/**
 * @file dget31.c
 * @brief DGET31 tests DLALN2, a routine for solving
 *        (ca A - w D)X = sB where A is 1x1 or 2x2.
 */

#include "verify.h"
#include <math.h>

extern f64 dlamch(const char* cmach);
extern void dlaln2(const int ltrans, const int na, const int nw,
                   const f64 smin, const f64 ca, const f64* A, const int lda,
                   const f64 d1, const f64 d2, const f64* B, const int ldb,
                   const f64 wr, const f64 wi, f64* X, const int ldx,
                   f64* scale, f64* xnorm, int* info);

/**
 * DGET31 tests DLALN2, a routine for solving
 *
 *    (ca A - w D)X = sB
 *
 * where A is an NA by NA matrix (NA=1 or 2 only), w is a real (NW=1) or
 * complex (NW=2) constant, ca is a real constant, D is an NA by NA real
 * diagonal matrix, and B is an NA by NW matrix (when NW=2 the second
 * column of B contains the imaginary part of the solution).  The code
 * returns X and s, where s is a scale factor, less than or equal to 1,
 * which is chosen to avoid overflow in X.
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Number of examples with INFO < 0 (ninfo[0]) and > 0 (ninfo[1]).
 * @param[out]    knt     Total number of examples tested.
 */
void dget31(f64* rmax, int* lmax, int ninfo[2], int* knt)
{
    const f64 ZERO  = 0.0;
    const f64 HALF  = 0.5;
    const f64 ONE   = 1.0;
    const f64 TWO   = 2.0;
    const f64 THREE = 3.0;
    const f64 FOUR  = 4.0;
    const f64 SEVEN = 7.0;
    const f64 TEN   = 10.0;
    const f64 TWNONE = 21.0;

    f64 eps = dlamch("P");
    f64 unfl = dlamch("U");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    f64 vsmin[4], vab[3], vwr[4], vwi[4], vdd[4], vca[5];
    vsmin[0] = smlnum;
    vsmin[1] = eps;
    vsmin[2] = ONE / (TEN * TEN);
    vsmin[3] = ONE / eps;
    vab[0] = sqrt(smlnum);
    vab[1] = ONE;
    vab[2] = sqrt(bignum);
    vwr[0] = ZERO;
    vwr[1] = HALF;
    vwr[2] = TWO;
    vwr[3] = ONE;
    vwi[0] = smlnum;
    vwi[1] = eps;
    vwi[2] = ONE;
    vwi[3] = TWO;
    vdd[0] = sqrt(smlnum);
    vdd[1] = ONE;
    vdd[2] = TWO;
    vdd[3] = sqrt(bignum);
    vca[0] = ZERO;
    vca[1] = sqrt(smlnum);
    vca[2] = eps;
    vca[3] = HALF;
    vca[4] = ONE;

    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;
    *lmax = 0;
    *rmax = ZERO;

    f64 a[2 * 2], b[2 * 2], x[2 * 2];
    f64 d1, d2, ca, smin, wr, wi, scale, xnorm, res, den, tmp;
    int na, nw, info;

    for (int id1 = 0; id1 < 4; id1++) {
        d1 = vdd[id1];
        for (int id2 = 0; id2 < 4; id2++) {
            d2 = vdd[id2];
            for (int ica = 0; ica < 5; ica++) {
                ca = vca[ica];
                for (int itrans = 0; itrans <= 1; itrans++) {
                    for (int ismin = 0; ismin < 4; ismin++) {
                        smin = vsmin[ismin];

                        na = 1;
                        nw = 1;
                        for (int ia = 0; ia < 3; ia++) {
                            a[0] = vab[ia];
                            for (int ib = 0; ib < 3; ib++) {
                                b[0] = vab[ib];
                                for (int iwr = 0; iwr < 4; iwr++) {
                                    if (d1 == ONE && d2 == ONE && ca == ONE) {
                                        wr = vwr[iwr] * a[0];
                                    } else {
                                        wr = vwr[iwr];
                                    }
                                    wi = ZERO;
                                    dlaln2(itrans, na, nw,
                                           smin, ca, a, 2, d1, d2, b, 2,
                                           wr, wi, x, 2, &scale, &xnorm,
                                           &info);
                                    if (info < 0)
                                        ninfo[0] = ninfo[0] + 1;
                                    if (info > 0)
                                        ninfo[1] = ninfo[1] + 1;
                                    res = fabs((ca * a[0] - wr * d1) *
                                          x[0] - scale * b[0]);
                                    if (info == 0) {
                                        den = fabs((ca * a[0] - wr * d1) * x[0]);
                                        den = eps * den;
                                        if (smlnum > den) den = smlnum;
                                    } else {
                                        den = smin * fabs(x[0]);
                                        if (smlnum > den) den = smlnum;
                                    }
                                    res = res / den;
                                    if (fabs(x[0]) < unfl &&
                                        fabs(b[0]) <= smlnum *
                                        fabs(ca * a[0] - wr * d1))
                                        res = ZERO;
                                    if (scale > ONE)
                                        res = res + ONE / eps;
                                    res = res + fabs(xnorm - fabs(x[0])) /
                                          (smlnum > xnorm ? smlnum : xnorm) / eps;
                                    if (info != 0 && info != 1)
                                        res = res + ONE / eps;
                                    (*knt)++;
                                    if (res > *rmax) {
                                        *lmax = *knt;
                                        *rmax = res;
                                    }
                                }
                            }
                        }

                        na = 1;
                        nw = 2;
                        for (int ia = 0; ia < 3; ia++) {
                            a[0] = vab[ia];
                            for (int ib = 0; ib < 3; ib++) {
                                b[0] = vab[ib];
                                b[0 + 1 * 2] = -HALF * vab[ib];
                                for (int iwr = 0; iwr < 4; iwr++) {
                                    if (d1 == ONE && d2 == ONE && ca == ONE) {
                                        wr = vwr[iwr] * a[0];
                                    } else {
                                        wr = vwr[iwr];
                                    }
                                    for (int iwi = 0; iwi < 4; iwi++) {
                                        if (d1 == ONE && d2 == ONE &&
                                            ca == ONE) {
                                            wi = vwi[iwi] * a[0];
                                        } else {
                                            wi = vwi[iwi];
                                        }
                                        dlaln2(itrans, na, nw,
                                               smin, ca, a, 2, d1, d2, b,
                                               2, wr, wi, x, 2, &scale,
                                               &xnorm, &info);
                                        if (info < 0)
                                            ninfo[0] = ninfo[0] + 1;
                                        if (info > 0)
                                            ninfo[1] = ninfo[1] + 1;
                                        res = fabs((ca * a[0] - wr * d1) *
                                              x[0] + (wi * d1) * x[0 + 1 * 2] -
                                              scale * b[0]);
                                        res = res + fabs((-wi * d1) * x[0] +
                                              (ca * a[0] - wr * d1) * x[0 + 1 * 2] -
                                              scale * b[0 + 1 * 2]);
                                        if (info == 0) {
                                            f64 t1 = fabs(ca * a[0] - wr * d1);
                                            f64 t2 = fabs(d1 * wi);
                                            den = (t1 > t2 ? t1 : t2) *
                                                  (fabs(x[0]) + fabs(x[0 + 1 * 2]));
                                            den = eps * den;
                                            if (smlnum > den) den = smlnum;
                                        } else {
                                            den = smin * (fabs(x[0]) +
                                                  fabs(x[0 + 1 * 2]));
                                            if (smlnum > den) den = smlnum;
                                        }
                                        res = res / den;
                                        if (fabs(x[0]) < unfl &&
                                            fabs(x[0 + 1 * 2]) < unfl &&
                                            fabs(b[0]) <= smlnum *
                                            fabs(ca * a[0] - wr * d1))
                                            res = ZERO;
                                        if (scale > ONE)
                                            res = res + ONE / eps;
                                        res = res + fabs(xnorm -
                                              fabs(x[0]) -
                                              fabs(x[0 + 1 * 2])) /
                                              (smlnum > xnorm ? smlnum : xnorm) / eps;
                                        if (info != 0 && info != 1)
                                            res = res + ONE / eps;
                                        (*knt)++;
                                        if (res > *rmax) {
                                            *lmax = *knt;
                                            *rmax = res;
                                        }
                                    }
                                }
                            }
                        }

                        na = 2;
                        nw = 1;
                        for (int ia = 0; ia < 3; ia++) {
                            a[0]         = vab[ia];
                            a[0 + 1 * 2] = -THREE * vab[ia];
                            a[1]         = -SEVEN * vab[ia];
                            a[1 + 1 * 2] = TWNONE * vab[ia];
                            for (int ib = 0; ib < 3; ib++) {
                                b[0] = vab[ib];
                                b[1] = -TWO * vab[ib];
                                for (int iwr = 0; iwr < 4; iwr++) {
                                    if (d1 == ONE && d2 == ONE && ca == ONE) {
                                        wr = vwr[iwr] * a[0];
                                    } else {
                                        wr = vwr[iwr];
                                    }
                                    wi = ZERO;
                                    dlaln2(itrans, na, nw,
                                           smin, ca, a, 2, d1, d2, b, 2,
                                           wr, wi, x, 2, &scale, &xnorm,
                                           &info);
                                    if (info < 0)
                                        ninfo[0] = ninfo[0] + 1;
                                    if (info > 0)
                                        ninfo[1] = ninfo[1] + 1;
                                    if (itrans == 1) {
                                        tmp = a[0 + 1 * 2];
                                        a[0 + 1 * 2] = a[1];
                                        a[1] = tmp;
                                    }
                                    res = fabs((ca * a[0] - wr * d1) *
                                          x[0] + (ca * a[0 + 1 * 2]) *
                                          x[1] - scale * b[0]);
                                    res = res + fabs((ca * a[1]) *
                                          x[0] + (ca * a[1 + 1 * 2] - wr * d2) *
                                          x[1] - scale * b[1]);
                                    if (info == 0) {
                                        f64 r1 = fabs(ca * a[0] - wr * d1) +
                                                  fabs(ca * a[0 + 1 * 2]);
                                        f64 r2 = fabs(ca * a[1]) +
                                                  fabs(ca * a[1 + 1 * 2] - wr * d2);
                                        den = (r1 > r2 ? r1 : r2) *
                                              (fabs(x[0]) > fabs(x[1]) ? fabs(x[0]) : fabs(x[1]));
                                        den = eps * den;
                                        if (smlnum > den) den = smlnum;
                                    } else {
                                        f64 r1 = fabs(ca * a[0] - wr * d1) +
                                                  fabs(ca * a[0 + 1 * 2]);
                                        f64 r2 = fabs(ca * a[1]) +
                                                  fabs(ca * a[1 + 1 * 2] - wr * d2);
                                        f64 rr = (r1 > r2 ? r1 : r2);
                                        f64 se = smin / eps;
                                        den = (se > rr ? se : rr) *
                                              (fabs(x[0]) > fabs(x[1]) ? fabs(x[0]) : fabs(x[1]));
                                        den = eps * den;
                                        if (smlnum > den) den = smlnum;
                                    }
                                    res = res / den;
                                    if (fabs(x[0]) < unfl &&
                                        fabs(x[1]) < unfl &&
                                        fabs(b[0]) + fabs(b[1]) <=
                                        smlnum * (fabs(ca * a[0] - wr * d1) +
                                        fabs(ca * a[0 + 1 * 2]) +
                                        fabs(ca * a[1]) +
                                        fabs(ca * a[1 + 1 * 2] - wr * d2)))
                                        res = ZERO;
                                    if (scale > ONE)
                                        res = res + ONE / eps;
                                    {
                                        f64 mx = fabs(x[0]) > fabs(x[1]) ? fabs(x[0]) : fabs(x[1]);
                                        res = res + fabs(xnorm - mx) /
                                              (smlnum > xnorm ? smlnum : xnorm) / eps;
                                    }
                                    if (info != 0 && info != 1)
                                        res = res + ONE / eps;
                                    (*knt)++;
                                    if (res > *rmax) {
                                        *lmax = *knt;
                                        *rmax = res;
                                    }
                                }
                            }
                        }

                        na = 2;
                        nw = 2;
                        for (int ia = 0; ia < 3; ia++) {
                            a[0]         = TWO * vab[ia];
                            a[0 + 1 * 2] = -THREE * vab[ia];
                            a[1]         = -SEVEN * vab[ia];
                            a[1 + 1 * 2] = TWNONE * vab[ia];
                            for (int ib = 0; ib < 3; ib++) {
                                b[0]         = vab[ib];
                                b[1]         = -TWO * vab[ib];
                                b[0 + 1 * 2] = FOUR * vab[ib];
                                b[1 + 1 * 2] = -SEVEN * vab[ib];
                                for (int iwr = 0; iwr < 4; iwr++) {
                                    if (d1 == ONE && d2 == ONE && ca == ONE) {
                                        wr = vwr[iwr] * a[0];
                                    } else {
                                        wr = vwr[iwr];
                                    }
                                    for (int iwi = 0; iwi < 4; iwi++) {
                                        if (d1 == ONE && d2 == ONE &&
                                            ca == ONE) {
                                            wi = vwi[iwi] * a[0];
                                        } else {
                                            wi = vwi[iwi];
                                        }
                                        dlaln2(itrans, na, nw,
                                               smin, ca, a, 2, d1, d2, b,
                                               2, wr, wi, x, 2, &scale,
                                               &xnorm, &info);
                                        if (info < 0)
                                            ninfo[0] = ninfo[0] + 1;
                                        if (info > 0)
                                            ninfo[1] = ninfo[1] + 1;
                                        if (itrans == 1) {
                                            tmp = a[0 + 1 * 2];
                                            a[0 + 1 * 2] = a[1];
                                            a[1] = tmp;
                                        }
                                        res = fabs((ca * a[0] - wr * d1) *
                                              x[0] + (ca * a[0 + 1 * 2]) *
                                              x[1] + (wi * d1) * x[0 + 1 * 2] -
                                              scale * b[0]);
                                        res = res + fabs((ca * a[0] - wr * d1) *
                                              x[0 + 1 * 2] +
                                              (ca * a[0 + 1 * 2]) * x[1 + 1 * 2] -
                                              (wi * d1) * x[0] - scale *
                                              b[0 + 1 * 2]);
                                        res = res + fabs((ca * a[1]) *
                                              x[0] + (ca * a[1 + 1 * 2] - wr * d2) *
                                              x[1] + (wi * d2) * x[1 + 1 * 2] -
                                              scale * b[1]);
                                        res = res + fabs((ca * a[1]) *
                                              x[0 + 1 * 2] +
                                              (ca * a[1 + 1 * 2] - wr * d2) *
                                              x[1 + 1 * 2] - (wi * d2) * x[1] -
                                              scale * b[1 + 1 * 2]);
                                        if (info == 0) {
                                            f64 r1 = fabs(ca * a[0] - wr * d1) +
                                                      fabs(ca * a[0 + 1 * 2]) +
                                                      fabs(wi * d1);
                                            f64 r2 = fabs(ca * a[1]) +
                                                      fabs(ca * a[1 + 1 * 2] - wr * d2) +
                                                      fabs(wi * d2);
                                            f64 c1 = fabs(x[0]) + fabs(x[1]);
                                            f64 c2 = fabs(x[0 + 1 * 2]) + fabs(x[1 + 1 * 2]);
                                            den = (r1 > r2 ? r1 : r2) *
                                                  (c1 > c2 ? c1 : c2);
                                            den = eps * den;
                                            if (smlnum > den) den = smlnum;
                                        } else {
                                            f64 r1 = fabs(ca * a[0] - wr * d1) +
                                                      fabs(ca * a[0 + 1 * 2]) +
                                                      fabs(wi * d1);
                                            f64 r2 = fabs(ca * a[1]) +
                                                      fabs(ca * a[1 + 1 * 2] - wr * d2) +
                                                      fabs(wi * d2);
                                            f64 rr = (r1 > r2 ? r1 : r2);
                                            f64 se = smin / eps;
                                            f64 c1 = fabs(x[0]) + fabs(x[1]);
                                            f64 c2 = fabs(x[0 + 1 * 2]) + fabs(x[1 + 1 * 2]);
                                            den = (se > rr ? se : rr) *
                                                  (c1 > c2 ? c1 : c2);
                                            den = eps * den;
                                            if (smlnum > den) den = smlnum;
                                        }
                                        res = res / den;
                                        if (fabs(x[0]) < unfl &&
                                            fabs(x[1]) < unfl &&
                                            fabs(x[0 + 1 * 2]) < unfl &&
                                            fabs(x[1 + 1 * 2]) < unfl &&
                                            fabs(b[0]) +
                                            fabs(b[1]) <= smlnum *
                                            (fabs(ca * a[0] - wr * d1) +
                                             fabs(ca * a[0 + 1 * 2]) +
                                             fabs(ca * a[1]) +
                                             fabs(ca * a[1 + 1 * 2] - wr * d2) +
                                             fabs(wi * d2) + fabs(wi * d1)))
                                            res = ZERO;
                                        if (scale > ONE)
                                            res = res + ONE / eps;
                                        {
                                            f64 c1 = fabs(x[0]) + fabs(x[0 + 1 * 2]);
                                            f64 c2 = fabs(x[1]) + fabs(x[1 + 1 * 2]);
                                            res = res + fabs(xnorm -
                                                  (c1 > c2 ? c1 : c2)) /
                                                  (smlnum > xnorm ? smlnum : xnorm) / eps;
                                        }
                                        if (info != 0 && info != 1)
                                            res = res + ONE / eps;
                                        (*knt)++;
                                        if (res > *rmax) {
                                            *lmax = *knt;
                                            *rmax = res;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
