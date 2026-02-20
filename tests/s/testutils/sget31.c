/**
 * @file sget31.c
 * @brief SGET31 tests SLALN2, a routine for solving
 *        (ca A - w D)X = sB where A is 1x1 or 2x2.
 */

#include "verify.h"
#include <math.h>

extern f32 slamch(const char* cmach);
extern void slaln2(const int ltrans, const int na, const int nw,
                   const f32 smin, const f32 ca, const f32* A, const int lda,
                   const f32 d1, const f32 d2, const f32* B, const int ldb,
                   const f32 wr, const f32 wi, f32* X, const int ldx,
                   f32* scale, f32* xnorm, int* info);

/**
 * SGET31 tests SLALN2, a routine for solving
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
void sget31(f32* rmax, int* lmax, int ninfo[2], int* knt)
{
    const f32 ZERO  = 0.0f;
    const f32 HALF  = 0.5f;
    const f32 ONE   = 1.0f;
    const f32 TWO   = 2.0f;
    const f32 THREE = 3.0f;
    const f32 FOUR  = 4.0f;
    const f32 SEVEN = 7.0f;
    const f32 TEN   = 10.0f;
    const f32 TWNONE = 21.0f;

    f32 eps = slamch("P");
    f32 unfl = slamch("U");
    f32 smlnum = slamch("S") / eps;
    f32 bignum = ONE / smlnum;

    f32 vsmin[4], vab[3], vwr[4], vwi[4], vdd[4], vca[5];
    vsmin[0] = smlnum;
    vsmin[1] = eps;
    vsmin[2] = ONE / (TEN * TEN);
    vsmin[3] = ONE / eps;
    vab[0] = sqrtf(smlnum);
    vab[1] = ONE;
    vab[2] = sqrtf(bignum);
    vwr[0] = ZERO;
    vwr[1] = HALF;
    vwr[2] = TWO;
    vwr[3] = ONE;
    vwi[0] = smlnum;
    vwi[1] = eps;
    vwi[2] = ONE;
    vwi[3] = TWO;
    vdd[0] = sqrtf(smlnum);
    vdd[1] = ONE;
    vdd[2] = TWO;
    vdd[3] = sqrtf(bignum);
    vca[0] = ZERO;
    vca[1] = sqrtf(smlnum);
    vca[2] = eps;
    vca[3] = HALF;
    vca[4] = ONE;

    *knt = 0;
    ninfo[0] = 0;
    ninfo[1] = 0;
    *lmax = 0;
    *rmax = ZERO;

    f32 a[2 * 2], b[2 * 2], x[2 * 2];
    f32 d1, d2, ca, smin, wr, wi, scale, xnorm, res, den, tmp;
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
                                    slaln2(itrans, na, nw,
                                           smin, ca, a, 2, d1, d2, b, 2,
                                           wr, wi, x, 2, &scale, &xnorm,
                                           &info);
                                    if (info < 0)
                                        ninfo[0] = ninfo[0] + 1;
                                    if (info > 0)
                                        ninfo[1] = ninfo[1] + 1;
                                    res = fabsf((ca * a[0] - wr * d1) *
                                          x[0] - scale * b[0]);
                                    if (info == 0) {
                                        den = fabsf((ca * a[0] - wr * d1) * x[0]);
                                        den = eps * den;
                                        if (smlnum > den) den = smlnum;
                                    } else {
                                        den = smin * fabsf(x[0]);
                                        if (smlnum > den) den = smlnum;
                                    }
                                    res = res / den;
                                    if (fabsf(x[0]) < unfl &&
                                        fabsf(b[0]) <= smlnum *
                                        fabsf(ca * a[0] - wr * d1))
                                        res = ZERO;
                                    if (scale > ONE)
                                        res = res + ONE / eps;
                                    res = res + fabsf(xnorm - fabsf(x[0])) /
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
                                        slaln2(itrans, na, nw,
                                               smin, ca, a, 2, d1, d2, b,
                                               2, wr, wi, x, 2, &scale,
                                               &xnorm, &info);
                                        if (info < 0)
                                            ninfo[0] = ninfo[0] + 1;
                                        if (info > 0)
                                            ninfo[1] = ninfo[1] + 1;
                                        res = fabsf((ca * a[0] - wr * d1) *
                                              x[0] + (wi * d1) * x[0 + 1 * 2] -
                                              scale * b[0]);
                                        res = res + fabsf((-wi * d1) * x[0] +
                                              (ca * a[0] - wr * d1) * x[0 + 1 * 2] -
                                              scale * b[0 + 1 * 2]);
                                        if (info == 0) {
                                            f32 t1 = fabsf(ca * a[0] - wr * d1);
                                            f32 t2 = fabsf(d1 * wi);
                                            den = (t1 > t2 ? t1 : t2) *
                                                  (fabsf(x[0]) + fabsf(x[0 + 1 * 2]));
                                            den = eps * den;
                                            if (smlnum > den) den = smlnum;
                                        } else {
                                            den = smin * (fabsf(x[0]) +
                                                  fabsf(x[0 + 1 * 2]));
                                            if (smlnum > den) den = smlnum;
                                        }
                                        res = res / den;
                                        if (fabsf(x[0]) < unfl &&
                                            fabsf(x[0 + 1 * 2]) < unfl &&
                                            fabsf(b[0]) <= smlnum *
                                            fabsf(ca * a[0] - wr * d1))
                                            res = ZERO;
                                        if (scale > ONE)
                                            res = res + ONE / eps;
                                        res = res + fabsf(xnorm -
                                              fabsf(x[0]) -
                                              fabsf(x[0 + 1 * 2])) /
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
                                    slaln2(itrans, na, nw,
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
                                    res = fabsf((ca * a[0] - wr * d1) *
                                          x[0] + (ca * a[0 + 1 * 2]) *
                                          x[1] - scale * b[0]);
                                    res = res + fabsf((ca * a[1]) *
                                          x[0] + (ca * a[1 + 1 * 2] - wr * d2) *
                                          x[1] - scale * b[1]);
                                    if (info == 0) {
                                        f32 r1 = fabsf(ca * a[0] - wr * d1) +
                                                  fabsf(ca * a[0 + 1 * 2]);
                                        f32 r2 = fabsf(ca * a[1]) +
                                                  fabsf(ca * a[1 + 1 * 2] - wr * d2);
                                        den = (r1 > r2 ? r1 : r2) *
                                              (fabsf(x[0]) > fabsf(x[1]) ? fabsf(x[0]) : fabsf(x[1]));
                                        den = eps * den;
                                        if (smlnum > den) den = smlnum;
                                    } else {
                                        f32 r1 = fabsf(ca * a[0] - wr * d1) +
                                                  fabsf(ca * a[0 + 1 * 2]);
                                        f32 r2 = fabsf(ca * a[1]) +
                                                  fabsf(ca * a[1 + 1 * 2] - wr * d2);
                                        f32 rr = (r1 > r2 ? r1 : r2);
                                        f32 se = smin / eps;
                                        den = (se > rr ? se : rr) *
                                              (fabsf(x[0]) > fabsf(x[1]) ? fabsf(x[0]) : fabsf(x[1]));
                                        den = eps * den;
                                        if (smlnum > den) den = smlnum;
                                    }
                                    res = res / den;
                                    if (fabsf(x[0]) < unfl &&
                                        fabsf(x[1]) < unfl &&
                                        fabsf(b[0]) + fabsf(b[1]) <=
                                        smlnum * (fabsf(ca * a[0] - wr * d1) +
                                        fabsf(ca * a[0 + 1 * 2]) +
                                        fabsf(ca * a[1]) +
                                        fabsf(ca * a[1 + 1 * 2] - wr * d2)))
                                        res = ZERO;
                                    if (scale > ONE)
                                        res = res + ONE / eps;
                                    {
                                        f32 mx = fabsf(x[0]) > fabsf(x[1]) ? fabsf(x[0]) : fabsf(x[1]);
                                        res = res + fabsf(xnorm - mx) /
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
                                        slaln2(itrans, na, nw,
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
                                        res = fabsf((ca * a[0] - wr * d1) *
                                              x[0] + (ca * a[0 + 1 * 2]) *
                                              x[1] + (wi * d1) * x[0 + 1 * 2] -
                                              scale * b[0]);
                                        res = res + fabsf((ca * a[0] - wr * d1) *
                                              x[0 + 1 * 2] +
                                              (ca * a[0 + 1 * 2]) * x[1 + 1 * 2] -
                                              (wi * d1) * x[0] - scale *
                                              b[0 + 1 * 2]);
                                        res = res + fabsf((ca * a[1]) *
                                              x[0] + (ca * a[1 + 1 * 2] - wr * d2) *
                                              x[1] + (wi * d2) * x[1 + 1 * 2] -
                                              scale * b[1]);
                                        res = res + fabsf((ca * a[1]) *
                                              x[0 + 1 * 2] +
                                              (ca * a[1 + 1 * 2] - wr * d2) *
                                              x[1 + 1 * 2] - (wi * d2) * x[1] -
                                              scale * b[1 + 1 * 2]);
                                        if (info == 0) {
                                            f32 r1 = fabsf(ca * a[0] - wr * d1) +
                                                      fabsf(ca * a[0 + 1 * 2]) +
                                                      fabsf(wi * d1);
                                            f32 r2 = fabsf(ca * a[1]) +
                                                      fabsf(ca * a[1 + 1 * 2] - wr * d2) +
                                                      fabsf(wi * d2);
                                            f32 c1 = fabsf(x[0]) + fabsf(x[1]);
                                            f32 c2 = fabsf(x[0 + 1 * 2]) + fabsf(x[1 + 1 * 2]);
                                            den = (r1 > r2 ? r1 : r2) *
                                                  (c1 > c2 ? c1 : c2);
                                            den = eps * den;
                                            if (smlnum > den) den = smlnum;
                                        } else {
                                            f32 r1 = fabsf(ca * a[0] - wr * d1) +
                                                      fabsf(ca * a[0 + 1 * 2]) +
                                                      fabsf(wi * d1);
                                            f32 r2 = fabsf(ca * a[1]) +
                                                      fabsf(ca * a[1 + 1 * 2] - wr * d2) +
                                                      fabsf(wi * d2);
                                            f32 rr = (r1 > r2 ? r1 : r2);
                                            f32 se = smin / eps;
                                            f32 c1 = fabsf(x[0]) + fabsf(x[1]);
                                            f32 c2 = fabsf(x[0 + 1 * 2]) + fabsf(x[1 + 1 * 2]);
                                            den = (se > rr ? se : rr) *
                                                  (c1 > c2 ? c1 : c2);
                                            den = eps * den;
                                            if (smlnum > den) den = smlnum;
                                        }
                                        res = res / den;
                                        if (fabsf(x[0]) < unfl &&
                                            fabsf(x[1]) < unfl &&
                                            fabsf(x[0 + 1 * 2]) < unfl &&
                                            fabsf(x[1 + 1 * 2]) < unfl &&
                                            fabsf(b[0]) +
                                            fabsf(b[1]) <= smlnum *
                                            (fabsf(ca * a[0] - wr * d1) +
                                             fabsf(ca * a[0 + 1 * 2]) +
                                             fabsf(ca * a[1]) +
                                             fabsf(ca * a[1 + 1 * 2] - wr * d2) +
                                             fabsf(wi * d2) + fabsf(wi * d1)))
                                            res = ZERO;
                                        if (scale > ONE)
                                            res = res + ONE / eps;
                                        {
                                            f32 c1 = fabsf(x[0]) + fabsf(x[0 + 1 * 2]);
                                            f32 c2 = fabsf(x[1]) + fabsf(x[1 + 1 * 2]);
                                            res = res + fabsf(xnorm -
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
