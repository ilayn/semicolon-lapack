/**
 * @file dget32.c
 * @brief DGET32 tests DLASY2, a routine for solving
 *        op(TL)*X + ISGN*X*op(TR) = SCALE*B.
 */

#include "verify.h"
#include <math.h>

extern f64 dlamch(const char* cmach);
extern void dlasy2(const int ltranl, const int ltranr, const int isgn,
                   const int n1, const int n2,
                   const f64* TL, const int ldtl,
                   const f64* TR, const int ldtr,
                   const f64* B, const int ldb,
                   f64* scale, f64* X, const int ldx,
                   f64* xnorm, int* info);

/**
 * DGET32 tests DLASY2, a routine for solving
 *
 *    op(TL)*X + ISGN*X*op(TR) = SCALE*B
 *
 * where TL is N1 by N1, TR is N2 by N2, and N1,N2 =1 or 2 only.
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Number of examples returned with INFO != 0.
 * @param[out]    knt     Total number of examples tested.
 */
void dget32(f64* rmax, int* lmax, int* ninfo, int* knt)
{
    const f64 ZERO  = 0.0;
    const f64 ONE   = 1.0;
    const f64 TWO   = 2.0;
    const f64 FOUR  = 4.0;
    const f64 EIGHT = 8.0;

    /* ITVAL(2,2,8) stored as itval[k][j][i] for C row-major access,
       but we index as itval[k][i][j] matching Fortran ITVAL(i,j,k) */
    const int itval[8][2][2] = {
        {{8, 2}, {4, 1}},
        {{4, 1}, {8, 2}},
        {{2, 8}, {1, 4}},
        {{1, 4}, {2, 8}},
        {{9, 2}, {4, 1}},
        {{4, 1}, {9, 2}},
        {{2, 9}, {1, 4}},
        {{1, 4}, {2, 9}},
    };

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    f64 val[3];
    val[0] = sqrt(smlnum);
    val[1] = ONE;
    val[2] = sqrt(bignum);

    *knt = 0;
    *ninfo = 0;
    *lmax = 0;
    *rmax = ZERO;

    f64 tl[2 * 2], tr[2 * 2], b[2 * 2], x[2 * 2];
    f64 scale, xnorm, xnrm, tnrm, res, den, sgn, tmp;
    int n1, n2, info;

    for (int itranl = 0; itranl <= 1; itranl++) {
        for (int itranr = 0; itranr <= 1; itranr++) {
            for (int isgn = -1; isgn <= 1; isgn += 2) {
                sgn = (f64)isgn;
                int ltranl = (itranl == 1);
                int ltranr = (itranr == 1);

                n1 = 1;
                n2 = 1;
                for (int itl = 0; itl < 3; itl++) {
                    for (int itr = 0; itr < 3; itr++) {
                        for (int ib = 0; ib < 3; ib++) {
                            tl[0] = val[itl];
                            tr[0] = val[itr];
                            b[0] = val[ib];
                            (*knt)++;
                            dlasy2(ltranl, ltranr, isgn, n1, n2, tl,
                                   2, tr, 2, b, 2, &scale, x, 2, &xnorm,
                                   &info);
                            if (info != 0)
                                (*ninfo)++;
                            res = fabs((tl[0] + sgn * tr[0]) *
                                  x[0] - scale * b[0]);
                            if (info == 0) {
                                den = eps * ((fabs(tr[0]) +
                                      fabs(tl[0])) * fabs(x[0]));
                                if (smlnum > den) den = smlnum;
                            } else {
                                den = smlnum * (fabs(x[0]) > ONE ? fabs(x[0]) : ONE);
                            }
                            res = res / den;
                            if (scale > ONE)
                                res = res + ONE / eps;
                            res = res + fabs(xnorm - fabs(x[0])) /
                                  (smlnum > xnorm ? smlnum : xnorm) / eps;
                            if (info != 0 && info != 1)
                                res = res + ONE / eps;
                            if (res > *rmax) {
                                *lmax = *knt;
                                *rmax = res;
                            }
                        }
                    }
                }

                n1 = 2;
                n2 = 1;
                for (int itl = 0; itl < 8; itl++) {
                    for (int itlscl = 0; itlscl < 3; itlscl++) {
                        for (int itr = 0; itr < 3; itr++) {
                            for (int ib1 = 0; ib1 < 3; ib1++) {
                                for (int ib2 = 0; ib2 < 3; ib2++) {
                                    b[0] = val[ib1];
                                    b[1] = -FOUR * val[ib2];
                                    tl[0]         = itval[itl][0][0] * val[itlscl];
                                    tl[1]         = itval[itl][1][0] * val[itlscl];
                                    tl[0 + 1 * 2] = itval[itl][0][1] * val[itlscl];
                                    tl[1 + 1 * 2] = itval[itl][1][1] * val[itlscl];
                                    tr[0] = val[itr];
                                    (*knt)++;
                                    dlasy2(ltranl, ltranr, isgn, n1, n2,
                                           tl, 2, tr, 2, b, 2, &scale, x,
                                           2, &xnorm, &info);
                                    if (info != 0)
                                        (*ninfo)++;
                                    if (ltranl) {
                                        tmp = tl[0 + 1 * 2];
                                        tl[0 + 1 * 2] = tl[1];
                                        tl[1] = tmp;
                                    }
                                    res = fabs((tl[0] + sgn * tr[0]) *
                                          x[0] + tl[0 + 1 * 2] * x[1] -
                                          scale * b[0]);
                                    res = res + fabs((tl[1 + 1 * 2] + sgn * tr[0]) *
                                          x[1] + tl[1] *
                                          x[0] - scale * b[1]);
                                    tnrm = fabs(tr[0]) +
                                           fabs(tl[0]) +
                                           fabs(tl[0 + 1 * 2]) +
                                           fabs(tl[1]) +
                                           fabs(tl[1 + 1 * 2]);
                                    xnrm = fabs(x[0]) > fabs(x[1]) ? fabs(x[0]) : fabs(x[1]);
                                    den = smlnum;
                                    if (smlnum * xnrm > den) den = smlnum * xnrm;
                                    if ((tnrm * eps) * xnrm > den) den = (tnrm * eps) * xnrm;
                                    res = res / den;
                                    if (scale > ONE)
                                        res = res + ONE / eps;
                                    res = res + fabs(xnorm - xnrm) /
                                          (smlnum > xnorm ? smlnum : xnorm) / eps;
                                    if (res > *rmax) {
                                        *lmax = *knt;
                                        *rmax = res;
                                    }
                                }
                            }
                        }
                    }
                }

                n1 = 1;
                n2 = 2;
                for (int itr = 0; itr < 8; itr++) {
                    for (int itrscl = 0; itrscl < 3; itrscl++) {
                        for (int itl = 0; itl < 3; itl++) {
                            for (int ib1 = 0; ib1 < 3; ib1++) {
                                for (int ib2 = 0; ib2 < 3; ib2++) {
                                    b[0] = val[ib1];
                                    b[0 + 1 * 2] = -TWO * val[ib2];
                                    tr[0]         = itval[itr][0][0] * val[itrscl];
                                    tr[1]         = itval[itr][1][0] * val[itrscl];
                                    tr[0 + 1 * 2] = itval[itr][0][1] * val[itrscl];
                                    tr[1 + 1 * 2] = itval[itr][1][1] * val[itrscl];
                                    tl[0] = val[itl];
                                    (*knt)++;
                                    dlasy2(ltranl, ltranr, isgn, n1, n2,
                                           tl, 2, tr, 2, b, 2, &scale, x,
                                           2, &xnorm, &info);
                                    if (info != 0)
                                        (*ninfo)++;
                                    if (ltranr) {
                                        tmp = tr[0 + 1 * 2];
                                        tr[0 + 1 * 2] = tr[1];
                                        tr[1] = tmp;
                                    }
                                    tnrm = fabs(tl[0]) +
                                           fabs(tr[0]) +
                                           fabs(tr[0 + 1 * 2]) +
                                           fabs(tr[1 + 1 * 2]) +
                                           fabs(tr[1]);
                                    xnrm = fabs(x[0]) + fabs(x[0 + 1 * 2]);
                                    res = fabs((tl[0] + sgn * tr[0]) *
                                          x[0] +
                                          (sgn * tr[1]) * x[0 + 1 * 2] -
                                          scale * b[0]);
                                    res = res + fabs((tl[0] + sgn * tr[1 + 1 * 2]) *
                                          x[0 + 1 * 2] +
                                          (sgn * tr[0 + 1 * 2]) * x[0] -
                                          scale * b[0 + 1 * 2]);
                                    den = smlnum;
                                    if (smlnum * xnrm > den) den = smlnum * xnrm;
                                    if ((tnrm * eps) * xnrm > den) den = (tnrm * eps) * xnrm;
                                    res = res / den;
                                    if (scale > ONE)
                                        res = res + ONE / eps;
                                    res = res + fabs(xnorm - xnrm) /
                                          (smlnum > xnorm ? smlnum : xnorm) / eps;
                                    if (res > *rmax) {
                                        *lmax = *knt;
                                        *rmax = res;
                                    }
                                }
                            }
                        }
                    }
                }

                n1 = 2;
                n2 = 2;
                for (int itr = 0; itr < 8; itr++) {
                    for (int itrscl = 0; itrscl < 3; itrscl++) {
                        for (int itl = 0; itl < 8; itl++) {
                            for (int itlscl = 0; itlscl < 3; itlscl++) {
                                for (int ib1 = 0; ib1 < 3; ib1++) {
                                    for (int ib2 = 0; ib2 < 3; ib2++) {
                                        for (int ib3 = 0; ib3 < 3; ib3++) {
                                            b[0]         = val[ib1];
                                            b[1]         = -FOUR * val[ib2];
                                            b[0 + 1 * 2] = -TWO * val[ib3];
                                            {
                                                f64 mv = val[ib1];
                                                if (val[ib2] < mv) mv = val[ib2];
                                                if (val[ib3] < mv) mv = val[ib3];
                                                b[1 + 1 * 2] = EIGHT * mv;
                                            }
                                            tr[0]         = itval[itr][0][0] * val[itrscl];
                                            tr[1]         = itval[itr][1][0] * val[itrscl];
                                            tr[0 + 1 * 2] = itval[itr][0][1] * val[itrscl];
                                            tr[1 + 1 * 2] = itval[itr][1][1] * val[itrscl];
                                            tl[0]         = itval[itl][0][0] * val[itlscl];
                                            tl[1]         = itval[itl][1][0] * val[itlscl];
                                            tl[0 + 1 * 2] = itval[itl][0][1] * val[itlscl];
                                            tl[1 + 1 * 2] = itval[itl][1][1] * val[itlscl];
                                            (*knt)++;
                                            dlasy2(ltranl, ltranr, isgn,
                                                   n1, n2, tl, 2, tr, 2,
                                                   b, 2, &scale, x, 2,
                                                   &xnorm, &info);
                                            if (info != 0)
                                                (*ninfo)++;
                                            if (ltranr) {
                                                tmp = tr[0 + 1 * 2];
                                                tr[0 + 1 * 2] = tr[1];
                                                tr[1] = tmp;
                                            }
                                            if (ltranl) {
                                                tmp = tl[0 + 1 * 2];
                                                tl[0 + 1 * 2] = tl[1];
                                                tl[1] = tmp;
                                            }
                                            tnrm = fabs(tr[0]) +
                                                   fabs(tr[1]) +
                                                   fabs(tr[0 + 1 * 2]) +
                                                   fabs(tr[1 + 1 * 2]) +
                                                   fabs(tl[0]) +
                                                   fabs(tl[1]) +
                                                   fabs(tl[0 + 1 * 2]) +
                                                   fabs(tl[1 + 1 * 2]);
                                            {
                                                f64 r1 = fabs(x[0]) + fabs(x[0 + 1 * 2]);
                                                f64 r2 = fabs(x[1]) + fabs(x[1 + 1 * 2]);
                                                xnrm = r1 > r2 ? r1 : r2;
                                            }
                                            res = fabs((tl[0] + sgn * tr[0]) *
                                                  x[0] +
                                                  (sgn * tr[1]) *
                                                  x[0 + 1 * 2] + tl[0 + 1 * 2] *
                                                  x[1] -
                                                  scale * b[0]);
                                            res = res + fabs(tl[0] *
                                                  x[0 + 1 * 2] +
                                                  (sgn * tr[0 + 1 * 2]) *
                                                  x[0] +
                                                  (sgn * tr[1 + 1 * 2]) *
                                                  x[0 + 1 * 2] + tl[0 + 1 * 2] *
                                                  x[1 + 1 * 2] -
                                                  scale * b[0 + 1 * 2]);
                                            res = res + fabs(tl[1] *
                                                  x[0] +
                                                  (sgn * tr[0]) *
                                                  x[1] +
                                                  (sgn * tr[1]) *
                                                  x[1 + 1 * 2] + tl[1 + 1 * 2] *
                                                  x[1] -
                                                  scale * b[1]);
                                            res = res + fabs((tl[1 + 1 * 2] + sgn * tr[1 + 1 * 2]) *
                                                  x[1 + 1 * 2] +
                                                  (sgn * tr[0 + 1 * 2]) *
                                                  x[1] + tl[1] *
                                                  x[0 + 1 * 2] -
                                                  scale * b[1 + 1 * 2]);
                                            den = smlnum;
                                            if (smlnum * xnrm > den) den = smlnum * xnrm;
                                            if ((tnrm * eps) * xnrm > den) den = (tnrm * eps) * xnrm;
                                            res = res / den;
                                            if (scale > ONE)
                                                res = res + ONE / eps;
                                            res = res + fabs(xnorm - xnrm) /
                                                  (smlnum > xnorm ? smlnum : xnorm) / eps;
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
}
