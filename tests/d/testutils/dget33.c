/**
 * @file dget33.c
 * @brief DGET33 tests DLANV2, a routine for putting 2 by 2 blocks into
 *        standard form.
 */

#include "verify.h"
#include <math.h>

/**
 * DGET33 tests DLANV2, a routine for putting 2 by 2 blocks into
 * standard form.  In other words, it computes a two by two rotation
 * [[C,S];[-S,C]] where in
 *
 *    [ C S ][T(1,1) T(1,2)][ C -S ] = [ T11 T12 ]
 *    [-S C ][T(2,1) T(2,2)][ S  C ]   [ T21 T22 ]
 *
 * either
 *    1) T21=0 (real eigenvalues), or
 *    2) T11=T22 and T21*T12<0 (complex conjugate eigenvalues).
 * We also verify that the residual is small.
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Number of examples returned with INFO != 0.
 * @param[out]    knt     Total number of examples tested.
 */
void dget33(f64* rmax, INT* lmax, INT* ninfo, INT* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE  = 1.0;
    const f64 TWO  = 2.0;
    const f64 FOUR = 4.0;

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    f64 val[4], vm[3];
    val[0] = ONE;
    val[1] = ONE + TWO * eps;
    val[2] = TWO;
    val[3] = TWO - FOUR * eps;
    vm[0] = smlnum;
    vm[1] = ONE;
    vm[2] = bignum;

    *knt = 0;
    *ninfo = 0;
    *lmax = 0;
    *rmax = ZERO;

    f64 t[2][2], t1[2][2], t2[2][2], q[2][2];
    f64 cs, sn, wr1, wi1, wr2, wi2, res, sum, tnrm;

    for (INT i1 = 0; i1 < 4; i1++) {
        for (INT i2 = 0; i2 < 4; i2++) {
            for (INT i3 = 0; i3 < 4; i3++) {
                for (INT i4 = 0; i4 < 4; i4++) {
                    for (INT im1 = 0; im1 < 3; im1++) {
                        for (INT im2 = 0; im2 < 3; im2++) {
                            for (INT im3 = 0; im3 < 3; im3++) {
                                for (INT im4 = 0; im4 < 3; im4++) {
                                    t[0][0] = val[i1] * vm[im1];
                                    t[0][1] = val[i2] * vm[im2];
                                    t[1][0] = -val[i3] * vm[im3];
                                    t[1][1] = val[i4] * vm[im4];
                                    tnrm = fabs(t[0][0]);
                                    if (fabs(t[0][1]) > tnrm) tnrm = fabs(t[0][1]);
                                    if (fabs(t[1][0]) > tnrm) tnrm = fabs(t[1][0]);
                                    if (fabs(t[1][1]) > tnrm) tnrm = fabs(t[1][1]);
                                    t1[0][0] = t[0][0];
                                    t1[0][1] = t[0][1];
                                    t1[1][0] = t[1][0];
                                    t1[1][1] = t[1][1];
                                    q[0][0] = ONE;
                                    q[0][1] = ZERO;
                                    q[1][0] = ZERO;
                                    q[1][1] = ONE;

                                    dlanv2(&t[0][0], &t[0][1],
                                           &t[1][0], &t[1][1],
                                           &wr1, &wi1, &wr2, &wi2,
                                           &cs, &sn);
                                    for (INT j1 = 0; j1 < 2; j1++) {
                                        res = q[j1][0] * cs + q[j1][1] * sn;
                                        q[j1][1] = -q[j1][0] * sn +
                                                    q[j1][1] * cs;
                                        q[j1][0] = res;
                                    }

                                    res = ZERO;
                                    res = res + fabs(q[0][0] * q[0][0] +
                                                q[0][1] * q[0][1] - ONE) / eps;
                                    res = res + fabs(q[1][1] * q[1][1] +
                                                q[1][0] * q[1][0] - ONE) / eps;
                                    res = res + fabs(q[0][0] * q[1][0] +
                                                q[0][1] * q[1][1]) / eps;
                                    for (INT j1 = 0; j1 < 2; j1++) {
                                        for (INT j2 = 0; j2 < 2; j2++) {
                                            t2[j1][j2] = ZERO;
                                            for (INT j3 = 0; j3 < 2; j3++) {
                                                t2[j1][j2] = t2[j1][j2] +
                                                              t1[j1][j3] *
                                                              q[j3][j2];
                                            }
                                        }
                                    }
                                    for (INT j1 = 0; j1 < 2; j1++) {
                                        for (INT j2 = 0; j2 < 2; j2++) {
                                            sum = t[j1][j2];
                                            for (INT j3 = 0; j3 < 2; j3++) {
                                                sum = sum - q[j3][j1] *
                                                            t2[j3][j2];
                                            }
                                            res = res + fabs(sum) / eps / tnrm;
                                        }
                                    }
                                    if (t[1][0] != ZERO &&
                                        (t[0][0] != t[1][1] ||
                                         copysign(ONE, t[0][1]) *
                                         copysign(ONE, t[1][0]) > ZERO))
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
