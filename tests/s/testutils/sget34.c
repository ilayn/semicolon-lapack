/**
 * @file sget34.c
 * @brief SGET34 tests SLAEXC, a routine for swapping adjacent blocks
 *        (either 1 by 1 or 2 by 2) on the diagonal of a matrix in real
 *        Schur form.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>
#include <string.h>

extern f32 slamch(const char* cmach);
extern void slaexc(const int wantq, const int n, f32* T, const int ldt,
                   f32* Q, const int ldq, const int j1,
                   const int n1, const int n2, f32* work, int* info);

/**
 * SGET34 tests SLAEXC, a routine for swapping adjacent blocks (either
 * 1 by 1 or 2 by 2) on the diagonal of a matrix in real Schur form.
 * Thus, SLAEXC computes an orthogonal matrix Q such that
 *
 *     Q' * [ A B ] * Q  = [ C1 B1 ]
 *          [ 0 C ]        [ 0  A1 ]
 *
 * where C1 is similar to C and A1 is similar to A.  Both A and C are
 * assumed to be in standard form (equal diagonal entries and
 * offdiagonal with differing signs) and A1 and C1 are returned with the
 * same properties.
 *
 * The test code verifies these last assertions, as well as that
 * the residual in the above equation is small.
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Number of examples where INFO is nonzero (2 elements).
 * @param[out]    knt     Total number of examples tested.
 */
#define LWORK 32

void sget34(f32* rmax, int* lmax, int ninfo[2], int* knt)
{
    const f32 ZERO = 0.0f;
    const f32 HALF = 0.5f;
    const f32 ONE  = 1.0f;
    const f32 TWO  = 2.0f;
    const f32 THREE = 3.0f;

    f32 eps = slamch("P");
    f32 smlnum = slamch("S") / eps;
    f32 bignum = ONE / smlnum;

    f32 val[9], vm[2];
    val[0] = ZERO;
    val[1] = sqrtf(smlnum);
    val[2] = ONE;
    val[3] = TWO;
    val[4] = sqrtf(bignum);
    val[5] = -sqrtf(smlnum);
    val[6] = -ONE;
    val[7] = -TWO;
    val[8] = -sqrtf(bignum);
    vm[0] = ONE;
    vm[1] = ONE + TWO * eps;

    f32 t[4 * 4], t1[4 * 4], q[4 * 4];
    f32 work[LWORK], result[2];
    f32 res;
    int info;

    for (int i = 0; i < 16; i++)
        t[i] = val[3];

    ninfo[0] = 0;
    ninfo[1] = 0;
    *knt = 0;
    *lmax = 0;
    *rmax = ZERO;

    for (int ia = 0; ia < 9; ia++) {
        for (int iam = 0; iam < 2; iam++) {
            for (int ib = 0; ib < 9; ib++) {
                for (int ic = 0; ic < 9; ic++) {
                    t[0 + 0 * 4] = val[ia] * vm[iam];
                    t[1 + 1 * 4] = val[ic];
                    t[0 + 1 * 4] = val[ib];
                    t[1 + 0 * 4] = ZERO;
                    memcpy(t1, t, 16 * sizeof(f32));
                    for (int i = 0; i < 16; i++)
                        q[i] = ZERO;
                    q[0 + 0 * 4] = ONE;
                    q[1 + 1 * 4] = ONE;
                    q[2 + 2 * 4] = ONE;
                    q[3 + 3 * 4] = ONE;
                    slaexc(1, 2, t, 4, q, 4, 0, 1, 1, work, &info);
                    if (info != 0)
                        ninfo[info - 1]++;
                    shst01(2, 0, 1, t1, 4, t, 4, q, 4, work, LWORK,
                           result);
                    res = result[0] + result[1];
                    if (info != 0)
                        res = res + ONE / eps;
                    if (t[0 + 0 * 4] != t1[1 + 1 * 4])
                        res = res + ONE / eps;
                    if (t[1 + 1 * 4] != t1[0 + 0 * 4])
                        res = res + ONE / eps;
                    if (t[1 + 0 * 4] != ZERO)
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

    for (int ia = 0; ia < 5; ia++) {
        for (int iam = 0; iam < 2; iam++) {
            for (int ib = 0; ib < 5; ib++) {
                for (int ic11 = 0; ic11 < 5; ic11++) {
                    for (int ic12 = 1; ic12 < 5; ic12++) {
                        for (int ic21 = 1; ic21 < 4; ic21++) {
                            for (int ic22 = -1; ic22 <= 1; ic22 += 2) {
                                t[0 + 0 * 4] = val[ia] * vm[iam];
                                t[0 + 1 * 4] = val[ib];
                                t[0 + 2 * 4] = -TWO * val[ib];
                                t[1 + 0 * 4] = ZERO;
                                t[1 + 1 * 4] = val[ic11];
                                t[1 + 2 * 4] = val[ic12];
                                t[2 + 0 * 4] = ZERO;
                                t[2 + 1 * 4] = -val[ic21];
                                t[2 + 2 * 4] = val[ic11] * (f32)ic22;
                                memcpy(t1, t, 16 * sizeof(f32));
                                for (int i = 0; i < 16; i++)
                                    q[i] = ZERO;
                                q[0 + 0 * 4] = ONE;
                                q[1 + 1 * 4] = ONE;
                                q[2 + 2 * 4] = ONE;
                                q[3 + 3 * 4] = ONE;
                                slaexc(1, 3, t, 4, q, 4, 0, 1, 2,
                                       work, &info);
                                if (info != 0)
                                    ninfo[info - 1]++;
                                shst01(3, 0, 2, t1, 4, t, 4, q, 4,
                                       work, LWORK, result);
                                res = result[0] + result[1];
                                if (info == 0) {
                                    if (t1[0 + 0 * 4] != t[2 + 2 * 4])
                                        res = res + ONE / eps;
                                    if (t[2 + 0 * 4] != ZERO)
                                        res = res + ONE / eps;
                                    if (t[2 + 1 * 4] != ZERO)
                                        res = res + ONE / eps;
                                    if (t[1 + 0 * 4] != 0 &&
                                        (t[0 + 0 * 4] != t[1 + 1 * 4] ||
                                         copysignf(ONE, t[0 + 1 * 4]) ==
                                         copysignf(ONE, t[1 + 0 * 4])))
                                        res = res + ONE / eps;
                                }
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

    for (int ia11 = 0; ia11 < 5; ia11++) {
        for (int ia12 = 1; ia12 < 5; ia12++) {
            for (int ia21 = 1; ia21 < 4; ia21++) {
                for (int ia22 = -1; ia22 <= 1; ia22 += 2) {
                    for (int icm = 0; icm < 2; icm++) {
                        for (int ib = 0; ib < 5; ib++) {
                            for (int ic = 0; ic < 5; ic++) {
                                t[0 + 0 * 4] = val[ia11];
                                t[0 + 1 * 4] = val[ia12];
                                t[0 + 2 * 4] = -TWO * val[ib];
                                t[1 + 0 * 4] = -val[ia21];
                                t[1 + 1 * 4] = val[ia11] * (f32)ia22;
                                t[1 + 2 * 4] = val[ib];
                                t[2 + 0 * 4] = ZERO;
                                t[2 + 1 * 4] = ZERO;
                                t[2 + 2 * 4] = val[ic] * vm[icm];
                                memcpy(t1, t, 16 * sizeof(f32));
                                for (int i = 0; i < 16; i++)
                                    q[i] = ZERO;
                                q[0 + 0 * 4] = ONE;
                                q[1 + 1 * 4] = ONE;
                                q[2 + 2 * 4] = ONE;
                                q[3 + 3 * 4] = ONE;
                                slaexc(1, 3, t, 4, q, 4, 0, 2, 1,
                                       work, &info);
                                if (info != 0)
                                    ninfo[info - 1]++;
                                shst01(3, 0, 2, t1, 4, t, 4, q, 4,
                                       work, LWORK, result);
                                res = result[0] + result[1];
                                if (info == 0) {
                                    if (t1[2 + 2 * 4] != t[0 + 0 * 4])
                                        res = res + ONE / eps;
                                    if (t[1 + 0 * 4] != ZERO)
                                        res = res + ONE / eps;
                                    if (t[2 + 0 * 4] != ZERO)
                                        res = res + ONE / eps;
                                    if (t[2 + 1 * 4] != 0 &&
                                        (t[1 + 1 * 4] != t[2 + 2 * 4] ||
                                         copysignf(ONE, t[1 + 2 * 4]) ==
                                         copysignf(ONE, t[2 + 1 * 4])))
                                        res = res + ONE / eps;
                                }
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

    for (int ia11 = 0; ia11 < 5; ia11++) {
        for (int ia12 = 1; ia12 < 5; ia12++) {
            for (int ia21 = 1; ia21 < 4; ia21++) {
                for (int ia22 = -1; ia22 <= 1; ia22 += 2) {
                    for (int ib = 0; ib < 5; ib++) {
                        for (int ic11 = 2; ic11 < 4; ic11++) {
                            for (int ic12 = 2; ic12 < 4; ic12++) {
                                for (int ic21 = 2; ic21 < 4; ic21++) {
                                    for (int ic22 = -1; ic22 <= 1; ic22 += 2) {
                                        for (int icm = 4; icm < 7; icm++) {
                                            int iam = 0;
                                            t[0 + 0 * 4] = val[ia11] * vm[iam];
                                            t[0 + 1 * 4] = val[ia12] * vm[iam];
                                            t[0 + 2 * 4] = -TWO * val[ib];
                                            t[0 + 3 * 4] = HALF * val[ib];
                                            t[1 + 0 * 4] = -t[0 + 1 * 4] * val[ia21];
                                            t[1 + 1 * 4] = val[ia11] *
                                                            (f32)ia22 * vm[iam];
                                            t[1 + 2 * 4] = val[ib];
                                            t[1 + 3 * 4] = THREE * val[ib];
                                            t[2 + 0 * 4] = ZERO;
                                            t[2 + 1 * 4] = ZERO;
                                            t[2 + 2 * 4] = val[ic11] *
                                                            fabsf(val[icm]);
                                            t[2 + 3 * 4] = val[ic12] *
                                                            fabsf(val[icm]);
                                            t[3 + 0 * 4] = ZERO;
                                            t[3 + 1 * 4] = ZERO;
                                            t[3 + 2 * 4] = -t[2 + 3 * 4] * val[ic21] *
                                                            fabsf(val[icm]);
                                            t[3 + 3 * 4] = val[ic11] *
                                                            (f32)ic22 *
                                                            fabsf(val[icm]);
                                            memcpy(t1, t, 16 * sizeof(f32));
                                            for (int i = 0; i < 16; i++)
                                                q[i] = ZERO;
                                            q[0 + 0 * 4] = ONE;
                                            q[1 + 1 * 4] = ONE;
                                            q[2 + 2 * 4] = ONE;
                                            q[3 + 3 * 4] = ONE;
                                            slaexc(1, 4, t, 4, q, 4,
                                                   0, 2, 2, work, &info);
                                            if (info != 0)
                                                ninfo[info - 1]++;
                                            shst01(4, 0, 3, t1, 4, t, 4,
                                                   q, 4, work, LWORK,
                                                   result);
                                            res = result[0] + result[1];
                                            if (info == 0) {
                                                if (t[2 + 0 * 4] != ZERO)
                                                    res = res + ONE / eps;
                                                if (t[3 + 0 * 4] != ZERO)
                                                    res = res + ONE / eps;
                                                if (t[2 + 1 * 4] != ZERO)
                                                    res = res + ONE / eps;
                                                if (t[3 + 1 * 4] != ZERO)
                                                    res = res + ONE / eps;
                                                if (t[1 + 0 * 4] != 0 &&
                                                    (t[0 + 0 * 4] != t[1 + 1 * 4] ||
                                                     copysignf(ONE, t[0 + 1 * 4]) ==
                                                     copysignf(ONE, t[1 + 0 * 4])))
                                                    res = res + ONE / eps;
                                                if (t[3 + 2 * 4] != 0 &&
                                                    (t[2 + 2 * 4] != t[3 + 3 * 4] ||
                                                     copysignf(ONE, t[2 + 3 * 4]) ==
                                                     copysignf(ONE, t[3 + 2 * 4])))
                                                    res = res + ONE / eps;
                                            }
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
}
