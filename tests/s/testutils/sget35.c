/**
 * @file sget35.c
 * @brief SGET35 tests STRSYL, a routine for solving the Sylvester matrix
 *        equation op(A)*X + ISGN*X*op(B) = scale*C.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>

/**
 * SGET35 tests STRSYL, a routine for solving the Sylvester matrix
 * equation
 *
 *    op(A)*X + ISGN*X*op(B) = scale*C,
 *
 * A and B are assumed to be in Schur canonical form, op() represents an
 * optional transpose, and ISGN can be -1 or +1.  Scale is an output
 * less than or equal to 1, chosen to avoid overflow in X.
 *
 * @param[out]    rmax    Value of the largest test ratio.
 * @param[out]    lmax    Example number where largest test ratio achieved.
 * @param[out]    ninfo   Number of examples where INFO is nonzero.
 * @param[out]    knt     Total number of examples tested.
 */
#define LDA 6

void sget35(f32* rmax, INT* lmax, INT* ninfo, INT* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE  = 1.0f;
    const f32 TWO  = 2.0f;
    const f32 FOUR = 4.0f;

    static const INT idim[8] = {1, 2, 3, 4, 3, 3, 6, 4};

    static const INT ival[8][6][6] = {
        {
            {  1,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0}
        },
        {
            {  1,  -2,   0,   0,   0,   0},
            {  2,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0}
        },
        {
            {  1,   5,  -8,   0,   0,   0},
            {  0,   1,  -2,   0,   0,   0},
            {  0,   2,   1,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0}
        },
        {
            {  3,  -5,   1,  -3,   0,   0},
            {  4,   3,   2,  -9,   0,   0},
            {  0,   0,   1,  -1,   0,   0},
            {  0,   0,   4,   1,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0}
        },
        {
            {  1,   2,   5,   0,   0,   0},
            {  0,   3,   6,   0,   0,   0},
            {  0,   0,   7,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0}
        },
        {
            {  1,   1,   2,   0,   0,   0},
            {  0,   3,   5,   0,   0,   0},
            {  0,  -4,   2,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0}
        },
        {
            {  1,  -2,   5,  -1,   8,   9},
            {  2,   0,   6,  -9,   8,   9},
            {  0,   0,   3,  -5,   8,   9},
            {  0,   0,   4,   2,   8,   9},
            {  0,   0,   0,   0,   5,  -7},
            {  0,   0,   0,   0,   6,   5}
        },
        {
            {  1,   1,   2,   1,   0,   0},
            {  0,   5, -21,   2,   0,   0},
            {  0,   2,   5,   3,   0,   0},
            {  0,   0,   0,   4,   0,   0},
            {  0,   0,   0,   0,   0,   0},
            {  0,   0,   0,   0,   0,   0}
        },
    };

    f32 eps = slamch("P");
    f32 smlnum = slamch("S") * FOUR / eps;
    f32 bignum = ONE / smlnum;

    f32 vm1[3], vm2[3];
    vm1[0] = sqrtf(smlnum);
    vm1[1] = ONE;
    vm1[2] = sqrtf(bignum);
    vm2[0] = ONE;
    vm2[1] = ONE + TWO * eps;
    vm2[2] = TWO;

    *knt = 0;
    *ninfo = 0;
    *lmax = 0;
    *rmax = ZERO;

    f32 a[LDA * LDA], b[LDA * LDA], c[LDA * LDA], cc[LDA * LDA];
    f32 dum[1];
    f32 scale, tnrm, cnrm, xnrm, rmul, res1, res, den;
    INT m, n, info;

    for (INT itrana = 0; itrana < 2; itrana++) {
        for (INT itranb = 0; itranb < 2; itranb++) {
            for (INT isgn = -1; isgn <= 1; isgn += 2) {
                for (INT ima = 0; ima < 8; ima++) {
                    for (INT imlda1 = 0; imlda1 < 3; imlda1++) {
                        for (INT imlda2 = 0; imlda2 < 3; imlda2++) {
                            for (INT imloff = 0; imloff < 2; imloff++) {
                                for (INT imb = 0; imb < 8; imb++) {
                                    for (INT imldb1 = 0; imldb1 < 3; imldb1++) {
                                        const char* trana = (itrana == 0) ? "N" : "T";
                                        const char* tranb = (itranb == 0) ? "N" : "T";
                                        m = idim[ima];
                                        n = idim[imb];
                                        tnrm = ZERO;
                                        for (INT i = 0; i < m; i++) {
                                            for (INT j = 0; j < m; j++) {
                                                a[i + j * LDA] = (f32)ival[ima][i][j];
                                                if (((i - j) < 0 ? (j - i) : (i - j)) <= 1) {
                                                    a[i + j * LDA] = a[i + j * LDA] *
                                                                      vm1[imlda1];
                                                    a[i + j * LDA] = a[i + j * LDA] *
                                                                      vm2[imlda2];
                                                } else {
                                                    a[i + j * LDA] = a[i + j * LDA] *
                                                                      vm1[imloff];
                                                }
                                                if (fabsf(a[i + j * LDA]) > tnrm)
                                                    tnrm = fabsf(a[i + j * LDA]);
                                            }
                                        }
                                        for (INT i = 0; i < n; i++) {
                                            for (INT j = 0; j < n; j++) {
                                                b[i + j * LDA] = (f32)ival[imb][i][j];
                                                if (((i - j) < 0 ? (j - i) : (i - j)) <= 1) {
                                                    b[i + j * LDA] = b[i + j * LDA] *
                                                                      vm1[imldb1];
                                                } else {
                                                    b[i + j * LDA] = b[i + j * LDA] *
                                                                      vm1[imloff];
                                                }
                                                if (fabsf(b[i + j * LDA]) > tnrm)
                                                    tnrm = fabsf(b[i + j * LDA]);
                                            }
                                        }
                                        cnrm = ZERO;
                                        for (INT i = 0; i < m; i++) {
                                            for (INT j = 0; j < n; j++) {
                                                c[i + j * LDA] = sinf((f32)((i + 1) * (j + 1)));
                                                if (c[i + j * LDA] > cnrm)
                                                    cnrm = c[i + j * LDA];
                                                cc[i + j * LDA] = c[i + j * LDA];
                                            }
                                        }
                                        (*knt)++;
                                        strsyl(trana, tranb, isgn, m, n,
                                               a, LDA, b, LDA, c, LDA, &scale,
                                               &info);
                                        if (info != 0)
                                            (*ninfo)++;
                                        xnrm = slange("M", m, n, c, LDA, dum);
                                        rmul = ONE;
                                        if (xnrm > ONE && tnrm > ONE) {
                                            if (xnrm > bignum / tnrm) {
                                                rmul = ONE / (xnrm > tnrm ? xnrm : tnrm);
                                            }
                                        }
                                        cblas_sgemm(CblasColMajor,
                                                    itrana == 0 ? CblasNoTrans : CblasTrans,
                                                    CblasNoTrans,
                                                    m, n, m, rmul,
                                                    a, LDA, c, LDA, -scale * rmul,
                                                    cc, LDA);
                                        cblas_sgemm(CblasColMajor,
                                                    CblasNoTrans,
                                                    itranb == 0 ? CblasNoTrans : CblasTrans,
                                                    m, n, n, (f32)isgn * rmul,
                                                    c, LDA, b, LDA, ONE,
                                                    cc, LDA);
                                        res1 = slange("M", m, n, cc, LDA, dum);
                                        den = smlnum;
                                        if (smlnum * xnrm > den) den = smlnum * xnrm;
                                        if (((rmul * tnrm) * eps) * xnrm > den)
                                            den = ((rmul * tnrm) * eps) * xnrm;
                                        res = res1 / den;
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
