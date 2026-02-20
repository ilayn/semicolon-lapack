/**
 * @file dget35.c
 * @brief DGET35 tests DTRSYL, a routine for solving the Sylvester matrix
 *        equation op(A)*X + ISGN*X*op(B) = scale*C.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>

extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                  const f64* A, const int lda, f64* work);
extern void dtrsyl(const char* trana, const char* tranb, const int isgn,
                   const int m, const int n,
                   const f64* A, const int lda,
                   const f64* B, const int ldb,
                   f64* C, const int ldc, f64* scale, int* info);

/**
 * DGET35 tests DTRSYL, a routine for solving the Sylvester matrix
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

void dget35(f64* rmax, int* lmax, int* ninfo, int* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE  = 1.0;
    const f64 TWO  = 2.0;
    const f64 FOUR = 4.0;

    static const int idim[8] = {1, 2, 3, 4, 3, 3, 6, 4};

    static const int ival[8][6][6] = {
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

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") * FOUR / eps;
    f64 bignum = ONE / smlnum;

    f64 vm1[3], vm2[3];
    vm1[0] = sqrt(smlnum);
    vm1[1] = ONE;
    vm1[2] = sqrt(bignum);
    vm2[0] = ONE;
    vm2[1] = ONE + TWO * eps;
    vm2[2] = TWO;

    *knt = 0;
    *ninfo = 0;
    *lmax = 0;
    *rmax = ZERO;

    f64 a[LDA * LDA], b[LDA * LDA], c[LDA * LDA], cc[LDA * LDA];
    f64 dum[1];
    f64 scale, tnrm, cnrm, xnrm, rmul, res1, res, den;
    int m, n, info;

    for (int itrana = 0; itrana < 2; itrana++) {
        for (int itranb = 0; itranb < 2; itranb++) {
            for (int isgn = -1; isgn <= 1; isgn += 2) {
                for (int ima = 0; ima < 8; ima++) {
                    for (int imlda1 = 0; imlda1 < 3; imlda1++) {
                        for (int imlda2 = 0; imlda2 < 3; imlda2++) {
                            for (int imloff = 0; imloff < 2; imloff++) {
                                for (int imb = 0; imb < 8; imb++) {
                                    for (int imldb1 = 0; imldb1 < 3; imldb1++) {
                                        const char* trana = (itrana == 0) ? "N" : "T";
                                        const char* tranb = (itranb == 0) ? "N" : "T";
                                        m = idim[ima];
                                        n = idim[imb];
                                        tnrm = ZERO;
                                        for (int i = 0; i < m; i++) {
                                            for (int j = 0; j < m; j++) {
                                                a[i + j * LDA] = (f64)ival[ima][i][j];
                                                if (((i - j) < 0 ? (j - i) : (i - j)) <= 1) {
                                                    a[i + j * LDA] = a[i + j * LDA] *
                                                                      vm1[imlda1];
                                                    a[i + j * LDA] = a[i + j * LDA] *
                                                                      vm2[imlda2];
                                                } else {
                                                    a[i + j * LDA] = a[i + j * LDA] *
                                                                      vm1[imloff];
                                                }
                                                if (fabs(a[i + j * LDA]) > tnrm)
                                                    tnrm = fabs(a[i + j * LDA]);
                                            }
                                        }
                                        for (int i = 0; i < n; i++) {
                                            for (int j = 0; j < n; j++) {
                                                b[i + j * LDA] = (f64)ival[imb][i][j];
                                                if (((i - j) < 0 ? (j - i) : (i - j)) <= 1) {
                                                    b[i + j * LDA] = b[i + j * LDA] *
                                                                      vm1[imldb1];
                                                } else {
                                                    b[i + j * LDA] = b[i + j * LDA] *
                                                                      vm1[imloff];
                                                }
                                                if (fabs(b[i + j * LDA]) > tnrm)
                                                    tnrm = fabs(b[i + j * LDA]);
                                            }
                                        }
                                        cnrm = ZERO;
                                        for (int i = 0; i < m; i++) {
                                            for (int j = 0; j < n; j++) {
                                                c[i + j * LDA] = sin((f64)((i + 1) * (j + 1)));
                                                if (c[i + j * LDA] > cnrm)
                                                    cnrm = c[i + j * LDA];
                                                cc[i + j * LDA] = c[i + j * LDA];
                                            }
                                        }
                                        (*knt)++;
                                        dtrsyl(trana, tranb, isgn, m, n,
                                               a, LDA, b, LDA, c, LDA, &scale,
                                               &info);
                                        if (info != 0)
                                            (*ninfo)++;
                                        xnrm = dlange("M", m, n, c, LDA, dum);
                                        rmul = ONE;
                                        if (xnrm > ONE && tnrm > ONE) {
                                            if (xnrm > bignum / tnrm) {
                                                rmul = ONE / (xnrm > tnrm ? xnrm : tnrm);
                                            }
                                        }
                                        cblas_dgemm(CblasColMajor,
                                                    itrana == 0 ? CblasNoTrans : CblasTrans,
                                                    CblasNoTrans,
                                                    m, n, m, rmul,
                                                    a, LDA, c, LDA, -scale * rmul,
                                                    cc, LDA);
                                        cblas_dgemm(CblasColMajor,
                                                    CblasNoTrans,
                                                    itranb == 0 ? CblasNoTrans : CblasTrans,
                                                    m, n, n, (f64)isgn * rmul,
                                                    c, LDA, b, LDA, ONE,
                                                    cc, LDA);
                                        res1 = dlange("M", m, n, cc, LDA, dum);
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
