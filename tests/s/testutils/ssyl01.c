/**
 * @file ssyl01.c
 * @brief SSYL01 tests STRSYL and STRSYL3 routines for the Sylvester equation.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "test_rng.h"

extern f32 slamch(const char* cmach);
extern f32 slange(const char* norm, const int m, const int n,
                  const f32* A, const int lda, f32* work);
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void strsyl(const char* trana, const char* tranb, const int isgn,
                   const int m, const int n,
                   const f32* A, const int lda,
                   const f32* B, const int ldb,
                   f32* C, const int ldc, f32* scale, int* info);
extern void strsyl3(const char* trana, const char* tranb, const int isgn,
                    const int m, const int n,
                    const f32* A, const int lda,
                    const f32* B, const int ldb,
                    f32* C, const int ldc, f32* scale,
                    int* iwork, const int liwork,
                    f32* swork, const int ldswork, int* info);

/**
 * SSYL01 tests STRSYL and STRSYL3, routines for solving the Sylvester matrix
 * equation
 *
 *    op(A)*X + ISGN*X*op(B) = scale*C,
 *
 * A and B are assumed to be in Schur canonical form, op() represents an
 * optional transpose, and ISGN can be -1 or +1.  Scale is an output
 * less than or equal to 1, chosen to avoid overflow in X.
 *
 * The test code verifies that the following residual does not exceed
 * the provided threshold:
 *
 *    norm(op(A)*X + ISGN*X*op(B) - scale*C) /
 *        (EPS*max(norm(A),norm(B))*norm(X))
 *
 * @param[in]     thresh  Threshold for test failure.
 * @param[out]    nfail   Integer array, dimension (3).
 *                        nfail[0] = No. of times residual STRSYL exceeds threshold
 *                        nfail[1] = No. of times residual STRSYL3 exceeds threshold
 *                        nfail[2] = No. of times STRSYL3 and STRSYL deviate
 * @param[out]    rmax    Double precision array, dimension (2).
 *                        rmax[0] = Value of the largest test ratio of STRSYL
 *                        rmax[1] = Value of the largest test ratio of STRSYL3
 * @param[out]    ninfo   Integer array, dimension (2).
 *                        ninfo[0] = No. of times STRSYL returns nonzero INFO
 *                        ninfo[1] = No. of times STRSYL3 returns nonzero INFO
 * @param[out]    knt     Total number of examples tested.
 */
void ssyl01(const f32 thresh, int* nfail, f32* rmax, int* ninfo, int* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
#define MAXM 101
#define MAXN 138
#define LDSWORK 18

    /* Get machine parameters */

    f32 eps = slamch("P");
    f32 smlnum = slamch("S") / eps;
    f32 bignum = ONE / smlnum;

    f32 vm[2];
    vm[0] = ONE;
    vm[1] = 0.05f;

    /* Initialize outputs */

    ninfo[0] = 0;
    ninfo[1] = 0;
    nfail[0] = 0;
    nfail[1] = 0;
    nfail[2] = 0;
    rmax[0] = ZERO;
    rmax[1] = ZERO;
    *knt = 0;

    /* Allocate large arrays on the heap */

    f32* A = malloc((size_t)MAXM * MAXM * sizeof(f32));
    f32* B = malloc((size_t)MAXN * MAXN * sizeof(f32));
    f32* C = malloc((size_t)MAXM * MAXN * sizeof(f32));
    f32* CC = malloc((size_t)MAXM * MAXN * sizeof(f32));
    f32* X = malloc((size_t)MAXM * MAXN * sizeof(f32));
    f32* swork = malloc((size_t)LDSWORK * 54 * sizeof(f32));

    /* Stack arrays */

    f32 duml[MAXM], dumr[MAXN];
    // int dmaxmn = MAXM > MAXN ? MAXM : MAXN;
    f32 d[245]; /* max(MAXM, MAXN) = 245 */
    int liwork = MAXM + MAXN + 2;
    int iwork[MAXM + MAXN + 2];

    uint64_t state[4];
    f32 scale, scale3;
    int iinfo, info;

    /* Begin test loop */

    for (int j = 0; j < 2; j++) {
        for (int isgn = -1; isgn <= 1; isgn += 2) {
            /* Reset seed (overwritten by LATMR) */
            rng_seed(state, 1);
            for (int m = 32; m <= MAXM; m += 71) {
                int kla = 0;
                int kua = m - 1;
                slatmr(m, m, "S", "N", d,
                       6, ONE, ONE, "T", "N",
                       duml, 1, ONE, dumr, 1, ONE,
                       "N", iwork, kla, kua, ZERO,
                       ONE, "NO", A, MAXM, iwork, &iinfo,
                       state);
                for (int i = 0; i < m; i++) {
                    A[i + (size_t)i * MAXM] *= vm[j];
                }
                f32 anrm = slange("M", m, m, A, MAXM, NULL);
                for (int n = 51; n <= MAXN; n += 47) {
                    int klb = 0;
                    int kub = n - 1;
                    slatmr(n, n, "S", "N", d,
                           6, ONE, ONE, "T", "N",
                           duml, 1, ONE, dumr, 1, ONE,
                           "N", iwork, klb, kub, ZERO,
                           ONE, "NO", B, MAXN, iwork, &iinfo,
                           state);
                    f32 bnrm = slange("M", n, n, B, MAXN, NULL);
                    f32 tnrm = anrm > bnrm ? anrm : bnrm;
                    slatmr(m, n, "S", "N", d,
                           6, ONE, ONE, "T", "N",
                           duml, 1, ONE, dumr, 1, ONE,
                           "N", iwork, m, n, ZERO, ONE,
                           "NO", C, MAXM, iwork, &iinfo,
                           state);
                    for (int itrana = 0; itrana < 2; itrana++) {
                        const char* trana = (itrana == 0) ? "N" : "T";
                        CBLAS_TRANSPOSE trana_flag = (itrana == 0) ? CblasNoTrans : CblasTrans;
                        for (int itranb = 0; itranb < 2; itranb++) {
                            const char* tranb = (itranb == 0) ? "N" : "T";
                            CBLAS_TRANSPOSE tranb_flag = (itranb == 0) ? CblasNoTrans : CblasTrans;
                            (*knt)++;

                            slacpy("All", m, n, C, MAXM, X, MAXM);
                            slacpy("All", m, n, C, MAXM, CC, MAXM);
                            strsyl(trana, tranb, isgn, m, n,
                                   A, MAXM, B, MAXN, X, MAXM,
                                   &scale, &iinfo);
                            if (iinfo != 0)
                                ninfo[0]++;
                            f32 xnrm = slange("M", m, n, X, MAXM, NULL);
                            f32 rmul = ONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    rmul = ONE / (xnrm > tnrm ? xnrm : tnrm);
                                }
                            }
                            cblas_sgemm(CblasColMajor, trana_flag, CblasNoTrans,
                                        m, n, m, rmul,
                                        A, MAXM, X, MAXM, -scale * rmul,
                                        CC, MAXM);
                            cblas_sgemm(CblasColMajor, CblasNoTrans, tranb_flag,
                                        m, n, n, (f32)isgn * rmul,
                                        X, MAXM, B, MAXN, ONE,
                                        CC, MAXM);
                            f32 res1 = slange("M", m, n, CC, MAXM, NULL);
                            f32 denom = smlnum;
                            if (smlnum * xnrm > denom) denom = smlnum * xnrm;
                            if ((rmul * tnrm) * eps * xnrm > denom) denom = (rmul * tnrm) * eps * xnrm;
                            f32 res = res1 / denom;
                            if (res > thresh)
                                nfail[0]++;
                            if (res > rmax[0])
                                rmax[0] = res;

                            slacpy("All", m, n, C, MAXM, X, MAXM);
                            slacpy("All", m, n, C, MAXM, CC, MAXM);
                            strsyl3(trana, tranb, isgn, m, n,
                                    A, MAXM, B, MAXN, X, MAXM,
                                    &scale3, iwork, liwork,
                                    swork, LDSWORK, &info);
                            if (info != 0)
                                ninfo[1]++;
                            xnrm = slange("M", m, n, X, MAXM, NULL);
                            rmul = ONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    rmul = ONE / (xnrm > tnrm ? xnrm : tnrm);
                                }
                            }
                            cblas_sgemm(CblasColMajor, trana_flag, CblasNoTrans,
                                        m, n, m, rmul,
                                        A, MAXM, X, MAXM, -scale3 * rmul,
                                        CC, MAXM);
                            cblas_sgemm(CblasColMajor, CblasNoTrans, tranb_flag,
                                        m, n, n, (f32)isgn * rmul,
                                        X, MAXM, B, MAXN, ONE,
                                        CC, MAXM);
                            res1 = slange("M", m, n, CC, MAXM, NULL);
                            denom = smlnum;
                            if (smlnum * xnrm > denom) denom = smlnum * xnrm;
                            if ((rmul * tnrm) * eps * xnrm > denom) denom = (rmul * tnrm) * eps * xnrm;
                            res = res1 / denom;
                            /* Verify that TRSYL3 only flushes if TRSYL flushes (but
                               there may be cases where TRSYL3 avoid flushing). */
                            if ((scale3 == ZERO && scale > ZERO) ||
                                iinfo != info) {
                                nfail[2]++;
                            }
                            if (res > thresh || isnan(res))
                                nfail[1]++;
                            if (res > rmax[1])
                                rmax[1] = res;
                        }
                    }
                }
            }
        }
    }

    free(A);
    free(B);
    free(C);
    free(CC);
    free(X);
    free(swork);
}
