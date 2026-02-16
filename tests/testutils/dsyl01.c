/**
 * @file dsyl01.c
 * @brief DSYL01 tests DTRSYL and DTRSYL3 routines for the Sylvester equation.
 */

#include "verify.h"
#include <cblas.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "test_rng.h"

extern f64 dlamch(const char* cmach);
extern f64 dlange(const char* norm, const int m, const int n,
                  const f64* A, const int lda, f64* work);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dtrsyl(const char* trana, const char* tranb, const int isgn,
                   const int m, const int n,
                   const f64* A, const int lda,
                   const f64* B, const int ldb,
                   f64* C, const int ldc, f64* scale, int* info);
extern void dtrsyl3(const char* trana, const char* tranb, const int isgn,
                    const int m, const int n,
                    const f64* A, const int lda,
                    const f64* B, const int ldb,
                    f64* C, const int ldc, f64* scale,
                    int* iwork, const int liwork,
                    f64* swork, const int ldswork, int* info);

/**
 * DSYL01 tests DTRSYL and DTRSYL3, routines for solving the Sylvester matrix
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
 *                        nfail[0] = No. of times residual DTRSYL exceeds threshold
 *                        nfail[1] = No. of times residual DTRSYL3 exceeds threshold
 *                        nfail[2] = No. of times DTRSYL3 and DTRSYL deviate
 * @param[out]    rmax    Double precision array, dimension (2).
 *                        rmax[0] = Value of the largest test ratio of DTRSYL
 *                        rmax[1] = Value of the largest test ratio of DTRSYL3
 * @param[out]    ninfo   Integer array, dimension (2).
 *                        ninfo[0] = No. of times DTRSYL returns nonzero INFO
 *                        ninfo[1] = No. of times DTRSYL3 returns nonzero INFO
 * @param[out]    knt     Total number of examples tested.
 */
void dsyl01(const f64 thresh, int* nfail, f64* rmax, int* ninfo, int* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
#define MAXM 245
#define MAXN 192
#define LDSWORK (36 + MAXM)

    /* Get machine parameters */

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    f64 vm[2];
    vm[0] = ONE;
    vm[1] = 0.000001;

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

    f64* A = malloc((size_t)MAXM * MAXM * sizeof(f64));
    f64* B = malloc((size_t)MAXN * MAXN * sizeof(f64));
    f64* C = malloc((size_t)MAXM * MAXN * sizeof(f64));
    f64* CC = malloc((size_t)MAXM * MAXN * sizeof(f64));
    f64* X = malloc((size_t)MAXM * MAXN * sizeof(f64));
    f64* swork = malloc((size_t)LDSWORK * 126 * sizeof(f64));

    /* Stack arrays */

    f64 duml[MAXM], dumr[MAXN];
    // int dmaxmn = MAXM > MAXN ? MAXM : MAXN;
    f64 d[245]; /* max(MAXM, MAXN) = 245 */
    int liwork = MAXM + MAXN + 2;
    int iwork[MAXM + MAXN + 2];

    uint64_t state[4];
    f64 scale, scale3;
    int iinfo, info;

    /* Begin test loop */

    for (int j = 0; j < 2; j++) {
        for (int isgn = -1; isgn <= 1; isgn += 2) {
            /* Reset seed (overwritten by LATMR) */
            rng_seed(state, 1);
            for (int m = 32; m <= MAXM; m += 71) {
                int kla = 0;
                int kua = m - 1;
                dlatmr(m, m, "S", "N", d,
                       6, ONE, ONE, "T", "N",
                       duml, 1, ONE, dumr, 1, ONE,
                       "N", iwork, kla, kua, ZERO,
                       ONE, "NO", A, MAXM, iwork, &iinfo,
                       state);
                for (int i = 0; i < m; i++) {
                    A[i + (size_t)i * MAXM] *= vm[j];
                }
                f64 anrm = dlange("M", m, m, A, MAXM, NULL);
                for (int n = 51; n <= MAXN; n += 47) {
                    int klb = 0;
                    int kub = n - 1;
                    dlatmr(n, n, "S", "N", d,
                           6, ONE, ONE, "T", "N",
                           duml, 1, ONE, dumr, 1, ONE,
                           "N", iwork, klb, kub, ZERO,
                           ONE, "NO", B, MAXN, iwork, &iinfo,
                           state);
                    f64 bnrm = dlange("M", n, n, B, MAXN, NULL);
                    f64 tnrm = anrm > bnrm ? anrm : bnrm;
                    dlatmr(m, n, "S", "N", d,
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

                            dlacpy("All", m, n, C, MAXM, X, MAXM);
                            dlacpy("All", m, n, C, MAXM, CC, MAXM);
                            dtrsyl(trana, tranb, isgn, m, n,
                                   A, MAXM, B, MAXN, X, MAXM,
                                   &scale, &iinfo);
                            if (iinfo != 0)
                                ninfo[0]++;
                            f64 xnrm = dlange("M", m, n, X, MAXM, NULL);
                            f64 rmul = ONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    rmul = ONE / (xnrm > tnrm ? xnrm : tnrm);
                                }
                            }
                            cblas_dgemm(CblasColMajor, trana_flag, CblasNoTrans,
                                        m, n, m, rmul,
                                        A, MAXM, X, MAXM, -scale * rmul,
                                        CC, MAXM);
                            cblas_dgemm(CblasColMajor, CblasNoTrans, tranb_flag,
                                        m, n, n, (f64)isgn * rmul,
                                        X, MAXM, B, MAXN, ONE,
                                        CC, MAXM);
                            f64 res1 = dlange("M", m, n, CC, MAXM, NULL);
                            f64 denom = smlnum;
                            if (smlnum * xnrm > denom) denom = smlnum * xnrm;
                            if ((rmul * tnrm) * eps * xnrm > denom) denom = (rmul * tnrm) * eps * xnrm;
                            f64 res = res1 / denom;
                            if (res > thresh)
                                nfail[0]++;
                            if (res > rmax[0])
                                rmax[0] = res;

                            dlacpy("All", m, n, C, MAXM, X, MAXM);
                            dlacpy("All", m, n, C, MAXM, CC, MAXM);
                            dtrsyl3(trana, tranb, isgn, m, n,
                                    A, MAXM, B, MAXN, X, MAXM,
                                    &scale3, iwork, liwork,
                                    swork, LDSWORK, &info);
                            if (info != 0)
                                ninfo[1]++;
                            xnrm = dlange("M", m, n, X, MAXM, NULL);
                            rmul = ONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    rmul = ONE / (xnrm > tnrm ? xnrm : tnrm);
                                }
                            }
                            cblas_dgemm(CblasColMajor, trana_flag, CblasNoTrans,
                                        m, n, m, rmul,
                                        A, MAXM, X, MAXM, -scale3 * rmul,
                                        CC, MAXM);
                            cblas_dgemm(CblasColMajor, CblasNoTrans, tranb_flag,
                                        m, n, n, (f64)isgn * rmul,
                                        X, MAXM, B, MAXN, ONE,
                                        CC, MAXM);
                            res1 = dlange("M", m, n, CC, MAXM, NULL);
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
