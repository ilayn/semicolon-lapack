/**
 * @file csyl01.c
 * @brief CSYL01 tests CTRSYL and CTRSYL3 routines for the Sylvester equation.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>
#include <stdlib.h>
#include "test_rng.h"

/**
 * CSYL01 tests CTRSYL and CTRSYL3, routines for solving the Sylvester matrix
 * equation
 *
 *    op(A)*X + ISGN*X*op(B) = scale*C,
 *
 * where op(A) and op(B) are both upper triangular form, op() represents an
 * optional conjugate transpose, and ISGN can be -1 or +1. Scale is an output
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
 * @param[out]    rmax    Double precision array, dimension (2).
 * @param[out]    ninfo   Integer array, dimension (2).
 * @param[out]    knt     Total number of examples tested.
 */
void csyl01(const f32 thresh, INT* nfail, f32* rmax, INT* ninfo, INT* knt)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const c64 CONE = CMPLXF(1.0f, 0.0f);
#define MAXM 185
#define MAXN 192
#define LDSWORK 36

    /* Get machine parameters */

    f32 eps = slamch("P");
    f32 smlnum = slamch("S") / eps;
    f32 bignum = ONE / smlnum;

    /* Expect INFO = 0 */
    f32 vm[2];
    vm[0] = ONE;
    /* Expect INFO = 1 */
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

    c64* A = malloc((size_t)MAXM * MAXM * sizeof(c64));
    c64* B = malloc((size_t)MAXN * MAXN * sizeof(c64));
    c64* C = malloc((size_t)MAXM * MAXN * sizeof(c64));
    c64* CC = malloc((size_t)MAXM * MAXN * sizeof(c64));
    c64* X = malloc((size_t)MAXM * MAXN * sizeof(c64));
    f32* swork = malloc((size_t)LDSWORK * 103 * sizeof(f32));

    /* Stack arrays */

    c64 duml[MAXM], dumr[MAXN];
    INT dmaxmn = MAXM > MAXN ? MAXM : MAXN;
    c64 d[MAXM > MAXN ? MAXM : MAXN];
    (void)dmaxmn;
    f32 dum[MAXN];
    INT iwork[MAXM + MAXN + 2];

    uint64_t state[4];
    f32 scale, scale3;
    INT iinfo, info;

    /* Begin test loop */

    for (INT j = 0; j < 2; j++) {
        for (INT isgn = -1; isgn <= 1; isgn += 2) {
            /* Reset seed (overwritten by LATMR) */
            rng_seed(state, 1);
            for (INT m = 32; m <= MAXM; m += 51) {
                INT kla = 0;
                INT kua = m - 1;
                clatmr(m, m, "S", "N", d,
                       6, ONE, CONE, "T", "N",
                       duml, 1, ONE, dumr, 1, ONE,
                       "N", iwork, kla, kua, ZERO,
                       ONE, "NO", A, MAXM, iwork, &iinfo,
                       state);
                for (INT i = 0; i < m; i++) {
                    A[i + (size_t)i * MAXM] *= vm[j];
                }
                f32 anrm = clange("M", m, m, A, MAXM, dum);
                for (INT n = 51; n <= MAXN; n += 47) {
                    INT klb = 0;
                    INT kub = n - 1;
                    clatmr(n, n, "S", "N", d,
                           6, ONE, CONE, "T", "N",
                           duml, 1, ONE, dumr, 1, ONE,
                           "N", iwork, klb, kub, ZERO,
                           ONE, "NO", B, MAXN, iwork, &iinfo,
                           state);
                    for (INT i = 0; i < n; i++) {
                        B[i + (size_t)i * MAXN] *= vm[j];
                    }
                    f32 bnrm = clange("M", n, n, B, MAXN, dum);
                    f32 tnrm = anrm > bnrm ? anrm : bnrm;
                    clatmr(m, n, "S", "N", d,
                           6, ONE, CONE, "T", "N",
                           duml, 1, ONE, dumr, 1, ONE,
                           "N", iwork, m, n, ZERO, ONE,
                           "NO", C, MAXM, iwork, &iinfo,
                           state);
                    for (INT itrana = 0; itrana < 2; itrana++) {
                        const char* trana = (itrana == 0) ? "N" : "C";
                        CBLAS_TRANSPOSE trana_cblas = (itrana == 0) ?
                            CblasNoTrans : CblasConjTrans;
                        for (INT itranb = 0; itranb < 2; itranb++) {
                            const char* tranb = (itranb == 0) ? "N" : "C";
                            CBLAS_TRANSPOSE tranb_cblas = (itranb == 0) ?
                                CblasNoTrans : CblasConjTrans;
                            (*knt)++;

                            clacpy("A", m, n, C, MAXM, X, MAXM);
                            clacpy("A", m, n, C, MAXM, CC, MAXM);
                            ctrsyl(trana, tranb, isgn, m, n,
                                   A, MAXM, B, MAXN, X, MAXM,
                                   &scale, &iinfo);
                            if (iinfo != 0)
                                ninfo[0]++;
                            f32 xnrm = clange("M", m, n, X, MAXM, dum);
                            c64 rmul = CONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    f32 denom = xnrm > tnrm ? xnrm : tnrm;
                                    rmul = CONE / denom;
                                }
                            }
                            c64 alpha = rmul;
                            c64 beta = -scale * rmul;
                            cblas_cgemm(CblasColMajor, trana_cblas,
                                        CblasNoTrans,
                                        m, n, m, &alpha,
                                        A, MAXM, X, MAXM, &beta,
                                        CC, MAXM);
                            c64 alpha2 = (f32)isgn * rmul;
                            cblas_cgemm(CblasColMajor, CblasNoTrans,
                                        tranb_cblas,
                                        m, n, n, &alpha2,
                                        X, MAXM, B, MAXN, &CONE,
                                        CC, MAXM);
                            f32 res1 = clange("M", m, n, CC, MAXM, dum);
                            f32 denom2 = smlnum;
                            if (smlnum * xnrm > denom2)
                                denom2 = smlnum * xnrm;
                            if ((cabsf(rmul) * tnrm) * eps * xnrm > denom2)
                                denom2 = (cabsf(rmul) * tnrm) * eps * xnrm;
                            f32 res = res1 / denom2;
                            if (res > thresh)
                                nfail[0]++;
                            if (res > rmax[0])
                                rmax[0] = res;

                            clacpy("A", m, n, C, MAXM, X, MAXM);
                            clacpy("A", m, n, C, MAXM, CC, MAXM);
                            ctrsyl3(trana, tranb, isgn, m, n,
                                    A, MAXM, B, MAXN, X, MAXM,
                                    &scale3, swork, LDSWORK, &info);
                            if (info != 0)
                                ninfo[1]++;
                            xnrm = clange("M", m, n, X, MAXM, dum);
                            rmul = CONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    f32 denom = xnrm > tnrm ? xnrm : tnrm;
                                    rmul = CONE / denom;
                                }
                            }
                            alpha = rmul;
                            beta = -scale3 * rmul;
                            cblas_cgemm(CblasColMajor, trana_cblas,
                                        CblasNoTrans,
                                        m, n, m, &alpha,
                                        A, MAXM, X, MAXM, &beta,
                                        CC, MAXM);
                            alpha2 = (f32)isgn * rmul;
                            cblas_cgemm(CblasColMajor, CblasNoTrans,
                                        tranb_cblas,
                                        m, n, n, &alpha2,
                                        X, MAXM, B, MAXN, &CONE,
                                        CC, MAXM);
                            res1 = clange("M", m, n, CC, MAXM, dum);
                            denom2 = smlnum;
                            if (smlnum * xnrm > denom2)
                                denom2 = smlnum * xnrm;
                            if ((cabsf(rmul) * tnrm) * eps * xnrm > denom2)
                                denom2 = (cabsf(rmul) * tnrm) * eps * xnrm;
                            res = res1 / denom2;
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

#undef MAXM
#undef MAXN
#undef LDSWORK
}
