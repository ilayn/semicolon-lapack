/**
 * @file zsyl01.c
 * @brief ZSYL01 tests ZTRSYL and ZTRSYL3 routines for the Sylvester equation.
 */

#include "semicolon_cblas.h"
#include "verify.h"
#include <math.h>
#include <stdlib.h>
#include "test_rng.h"

/**
 * ZSYL01 tests ZTRSYL and ZTRSYL3, routines for solving the Sylvester matrix
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
void zsyl01(const f64 thresh, INT* nfail, f64* rmax, INT* ninfo, INT* knt)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    const c128 CONE = CMPLX(1.0, 0.0);
#define MAXM 185
#define MAXN 192
#define LDSWORK 36

    /* Get machine parameters */

    f64 eps = dlamch("P");
    f64 smlnum = dlamch("S") / eps;
    f64 bignum = ONE / smlnum;

    /* Expect INFO = 0 */
    f64 vm[2];
    vm[0] = ONE;
    /* Expect INFO = 1 */
    vm[1] = 0.05;

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

    c128* A = malloc((size_t)MAXM * MAXM * sizeof(c128));
    c128* B = malloc((size_t)MAXN * MAXN * sizeof(c128));
    c128* C = malloc((size_t)MAXM * MAXN * sizeof(c128));
    c128* CC = malloc((size_t)MAXM * MAXN * sizeof(c128));
    c128* X = malloc((size_t)MAXM * MAXN * sizeof(c128));
    f64* swork = malloc((size_t)LDSWORK * 103 * sizeof(f64));

    /* Stack arrays */

    c128 duml[MAXM], dumr[MAXN];
    INT dmaxmn = MAXM > MAXN ? MAXM : MAXN;
    c128 d[MAXM > MAXN ? MAXM : MAXN];
    (void)dmaxmn;
    f64 dum[MAXN];
    INT iwork[MAXM + MAXN + 2];

    uint64_t state[4];
    f64 scale, scale3;
    INT iinfo, info;

    /* Begin test loop */

    for (INT j = 0; j < 2; j++) {
        for (INT isgn = -1; isgn <= 1; isgn += 2) {
            /* Reset seed (overwritten by LATMR) */
            rng_seed(state, 1);
            for (INT m = 32; m <= MAXM; m += 51) {
                INT kla = 0;
                INT kua = m - 1;
                zlatmr(m, m, "S", "N", d,
                       6, ONE, CONE, "T", "N",
                       duml, 1, ONE, dumr, 1, ONE,
                       "N", iwork, kla, kua, ZERO,
                       ONE, "NO", A, MAXM, iwork, &iinfo,
                       state);
                for (INT i = 0; i < m; i++) {
                    A[i + (size_t)i * MAXM] *= vm[j];
                }
                f64 anrm = zlange("M", m, m, A, MAXM, dum);
                for (INT n = 51; n <= MAXN; n += 47) {
                    INT klb = 0;
                    INT kub = n - 1;
                    zlatmr(n, n, "S", "N", d,
                           6, ONE, CONE, "T", "N",
                           duml, 1, ONE, dumr, 1, ONE,
                           "N", iwork, klb, kub, ZERO,
                           ONE, "NO", B, MAXN, iwork, &iinfo,
                           state);
                    for (INT i = 0; i < n; i++) {
                        B[i + (size_t)i * MAXN] *= vm[j];
                    }
                    f64 bnrm = zlange("M", n, n, B, MAXN, dum);
                    f64 tnrm = anrm > bnrm ? anrm : bnrm;
                    zlatmr(m, n, "S", "N", d,
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

                            zlacpy("A", m, n, C, MAXM, X, MAXM);
                            zlacpy("A", m, n, C, MAXM, CC, MAXM);
                            ztrsyl(trana, tranb, isgn, m, n,
                                   A, MAXM, B, MAXN, X, MAXM,
                                   &scale, &iinfo);
                            if (iinfo != 0)
                                ninfo[0]++;
                            f64 xnrm = zlange("M", m, n, X, MAXM, dum);
                            c128 rmul = CONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    f64 denom = xnrm > tnrm ? xnrm : tnrm;
                                    rmul = CONE / denom;
                                }
                            }
                            c128 alpha = rmul;
                            c128 beta = -scale * rmul;
                            cblas_zgemm(CblasColMajor, trana_cblas,
                                        CblasNoTrans,
                                        m, n, m, &alpha,
                                        A, MAXM, X, MAXM, &beta,
                                        CC, MAXM);
                            c128 alpha2 = (f64)isgn * rmul;
                            cblas_zgemm(CblasColMajor, CblasNoTrans,
                                        tranb_cblas,
                                        m, n, n, &alpha2,
                                        X, MAXM, B, MAXN, &CONE,
                                        CC, MAXM);
                            f64 res1 = zlange("M", m, n, CC, MAXM, dum);
                            f64 denom2 = smlnum;
                            if (smlnum * xnrm > denom2)
                                denom2 = smlnum * xnrm;
                            if ((cabs(rmul) * tnrm) * eps * xnrm > denom2)
                                denom2 = (cabs(rmul) * tnrm) * eps * xnrm;
                            f64 res = res1 / denom2;
                            if (res > thresh)
                                nfail[0]++;
                            if (res > rmax[0])
                                rmax[0] = res;

                            zlacpy("A", m, n, C, MAXM, X, MAXM);
                            zlacpy("A", m, n, C, MAXM, CC, MAXM);
                            ztrsyl3(trana, tranb, isgn, m, n,
                                    A, MAXM, B, MAXN, X, MAXM,
                                    &scale3, swork, LDSWORK, &info);
                            if (info != 0)
                                ninfo[1]++;
                            xnrm = zlange("M", m, n, X, MAXM, dum);
                            rmul = CONE;
                            if (xnrm > ONE && tnrm > ONE) {
                                if (xnrm > bignum / tnrm) {
                                    f64 denom = xnrm > tnrm ? xnrm : tnrm;
                                    rmul = CONE / denom;
                                }
                            }
                            alpha = rmul;
                            beta = -scale3 * rmul;
                            cblas_zgemm(CblasColMajor, trana_cblas,
                                        CblasNoTrans,
                                        m, n, m, &alpha,
                                        A, MAXM, X, MAXM, &beta,
                                        CC, MAXM);
                            alpha2 = (f64)isgn * rmul;
                            cblas_zgemm(CblasColMajor, CblasNoTrans,
                                        tranb_cblas,
                                        m, n, n, &alpha2,
                                        X, MAXM, B, MAXN, &CONE,
                                        CC, MAXM);
                            res1 = zlange("M", m, n, CC, MAXM, dum);
                            denom2 = smlnum;
                            if (smlnum * xnrm > denom2)
                                denom2 = smlnum * xnrm;
                            if ((cabs(rmul) * tnrm) * eps * xnrm > denom2)
                                denom2 = (cabs(rmul) * tnrm) * eps * xnrm;
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
