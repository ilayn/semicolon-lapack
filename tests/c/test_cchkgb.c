/**
 * @file test_cchkgb.c
 * @brief Comprehensive test suite for complex general band matrix (CGB) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkgb.f to C using CMocka.
 * Tests CGBTRF, CGBTRS, CGBRFS, and CGBCON.
 *
 * Each (m, n, kl, ku, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkgb.f:
 *   TEST 1: LU factorization residual via cgbt01
 *   TEST 2: Solution residual via cgbt02
 *   TEST 3: Solution accuracy via cget04
 *   TEST 4: Refined solution accuracy via cget04 (after cgbrfs)
 *   TEST 5-6: Error bounds via cgbt05
 *   TEST 7: Condition number via sget06
 *
 * Parameters from ztest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   KL/KU values: 0, (5*M+1)/4, (3M-1)/4, (M+1)/4 (clamped to matrix size)
 *   NRHS values: 1, 2, 15
 *   NB values: 1, 3, 3, 3, 20
 *   Matrix types: 1-8
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>

static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};
static const INT NBVAL[] = {1, 3, 3, 3, 20};
static const char TRANSS[] = {'N', 'T', 'C'};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTRAN   (sizeof(TRANSS) / sizeof(TRANSS[0]))
#define NTYPES  8
#define NTESTS  7
#define THRESH  30.0f
#define NMAX    50
#define NSMAX   15
#define NBW     4

typedef struct {
    INT m;
    INT n;
    INT kl;
    INT ku;
    INT imat;
    INT inb;
    char name[80];
} zchkgb_params_t;

typedef struct {
    c64* A;
    c64* AFAC;
    c64* B;
    c64* X;
    c64* XACT;
    c64* WORK;
    f32* RWORK;
    f32* D;
    f32* FERR;
    f32* BERR;
    INT* IPIV;
} zchkgb_workspace_t;

static zchkgb_workspace_t* g_workspace = NULL;

#define KLMAX   (NMAX - 1)
#define KUMAX   (NMAX - 1)
#define LA      ((KLMAX + KUMAX + 1) * NMAX)
#define LAFAC   ((2 * KLMAX + KUMAX + 1) * NMAX)

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkgb_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(LA * sizeof(c64));
    g_workspace->AFAC = malloc(LAFAC * sizeof(c64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->WORK = malloc(3 * NMAX * NMAX * sizeof(c64));
    g_workspace->RWORK = malloc((NMAX + 2 * NSMAX) * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
    g_workspace->IPIV = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->IPIV) {
        return -1;
    }

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->AFAC);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->D);
        free(g_workspace->FERR);
        free(g_workspace->BERR);
        free(g_workspace->IPIV);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_zchkgb_single(INT m, INT n, INT kl, INT ku, INT imat, INT inb)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    zchkgb_workspace_t* ws = g_workspace;

    char type, dist;
    INT kl_gen, ku_gen, mode;
    f32 anorm_gen, cndnum;
    INT info, izero;
    INT lda = kl + ku + 1;
    INT ldafac = 2 * kl + ku + 1;
    INT ldb = (n > 1) ? n : 1;
    INT trfcon;
    f32 anormo = 0.0f, anormi = 0.0f, rcondo = 0.0f, rcondi = 0.0f, rcond, rcondc;
    f32 result[NTESTS];

    INT nb = NBVAL[inb];
    xlaenv(1, nb);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL +
                    (uint64_t)(m * 10000 + n * 1000 + kl * 100 + ku * 10 + imat));

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    clatb4("CGB", imat, m, n, &type, &kl_gen, &ku_gen, &anorm_gen, &mode, &cndnum, &dist);

    clatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm_gen,
           kl, ku, "Z", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        return;
    }

    INT zerot = (imat >= 2 && imat <= 4);
    izero = -1;
    if (zerot) {
        INT minmn = (m < n) ? m : n;
        if (imat == 2) {
            izero = 0;
        } else if (imat == 3) {
            izero = minmn - 1;
        } else {
            izero = (minmn - 1) / 2;
        }
        INT ioff = izero * lda;
        INT i1 = (ku - izero > 0) ? ku - izero : 0;
        INT i2 = (ku + m - izero < kl + ku + 1) ? ku + m - izero : kl + ku + 1;
        if (imat < 4) {
            for (INT i = i1; i < i2; i++) {
                ws->A[ioff + i] = ZERO;
            }
        } else {
            for (INT j = izero; j < n; j++) {
                INT ji1 = (ku - j > 0) ? ku - j : 0;
                INT ji2 = (ku + m - j < kl + ku + 1) ? ku + m - j : kl + ku + 1;
                for (INT i = ji1; i < ji2; i++) {
                    ws->A[j * lda + i] = ZERO;
                }
            }
        }
    }

    if (m > 0 && n > 0) {
        for (INT j = 0; j < n; j++) {
            for (INT i = 0; i < kl + ku + 1; i++) {
                ws->AFAC[(kl + i) + j * ldafac] = ws->A[i + j * lda];
            }
        }
    }

    cgbtrf(m, n, kl, ku, ws->AFAC, ldafac, ws->IPIV, &info);

    if (izero >= 0) {
        assert_true(info >= 0);
    } else {
        assert_int_equal(info, 0);
    }
    trfcon = (info != 0);

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    cgbt01(m, n, kl, ku, ws->A, lda, ws->AFAC, ldafac,
           ws->IPIV, ws->WORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    if (inb > 0 || m != n) {
        goto test7;
    }

    anormo = clangb("O", n, kl, ku, ws->A, lda, ws->RWORK);
    anormi = clangb("I", n, kl, ku, ws->A, lda, ws->RWORK);

    if (info == 0) {
        INT ldb_inv = (n > 1) ? n : 1;
        claset("F", n, n, CMPLXF(ZERO, ZERO), CMPLXF(ONE, ZERO), ws->WORK, ldb_inv);

        cgbtrs("N", n, kl, ku, n, ws->AFAC, ldafac, ws->IPIV, ws->WORK, ldb_inv, &info);

        f32 ainvnm = clange("O", n, n, ws->WORK, ldb_inv, ws->RWORK);
        if (anormo <= ZERO || ainvnm <= ZERO) {
            rcondo = ONE;
        } else {
            rcondo = (ONE / anormo) / ainvnm;
        }

        ainvnm = clange("I", n, n, ws->WORK, ldb_inv, ws->RWORK);
        if (anormi <= ZERO || ainvnm <= ZERO) {
            rcondi = ONE;
        } else {
            rcondi = (ONE / anormi) / ainvnm;
        }
    } else {
        trfcon = 1;
        rcondo = ZERO;
        rcondi = ZERO;
    }

    if (trfcon) {
        goto test7;
    }

    /*
     * TESTs 2-6: Solve tests for each NRHS and each TRANS
     */
    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];
        char xtype = 'N';

        for (INT itran = 0; itran < (INT)NTRAN; itran++) {
            char trans[2] = {TRANSS[itran], '\0'};
            if (itran == 0) {
                rcondc = rcondo;
            } else {
                rcondc = rcondi;
            }

            clarhs("CGB", &xtype, " ", trans, n, n, kl, ku, nrhs,
                   ws->A, lda, ws->XACT, ldb, ws->B, ldb, &info, rng_state);
            xtype = 'C';

            clacpy("F", n, nrhs, ws->B, ldb, ws->X, ldb);

            /*
             * TEST 2: Solve and compute residual.
             */
            cgbtrs(trans, n, kl, ku, nrhs, ws->AFAC, ldafac, ws->IPIV,
                   ws->X, ldb, &info);
            assert_int_equal(info, 0);

            clacpy("F", n, nrhs, ws->B, ldb, ws->WORK, ldb);
            cgbt02(trans, m, n, kl, ku, nrhs, ws->A, lda, ws->X, ldb,
                   ws->WORK, ldb, ws->RWORK, &result[1]);

            /*
             * TEST 3: Check solution from generated exact solution.
             */
            cget04(n, nrhs, ws->X, ldb, ws->XACT, ldb, rcondc, &result[2]);

            /*
             * TESTs 4, 5, 6: Use iterative refinement.
             */
            cgbrfs(trans, n, kl, ku, nrhs, ws->A, lda, ws->AFAC, ldafac,
                   ws->IPIV, ws->B, ldb, ws->X, ldb,
                   ws->FERR, ws->BERR, ws->WORK, ws->RWORK, &info);
            assert_int_equal(info, 0);

            cget04(n, nrhs, ws->X, ldb, ws->XACT, ldb, rcondc, &result[3]);

            f32 reslts[2];
            cgbt05(trans, n, kl, ku, nrhs, ws->A, lda, ws->B, ldb,
                   ws->X, ldb, ws->XACT, ldb, ws->FERR, ws->BERR, reslts);
            result[4] = reslts[0];
            result[5] = reslts[1];

            for (INT k = 1; k < 6; k++) {
                assert_residual_below(result[k], THRESH);
            }
        }
    }

    /*
     * TEST 7: Get an estimate of RCOND = 1/CNDNUM.
     */
test7:
    if (n <= 0 || m != n || inb > 0) {
        return;
    }

    for (INT itran = 0; itran < 2; itran++) {
        f32 anorm_est;
        char norm[2];
        if (itran == 0) {
            anorm_est = anormo;
            rcondc = rcondo;
            norm[0] = 'O';
        } else {
            anorm_est = anormi;
            rcondc = rcondi;
            norm[0] = 'I';
        }
        norm[1] = '\0';

        cgbcon(norm, n, kl, ku, ws->AFAC, ldafac, ws->IPIV,
               anorm_est, &rcond, ws->WORK, ws->RWORK, &info);
        assert_int_equal(info, 0);

        result[6] = sget06(rcond, rcondc);
        assert_residual_below(result[6], THRESH);
    }
}

static void test_zchkgb(void** state)
{
    zchkgb_params_t* params = (zchkgb_params_t*)*state;
    set_test_context(params->name);
    run_zchkgb_single(params->m, params->n, params->kl, params->ku, params->imat, params->inb);
    clear_test_context();
}

static void get_klval(INT m, INT klval[NBW])
{
    klval[0] = 0;
    klval[1] = m + (m + 1) / 4;
    klval[2] = (3 * m - 1) / 4;
    klval[3] = (m + 1) / 4;
}

static void get_kuval(INT n, INT kuval[NBW])
{
    kuval[0] = 0;
    kuval[1] = n + (n + 1) / 4;
    kuval[2] = (3 * n - 1) / 4;
    kuval[3] = (n + 1) / 4;
}

int main(void)
{
    INT test_count = 0;
    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];
        INT klval[NBW];
        get_klval(m, klval);
        INT nkl = (m + 1 < NBW) ? m + 1 : NBW;

        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];
            INT kuval[NBW];
            get_kuval(n, kuval);
            INT nku = (n + 1 < NBW) ? n + 1 : NBW;
            INT nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (INT ikl = 0; ikl < nkl; ikl++) {
                INT kl = klval[ikl];
                if (kl > m - 1 && m > 0) kl = m - 1;
                if (kl < 0) kl = 0;

                for (INT iku = 0; iku < nku; iku++) {
                    INT ku = kuval[iku];
                    if (ku > n - 1 && n > 0) ku = n - 1;
                    if (ku < 0) ku = 0;

                    for (INT imat = 1; imat <= nimat; imat++) {
                        INT zerot = (imat >= 2 && imat <= 4);
                        INT minmn = (m < n) ? m : n;
                        if (zerot && minmn < imat - 1) {
                            continue;
                        }

                        for (INT inb = 0; inb < (INT)NNB; inb++) {
                            test_count++;
                        }
                    }
                }
            }
        }
    }

    struct CMUnitTest* tests = malloc(test_count * sizeof(struct CMUnitTest));
    zchkgb_params_t* params = malloc(test_count * sizeof(zchkgb_params_t));

    if (!tests || !params) {
        fprintf(stderr, "Failed to allocate test arrays\n");
        return 1;
    }

    INT idx = 0;
    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];
        INT klval[NBW];
        get_klval(m, klval);
        INT nkl = (m + 1 < NBW) ? m + 1 : NBW;

        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];
            INT kuval[NBW];
            get_kuval(n, kuval);
            INT nku = (n + 1 < NBW) ? n + 1 : NBW;
            INT nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (INT ikl = 0; ikl < nkl; ikl++) {
                INT kl = klval[ikl];
                if (kl > m - 1 && m > 0) kl = m - 1;
                if (kl < 0) kl = 0;

                for (INT iku = 0; iku < nku; iku++) {
                    INT ku = kuval[iku];
                    if (ku > n - 1 && n > 0) ku = n - 1;
                    if (ku < 0) ku = 0;

                    for (INT imat = 1; imat <= nimat; imat++) {
                        INT zerot = (imat >= 2 && imat <= 4);
                        INT minmn = (m < n) ? m : n;
                        if (zerot && minmn < imat - 1) {
                            continue;
                        }

                        for (INT inb = 0; inb < (INT)NNB; inb++) {
                            INT nb = NBVAL[inb];
                            params[idx].m = m;
                            params[idx].n = n;
                            params[idx].kl = kl;
                            params[idx].ku = ku;
                            params[idx].imat = imat;
                            params[idx].inb = inb;
                            snprintf(params[idx].name, sizeof(params[idx].name),
                                    "zchkgb_m%d_n%d_kl%d_ku%d_type%d_nb%d_%d",
                                    m, n, kl, ku, imat, nb, inb);

                            tests[idx].name = params[idx].name;
                            tests[idx].test_func = test_zchkgb;
                            tests[idx].setup_func = NULL;
                            tests[idx].teardown_func = NULL;
                            tests[idx].initial_state = &params[idx];

                            idx++;
                        }
                    }
                }
            }
        }
    }

    INT result = _cmocka_run_group_tests("zchkgb", tests, idx,
                                         group_setup, group_teardown);

    free(tests);
    free(params);
    return result;
}
