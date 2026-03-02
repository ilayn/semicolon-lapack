/**
 * @file test_zchkgt.c
 * @brief Comprehensive test suite for complex general tridiagonal matrix (ZGT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkgt.f to C using CMocka.
 * Tests ZGTTRF, ZGTTRS, ZGTRFS, and ZGTCON.
 *
 * Each (n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkgt.f:
 *   TEST 1: LU factorization residual via zgtt01
 *   TEST 2: Solution residual via zgtt02
 *   TEST 3: Solution accuracy via zget04
 *   TEST 4: Refined solution accuracy via zget04 (after zgtrfs)
 *   TEST 5-6: Error bounds via zgtt05
 *   TEST 7: Condition number via dget06
 *
 * Parameters from ztest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   TRANS values: 'N', 'T', 'C'
 *   Matrix types: 1-12
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>

static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};
static const char TRANSS[] = {'N', 'T', 'C'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTRAN   (sizeof(TRANSS) / sizeof(TRANSS[0]))
#define NTYPES  12
#define NTESTS  7
#define THRESH  30.0
#define NMAX    50
#define NSMAX   15

typedef struct {
    INT n;
    INT imat;
    char name[64];
} zchkgt_params_t;

typedef struct {
    c128* DL;
    c128* D;
    c128* DU;
    c128* DLF;
    c128* DF;
    c128* DUF;
    c128* DU2;
    c128* B;
    c128* X;
    c128* XACT;
    c128* WORK;
    f64* RWORK;
    f64* FERR;
    f64* BERR;
    INT* IPIV;
} zchkgt_workspace_t;

static zchkgt_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkgt_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->DL = malloc(NMAX * sizeof(c128));
    g_workspace->D = malloc(NMAX * sizeof(c128));
    g_workspace->DU = malloc(NMAX * sizeof(c128));
    g_workspace->DLF = malloc(NMAX * sizeof(c128));
    g_workspace->DF = malloc(NMAX * sizeof(c128));
    g_workspace->DUF = malloc(NMAX * sizeof(c128));
    g_workspace->DU2 = malloc(NMAX * sizeof(c128));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->RWORK = malloc((NMAX + 2 * NSMAX) * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
    g_workspace->IPIV = malloc(NMAX * sizeof(INT));

    if (!g_workspace->DL || !g_workspace->D || !g_workspace->DU ||
        !g_workspace->DLF || !g_workspace->DF || !g_workspace->DUF ||
        !g_workspace->DU2 || !g_workspace->B || !g_workspace->X ||
        !g_workspace->XACT || !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->IPIV) {
        return -1;
    }

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->DL);
        free(g_workspace->D);
        free(g_workspace->DU);
        free(g_workspace->DLF);
        free(g_workspace->DF);
        free(g_workspace->DUF);
        free(g_workspace->DU2);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->FERR);
        free(g_workspace->BERR);
        free(g_workspace->IPIV);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void generate_zgt_matrix(INT n, INT imat, c128* DL, c128* D, c128* DU,
                                uint64_t rng_state[static 4], INT* izero)
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    INT m = (n > 1) ? n - 1 : 0;

    if (n <= 0) {
        *izero = -1;
        return;
    }

    zlatb4("ZGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    INT zerot = (imat >= 8 && imat <= 10);
    *izero = -1;

    if (imat >= 1 && imat <= 6) {
        INT lda_band = 3;
        INT max1n = (1 > n) ? 1 : n;
        INT koff_a = 2 - ku;
        INT koff_b = 3 - max1n;
        INT koff = ((koff_a > koff_b) ? koff_a : koff_b) - 1;
        if (koff < 0) koff = 0;
        INT ab_needed = koff + lda_band * n;
        INT ab_size = (n * n > ab_needed) ? n * n : ab_needed;
        c128* AB = calloc(ab_size, sizeof(c128));
        f64* d_sing = malloc(n * sizeof(f64));

        if (!AB || !d_sing) {
            free(AB);
            free(d_sing);
            return;
        }

        zlatms(n, n, &dist, &type, d_sing, mode, cndnum, anorm,
               kl, ku, "Z", AB + koff, lda_band, g_workspace->WORK, &info, rng_state);

        if (info == 0) {
            for (INT i = 0; i < n; i++) {
                D[i] = AB[1 + i * lda_band];
            }
            for (INT i = 0; i < m; i++) {
                DU[i] = AB[0 + (i + 1) * lda_band];
                DL[i] = AB[2 + i * lda_band];
            }
        } else {
            for (INT i = 0; i < n; i++) {
                D[i] = 2.0 * anorm;
            }
            for (INT i = 0; i < m; i++) {
                DL[i] = -anorm * 0.5;
                DU[i] = -anorm * 0.5;
            }
        }

        free(AB);
        free(d_sing);
    } else {
        for (INT i = 0; i < m; i++) {
            DL[i] = CMPLX(rng_uniform_symmetric(rng_state),
                           rng_uniform_symmetric(rng_state));
        }
        for (INT i = 0; i < n; i++) {
            D[i] = CMPLX(rng_uniform_symmetric(rng_state),
                          rng_uniform_symmetric(rng_state));
        }
        for (INT i = 0; i < m; i++) {
            DU[i] = CMPLX(rng_uniform_symmetric(rng_state),
                           rng_uniform_symmetric(rng_state));
        }

        if (anorm != ONE) {
            for (INT i = 0; i < m; i++) {
                DL[i] = DL[i] * anorm;
            }
            for (INT i = 0; i < n; i++) {
                D[i] = D[i] * anorm;
            }
            for (INT i = 0; i < m; i++) {
                DU[i] = DU[i] * anorm;
            }
        }

        if (zerot) {
            if (imat == 8) {
                *izero = 0;
                D[0] = ZERO;
                if (n > 1) {
                    DL[0] = ZERO;
                }
            } else if (imat == 9) {
                *izero = n - 1;
                D[n - 1] = ZERO;
                if (n > 1) {
                    DU[n - 2] = ZERO;
                }
            } else {
                *izero = (n - 1) / 2;
                for (INT j = *izero; j < n; j++) {
                    D[j] = ZERO;
                }
                for (INT j = *izero; j < n - 1; j++) {
                    DL[j] = ZERO;
                }
                INT du_start = (*izero >= 1) ? *izero - 1 : 0;
                for (INT j = du_start; j < n - 1; j++) {
                    DU[j] = ZERO;
                }
            }
        }
    }
}

static void run_zchkgt_single(INT n, INT imat)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    zchkgt_workspace_t* ws = g_workspace;

    INT info, izero;
    INT m = (n > 1) ? n - 1 : 0;
    INT lda = (n > 1) ? n : 1;
    INT trfcon;
    f64 anorm, rcond, rcondc, rcondo, rcondi, ainvnm;
    f64 result[NTESTS];
    char ctx[128];

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + imat));

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    generate_zgt_matrix(n, imat, ws->DL, ws->D, ws->DU, rng_state, &izero);

    memcpy(ws->DLF, ws->DL, m * sizeof(c128));
    memcpy(ws->DF, ws->D, n * sizeof(c128));
    memcpy(ws->DUF, ws->DU, m * sizeof(c128));

    /*
     * TEST 1: Factor A as L*U and compute the ratio
     *         norm(L*U - A) / (n * norm(A) * EPS)
     */
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 1 (factorization)", n, imat);
    set_test_context(ctx);
    zgttrf(n, ws->DLF, ws->DF, ws->DUF, ws->DU2, ws->IPIV, &info);

    if (izero >= 0) {
        assert_true(info >= 0);
    } else {
        assert_int_equal(info, 0);
    }
    trfcon = (info != 0);

    zgtt01(n, ws->DL, ws->D, ws->DU, ws->DLF, ws->DF, ws->DUF,
           ws->DU2, ws->IPIV, ws->WORK, lda, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 7: Condition number estimation (for both 'O' and 'I' norms)
     */
    for (INT itran = 0; itran < 2; itran++) {
        char norm = (itran == 0) ? 'O' : 'I';
        char norm_str[2] = {norm, '\0'};
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 7 (condition norm=%c)", n, imat, norm);
        set_test_context(ctx);
        anorm = zlangt(norm_str, n, ws->DL, ws->D, ws->DU);

        if (!trfcon) {
            ainvnm = ZERO;
            for (INT i = 0; i < n; i++) {
                for (INT j = 0; j < n; j++) {
                    ws->X[j] = ZERO;
                }
                ws->X[i] = ONE;
                char trans_str[2] = {(itran == 0) ? 'N' : 'T', '\0'};
                zgttrs(trans_str, n, 1, ws->DLF, ws->DF, ws->DUF, ws->DU2,
                       ws->IPIV, ws->X, lda, &info);
                f64 sum = ZERO;
                for (INT j = 0; j < n; j++) {
                    sum += fabs(creal(ws->X[j])) + fabs(cimag(ws->X[j]));
                }
                if (sum > ainvnm) ainvnm = sum;
            }

            if (anorm <= ZERO || ainvnm <= ZERO) {
                rcondc = ONE;
            } else {
                rcondc = (ONE / anorm) / ainvnm;
            }
            if (itran == 0) {
                rcondo = rcondc;
            } else {
                rcondi = rcondc;
            }
        } else {
            rcondc = ZERO;
        }

        zgtcon(norm_str, n, ws->DLF, ws->DF, ws->DUF, ws->DU2, ws->IPIV,
               anorm, &rcond, ws->WORK, &info);
        assert_int_equal(info, 0);

        result[6] = dget06(rcond, rcondc);
        assert_residual_below(result[6], THRESH);
    }

    if (trfcon) {
        return;
    }

    /*
     * TESTS 2-6: Solve tests for each NRHS and TRANS
     */
    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                ws->XACT[i + j * lda] = CMPLX(rng_uniform_symmetric(rng_state),
                                               rng_uniform_symmetric(rng_state));
            }
        }

        for (INT itran = 0; itran < (INT)NTRAN; itran++) {
            char trans = TRANSS[itran];
            char trans_str[2] = {trans, '\0'};
            rcondc = (itran == 0) ? rcondo : rcondi;

            zlagtm(trans_str, n, nrhs, ONE, ws->DL, ws->D, ws->DU,
                   ws->XACT, lda, ZERO, ws->B, lda);

            /*
             * TEST 2: Solve op(A) * X = B and compute residual
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d trans=%c TEST 2 (solve)", n, imat, nrhs, trans);
            set_test_context(ctx);
            zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            zgttrs(trans_str, n, nrhs, ws->DLF, ws->DF, ws->DUF, ws->DU2,
                   ws->IPIV, ws->X, lda, &info);
            assert_int_equal(info, 0);

            zlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
            zgtt02(trans_str, n, nrhs, ws->DL, ws->D, ws->DU,
                   ws->X, lda, ws->WORK, lda, &result[1]);
            assert_residual_below(result[1], THRESH);

            /*
             * TEST 3: Check solution from generated exact solution
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d trans=%c TEST 3 (accuracy)", n, imat, nrhs, trans);
            set_test_context(ctx);
            zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            assert_residual_below(result[2], THRESH);

            /*
             * TESTS 4, 5, 6: Iterative refinement
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d trans=%c TEST 4-6 (refinement)", n, imat, nrhs, trans);
            set_test_context(ctx);

            zgtrfs(trans_str, n, nrhs, ws->DL, ws->D, ws->DU,
                   ws->DLF, ws->DF, ws->DUF, ws->DU2, ws->IPIV,
                   ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
                   ws->WORK, ws->RWORK, &info);
            assert_int_equal(info, 0);

            zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
            zgtt05(trans_str, n, nrhs, ws->DL, ws->D, ws->DU,
                   ws->B, lda, ws->X, lda, ws->XACT, lda,
                   ws->FERR, ws->BERR, &result[4]);

            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);
        }
    }

    clear_test_context();
}

static void test_zchkgt_case(void** state)
{
    zchkgt_params_t* params = *state;
    run_zchkgt_single(params->n, params->imat);
}

#define MAX_TESTS (NN * NTYPES)

static zchkgt_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];

        INT nimat = NTYPES;
        if (n <= 0) {
            nimat = 1;
        }

        for (INT imat = 1; imat <= nimat; imat++) {
            zchkgt_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->imat = imat;
            snprintf(p->name, sizeof(p->name), "zchkgt_n%d_type%d", n, imat);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zchkgt_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
        }
    }
}

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("zchkgt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
