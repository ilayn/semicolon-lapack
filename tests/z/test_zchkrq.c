/**
 * @file test_zchkrq.c
 * @brief Comprehensive test suite for RQ factorization (ZRQ) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkrq.f to C using CMocka.
 * Tests ZGERQF, ZUNGRQ, and ZUNMRQ.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkrq.f:
 *   TEST 1-2: RQ factorization via zrqt01 (norm(R - A*Q'), norm(I - Q*Q'))
 *   TEST 3-6: ZUNMRQ tests via zrqt03 (applying Q from left/right with trans)
 *   TEST 7: Least squares solve via zgerqs and zget02
 *
 * Parameters from ztest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   Matrix types: 1-8
 *   K values: MINMN, 0, 1, MINMN/2
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
static const INT NXVAL[] = {1, 0, 5, 9, 1};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPES  8
#define NTESTS  7
#define THRESH  30.0
#define NMAX    50
#define NSMAX   15

typedef struct {
    INT m;
    INT n;
    INT imat;
    INT inb;
    char name[64];
} zchkrq_params_t;

typedef struct {
    c128* A;
    c128* AF;
    c128* Q;
    c128* R;
    c128* C;
    c128* CC;
    c128* B;
    c128* X;
    c128* XACT;
    c128* TAU;
    c128* WORK;
    f64* RWORK;
    f64* D;
    INT* IWORK;
} zchkrq_workspace_t;

static zchkrq_workspace_t* g_workspace = NULL;

static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkrq_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AF = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->Q = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->R = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->C = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->CC = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->TAU = malloc(NMAX * sizeof(c128));
    g_workspace->WORK = malloc(lwork * sizeof(c128));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->IWORK = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AF || !g_workspace->Q ||
        !g_workspace->R || !g_workspace->C || !g_workspace->CC ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->TAU || !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->D || !g_workspace->IWORK) {
        return -1;
    }

    return 0;
}

static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->AF);
        free(g_workspace->Q);
        free(g_workspace->R);
        free(g_workspace->C);
        free(g_workspace->CC);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->TAU);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->D);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

static void run_zchkrq_single(INT m, INT n, INT imat, INT inb_idx)
{
    const f64 ZERO = 0.0;
    zchkrq_workspace_t* ws = g_workspace;

    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info;
    INT lda = NMAX;
    INT lwork = NMAX * NMAX;
    INT minmn = (m < n) ? m : n;
    f64 result[NTESTS];
    char ctx[128];

    INT nb = NBVAL[inb_idx];
    INT nx = NXVAL[inb_idx];
    xlaenv(1, nb);
    xlaenv(3, nx);

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imat));

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    zlatb4("ZRQ", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    zlatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    INT kval[4];
    kval[0] = minmn;
    kval[1] = 0;
    kval[2] = 1;
    kval[3] = minmn / 2;

    INT nk;
    if (minmn == 0) {
        nk = 1;
    } else if (minmn == 1) {
        nk = 2;
    } else if (minmn <= 3) {
        nk = 3;
    } else {
        nk = 4;
    }

    for (INT ik = 0; ik < nk; ik++) {
        INT k = kval[ik];

        if (ik == 0) {
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1-2 (RQ factorization)", m, n, imat, k);
            set_test_context(ctx);
            zrqt01(m, n, ws->A, ws->AF, ws->Q, ws->R, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            assert_residual_below(result[1], THRESH);
        } else if (m <= n) {
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1-2 (ZUNGRQ)", m, n, imat, k);
            set_test_context(ctx);
            zrqt02(m, n, k, ws->A, ws->AF, ws->Q, ws->R, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            assert_residual_below(result[1], THRESH);
        }

        if (m >= k) {
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 3-6 (ZUNMRQ)", m, n, imat, k);
            set_test_context(ctx);
            zrqt03(m, n, k, ws->AF, ws->C, ws->CC, ws->Q, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);
            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);

            if (k == m && inb_idx == 0) {
                INT nrhs = NSVAL[0];

                snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d nrhs=%d TEST 7 (ZGERQS)", m, n, imat, k, nrhs);
                set_test_context(ctx);

                zlarhs("ZRQ", "N", "F", "N", m, n, 0, 0, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

                /* Fortran: ZLACPY('Full', M, NRHS, B, LDA, X(N-M+1), LDA)
                 * X(N-M+1) is 1-based offset into flat array = 0-based offset n-m */
                zlacpy("F", m, nrhs, ws->B, lda, ws->X + (n - m), lda);

                zgerqs(m, n, nrhs, ws->AF, lda, ws->TAU, ws->X,
                       lda, ws->WORK, lwork, &info);
                assert_int_equal(info, 0);

                zlacpy("F", m, nrhs, ws->B, lda, ws->WORK, lda);
                zget02("N", m, n, nrhs, ws->A, lda, ws->X, lda,
                       ws->WORK, lda, ws->RWORK, &result[6]);

                assert_residual_below(result[6], THRESH);
            }
        }
    }

    clear_test_context();
}

static void test_zchkrq_case(void** state)
{
    zchkrq_params_t* params = *state;
    run_zchkrq_single(params->m, params->n, params->imat, params->inb);
}

#define MAX_TESTS (NM * NN * NTYPES * NNB)

static zchkrq_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];

        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];

            for (INT imat = 1; imat <= NTYPES; imat++) {
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    zchkrq_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imat = imat;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "zchkrq_m%d_n%d_type%d_nb%d_%d",
                             m, n, imat, nb, inb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchkrq_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("zchkrq", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
