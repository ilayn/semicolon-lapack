/**
 * @file test_schklq.c
 * @brief Comprehensive test suite for LQ factorization (SLQ) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchklq.f to C using CMocka.
 * Tests SGELQF, SORGLQ, and SORMLQ.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchklq.f:
 *   TEST 1-2: LQ factorization via slqt01 (norm(L - A*Q'), norm(I - Q*Q'))
 *   TEST 3-6: SORMLQ tests via slqt03 (applying Q from left/right with trans)
 *   TEST 7: Least squares solve via sgels and sget02
 *
 * Parameters from dtest.in:
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
/* Test parameters from dtest.in */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values for least squares */
static const INT NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from dtest.in */
static const INT NXVAL[] = {1, 0, 5, 9, 1};   /* Crossover points from dtest.in */

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPES  8
#define NTESTS  7   /* We test 1-2 (slqt01), 3-6 (slqt03), 7 (sgels) */
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
/* Verification routines */
/* Matrix generation */
/* Utilities */
/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT m;
    INT n;
    INT imat;
    INT inb;    /* Index into NBVAL[] */
    char name[64];
} dchklq_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* A;      /* Original matrix (NMAX x NMAX) */
    f32* AF;     /* Factored matrix (NMAX x NMAX) */
    f32* Q;      /* Orthogonal matrix Q (NMAX x NMAX) */
    f32* L;      /* Lower triangular L (NMAX x NMAX) */
    f32* C;      /* Workspace for slqt03 (NMAX x NMAX) */
    f32* CC;     /* Another workspace for slqt03 (NMAX x NMAX) */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* TAU;    /* Scalar factors of elementary reflectors */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* D;      /* Singular values for slatms */
    INT* IWORK;     /* Integer workspace */
} dchklq_workspace_t;

static dchklq_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchklq_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AF = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->Q = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->L = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->C = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->CC = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->TAU = malloc(NMAX * sizeof(f32));
    g_workspace->WORK = malloc(lwork * sizeof(f32));
    g_workspace->RWORK = malloc(NMAX * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->IWORK = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AF || !g_workspace->Q ||
        !g_workspace->L || !g_workspace->C || !g_workspace->CC ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->TAU || !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->D || !g_workspace->IWORK) {
        return -1;
    }

    return 0;
}

/**
 * Group teardown - free workspace.
 */
static int group_teardown(void** state)
{
    (void)state;
    if (g_workspace) {
        free(g_workspace->A);
        free(g_workspace->AF);
        free(g_workspace->Q);
        free(g_workspace->L);
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

/**
 * Run the full dchklq test battery for a single (m, n, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Unlike dchkge/dchkpo/dchksy, ALL tests run for each NB value because
 * the LQ factorization and Q operations are all affected by blocking.
 */
static void run_dchklq_single(INT m, INT n, INT imat, INT inb)
{
    const f32 ZERO = 0.0f;
    dchklq_workspace_t* ws = g_workspace;

    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    INT info;
    INT lda = NMAX;
    INT lwork = NMAX * NMAX;
    INT minmn = (m < n) ? m : n;
    f32 result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Set block size and crossover point for this test via xlaenv */
    INT nb = NBVAL[inb];
    INT nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Seed based on (m, n, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imat));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    slatb4("SLQ", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    slatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* Set K values to test: MINMN, 0, 1, MINMN/2 */
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
            /*
             * TEST 1-2: Test SGELQF via slqt01
             * Computes:
             *   result[0] = norm(L - A*Q') / (N * norm(A) * eps)
             *   result[1] = norm(I - Q*Q') / (N * eps)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1 (LQ factorization L)", m, n, imat, k);
            set_test_context(ctx);
            slqt01(m, n, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 2 (LQ factorization Q)", m, n, imat, k);
            set_test_context(ctx);
            assert_residual_below(result[1], THRESH);
        } else if (m <= n) {
            /*
             * TEST 1-2: Test SORGLQ via slqt02
             * Uses factorization from slqt01
             * Note: For LQ, we test when m <= n (wide/square matrices)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1-2 (SORGLQ)", m, n, imat, k);
            set_test_context(ctx);
            slqt02(m, n, k, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            assert_residual_below(result[1], THRESH);
        }

        if (m >= k) {
            /*
             * TEST 3-6: Test SORMLQ via slqt03
             * Tests Q from left/right with 'N'/'T' transpose options
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 3-6 (SORMLQ)", m, n, imat, k);
            set_test_context(ctx);
            slqt03(m, n, k, ws->AF, ws->C, ws->CC, ws->Q, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);
            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);

            /*
             * TEST 7: If M <= N and K == M, test SGELS least squares solve
             * Note: For LQ, condition is K == M (underdetermined system)
             * Only test once per (M, N, IMAT) when INB == 1
             */
            if (k == m && m <= n && m > 0 && inb == 0) {
                INT nrhs = NSVAL[0];  /* Use first NRHS value */

                snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d nrhs=%d TEST 7 (SGELS)", m, n, imat, k, nrhs);
                set_test_context(ctx);

                /* Generate right-hand side */
                slarhs("SLQ", "N", "F", "N", m, n, 0, 0, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

                slacpy("F", m, nrhs, ws->B, lda, ws->X, lda);
                /* Use ws->C instead of ws->AF for SGELS - SGELS modifies its input
                 * and we need to preserve AF for subsequent slqt02/slqt03 tests */
                slacpy("F", m, n, ws->A, lda, ws->C, lda);

                sgels("N", m, n, nrhs, ws->C, lda, ws->X, lda,
                      ws->WORK, lwork, &info);
                assert_int_equal(info, 0);

                /* Check residual */
                slacpy("F", m, nrhs, ws->B, lda, ws->WORK, lda);
                sget02("N", m, n, nrhs, ws->A, lda, ws->X, lda,
                       ws->WORK, lda, ws->RWORK, &result[6]);

                assert_residual_below(result[6], THRESH);
            }
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchklq_single based on prestate.
 */
static void test_dchklq_case(void** state)
{
    dchklq_params_t* params = *state;
    run_dchklq_single(params->m, params->n, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NTYPES * NNB = 7 * 7 * 8 * 5 = 1960 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NM * NN * NTYPES * NNB)

static dchklq_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];

        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];

            for (INT imat = 1; imat <= NTYPES; imat++) {
                /* Loop over block sizes */
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    /* Store parameters */
                    dchklq_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imat = imat;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchklq_m%d_n%d_type%d_nb%d_%d",
                             m, n, imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchklq_case;
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
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace.
     * We use _cmocka_run_group_tests directly because the test array
     * is built dynamically and the standard macro uses sizeof() which
     * only works for compile-time array sizes. */
    return _cmocka_run_group_tests("dchklq", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
