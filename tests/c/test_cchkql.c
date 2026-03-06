/**
 * @file test_cchkql.c
 * @brief Comprehensive test suite for QL factorization (CQL) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkql.f to C using CMocka.
 * Tests CGEQLF, CUNGQL, and CUNMQL.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkql.f:
 *   TEST 1-2: QL factorization via cqlt01 (norm(L - Q'*A), norm(I - Q'*Q))
 *   TEST 3-6: CUNMQL tests via cqlt03 (applying Q from left/right with trans)
 *   TEST 7: Least squares solve via cgeqls and cget02
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
#include "semicolon_lapack_complex_single.h"
#include <string.h>
#include <stdio.h>
/* Test parameters from ztest.in */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values for least squares */
static const INT NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from ztest.in */
static const INT NXVAL[] = {1, 0, 5, 9, 1};   /* Crossover points from ztest.in */

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPES  8
#define NTESTS  7   /* We test 1-2 (cqlt01), 3-6 (cqlt03), 7 (cgeqls) */
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */


/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT m;
    INT n;
    INT imat;
    INT inb;    /* Index into NBVAL[] */
    char name[64];
} zchkql_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    c64* A;      /* Original matrix (NMAX x NMAX) */
    c64* AF;     /* Factored matrix (NMAX x NMAX) */
    c64* Q;      /* Unitary matrix Q (NMAX x NMAX) */
    c64* L;      /* Lower triangular L (NMAX x NMAX) */
    c64* C;      /* Workspace for cqlt03 (NMAX x NMAX) */
    c64* CC;     /* Another workspace for cqlt03 (NMAX x NMAX) */
    c64* B;      /* Right-hand side (NMAX x NSMAX) */
    c64* X;      /* Solution (NMAX x NSMAX) */
    c64* XACT;   /* Exact solution (NMAX x NSMAX) */
    c64* TAU;    /* Scalar factors of elementary reflectors */
    c64* WORK;   /* General workspace */
    f32* RWORK;   /* Real workspace */
    f32* D;       /* Singular values for clatms */
    INT* IWORK;   /* Integer workspace */
} zchkql_workspace_t;

static zchkql_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkql_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->AF = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->Q = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->L = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->C = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->CC = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->TAU = malloc(NMAX * sizeof(c64));
    g_workspace->WORK = malloc(lwork * sizeof(c64));
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
 * Run the full zchkql test battery for a single (m, n, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Unlike zchkge/zchkpo/zchksy, ALL tests run for each NB value because
 * the QL factorization and Q operations are all affected by blocking.
 */
static void run_zchkql_single(INT m, INT n, INT imat, INT inb)
{
    const f32 ZERO = 0.0f;
    zchkql_workspace_t* ws = g_workspace;

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
    for (INT i = 0; i < NTESTS; i++) {
        result[i] = ZERO;
    }

    /* Get matrix parameters for this type */
    clatb4("CQL", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    clatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
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
             * TEST 1-2: Test CGEQLF via cqlt01
             * Computes:
             *   result[0] = norm(L - Q'*A) / (M * norm(A) * eps)
             *   result[1] = norm(I - Q'*Q) / (M * eps)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1 (QL factorization L)", m, n, imat, k);
            set_test_context(ctx);
            cqlt01(m, n, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 2 (QL factorization Q)", m, n, imat, k);
            set_test_context(ctx);
            assert_residual_below(result[1], THRESH);
        } else if (m >= n) {
            /*
             * TEST 1-2: Test CUNGQL via cqlt02
             * Uses factorization from cqlt01
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1-2 (CUNGQL)", m, n, imat, k);
            set_test_context(ctx);
            cqlt02(m, n, k, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            assert_residual_below(result[1], THRESH);
        }

        if (m >= k) {
            /*
             * TEST 3-6: Test CUNMQL via cqlt03
             * Tests Q from left/right with 'N'/'C' conjugate transpose options
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 3-6 (CUNMQL)", m, n, imat, k);
            set_test_context(ctx);
            cqlt03(m, n, k, ws->AF, ws->C, ws->CC, ws->Q, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);
            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);

            /*
             * TEST 7: If M >= N and K == N, test CGEQLS least squares solve
             * Only test once per (M, N, IMAT) when INB == 1
             */
            if (k == n && m >= n && n > 0 && inb == 0) {
                INT nrhs = NSVAL[0];  /* Use first NRHS value */

                snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d nrhs=%d TEST 7 (CGEQLS)", m, n, imat, k, nrhs);
                set_test_context(ctx);

                /* Generate right-hand side */
                clarhs("CQL", "N", "F", "N", m, n, 0, 0, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

                clacpy("F", m, nrhs, ws->B, lda, ws->X, lda);

                /* Use the factorization already in AF */
                cgeqls(m, n, nrhs, ws->AF, lda, ws->TAU, ws->X, lda,
                       ws->WORK, lwork, &info);
                assert_int_equal(info, 0);

                /* Check residual: X is in X(m-n:m-1, :) */
                clacpy("F", m, nrhs, ws->B, lda, ws->WORK, lda);
                cget02("N", m, n, nrhs, ws->A, lda, &ws->X[(m - n)], lda,
                       ws->WORK, lda, ws->RWORK, &result[6]);

                assert_residual_below(result[6], THRESH);
            }
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_zchkql_single based on prestate.
 */
static void test_zchkql_case(void** state)
{
    zchkql_params_t* params = *state;
    run_zchkql_single(params->m, params->n, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NTYPES * NNB = 7 * 7 * 8 * 5 = 1960 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NM * NN * NTYPES * NNB)

static zchkql_params_t g_params[MAX_TESTS];
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
                    zchkql_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imat = imat;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "zchkql_m%d_n%d_type%d_nb%d_%d",
                             m, n, imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchkql_case;
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
    return _cmocka_run_group_tests("zchkql", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
