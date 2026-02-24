/**
 * @file test_dchkq3.c
 * @brief Comprehensive test suite for DGEQP3 (QR with column pivoting).
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkq3.f to C using CMocka.
 * Tests DGEQP3 for various matrix types and block sizes.
 *
 * Test structure from dchkq3.f:
 *   TEST 1: norm(svd(R) - svd(A)) / (||svd(A)|| * eps * max(M,N)) via dqrt12
 *   TEST 2: norm(A*P - Q*R) / (||A|| * eps * max(M,N)) via dqpt01
 *   TEST 3: norm(Q'*Q - I) / (eps * M) via dqrt11
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NB values: 1, 3, 3, 3, 20
 *   Matrix types: 1-6
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
static const INT NBVAL[] = {1, 3, 3, 3, 20};
static const INT NXVAL[] = {1, 0, 5, 9, 1};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPES  6
#define NTESTS  3
#define THRESH  30.0
#define NMAX    50

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
    INT imode;
    INT inb;
    char name[64];
} dchkq3_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Working matrix (NMAX x NMAX) */
    f64* COPYA;  /* Copy of original matrix (NMAX x NMAX) */
    f64* S;      /* Singular values (NMAX) */
    f64* TAU;    /* Scalar factors of elementary reflectors (NMAX) */
    f64* WORK;   /* General workspace */
    INT* IWORK;     /* Integer workspace (2*NMAX) */
} dchkq3_workspace_t;

static dchkq3_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkq3_workspace_t));
    if (!g_workspace) return -1;

    /* Workspace size: m*max(m,n) + 4*min(m,n) + max(m,n) + m*n + 2*min(m,n) + 4*n */
    INT lwork = NMAX * NMAX * 2 + 10 * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->COPYA = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->S = malloc(NMAX * sizeof(f64));
    g_workspace->TAU = malloc(NMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->COPYA || !g_workspace->S ||
        !g_workspace->TAU || !g_workspace->WORK || !g_workspace->IWORK) {
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
        free(g_workspace->COPYA);
        free(g_workspace->S);
        free(g_workspace->TAU);
        free(g_workspace->WORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full dchkq3 test battery for a single (m, n, imode, inb) combination.
 *
 * Matrix types:
 *   1: zero matrix
 *   2: one small singular value
 *   3: geometric distribution of singular values
 *   4: first n/2 columns fixed
 *   5: last n/2 columns fixed
 *   6: every second column fixed
 */
static void run_dchkq3_single(INT m, INT n, INT imode, INT inb)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    dchkq3_workspace_t* ws = g_workspace;

    INT info;
    INT lda = (m > 1) ? m : 1;
    INT lwork = NMAX * NMAX * 2 + 10 * NMAX;
    INT mnmin = (m < n) ? m : n;
    f64 eps = dlamch("E");
    f64 result[NTESTS];
    char ctx[128];

    /* Set block size and crossover point */
    INT nb = NBVAL[inb];
    INT nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Seed based on (m, n, imode) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imode));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Initialize IWORK (first N elements are pivot indicators) */
    for (INT i = 0; i < n; i++) {
        ws->IWORK[i] = 0;
    }

    /* Determine MODE for dlatms based on IMODE */
    INT mode = imode;
    if (imode > 3) {
        mode = 1;
    }

    /* Generate test matrix */
    if (imode == 1) {
        /* Zero matrix */
        dlaset("F", m, n, ZERO, ZERO, ws->COPYA, lda);
        for (INT i = 0; i < mnmin; i++) {
            ws->S[i] = ZERO;
        }
    } else {
        /* Generate matrix with specified singular value distribution */
        dlatms(m, n, "U",
               "N", ws->S, mode, ONE / eps, ONE,
               m, n, "N", ws->COPYA, lda, ws->WORK, &info, rng_state);

        /* For imode 4-6, set column fixing indicators */
        if (imode >= 4) {
            INT ilow, ihigh, istep;
            if (imode == 4) {
                /* First n/2 columns fixed */
                ilow = 0;
                istep = 1;
                ihigh = (n / 2 > 1) ? n / 2 : 1;
            } else if (imode == 5) {
                /* Last n/2 columns fixed */
                ilow = (n / 2 > 1) ? n / 2 : 1;
                istep = 1;
                ihigh = n;
            } else {
                /* Every second column fixed */
                ilow = 0;
                istep = 2;
                ihigh = n;
            }
            for (INT i = ilow; i < ihigh; i += istep) {
                ws->IWORK[i] = 1;
            }
        }

        /* Sort singular values in decreasing order */
        dlaord("D", mnmin, ws->S, 1);
    }

    /* Get working copy of COPYA into A and copy pivot indicators */
    dlacpy("A", m, n, ws->COPYA, lda, ws->A, lda);
    for (INT i = 0; i < n; i++) {
        ws->IWORK[n + i] = ws->IWORK[i];
    }

    /* Workspace size for DGEQP3 */
    INT lw = (2 * n + nb * (n + 1) > 1) ? (2 * n + nb * (n + 1)) : 1;

    /* Compute QR factorization with pivoting */
    dgeqp3(m, n, ws->A, lda, &ws->IWORK[n], ws->TAU, ws->WORK, lw, &info);

    /* TEST 1: norm(svd(R) - svd(A)) / (||svd(A)|| * eps * max(M,N)) */
    snprintf(ctx, sizeof(ctx), "m=%d n=%d type=%d nb=%d TEST 1 (SVD comparison)", m, n, imode, nb);
    set_test_context(ctx);
    result[0] = dqrt12(m, n, ws->A, lda, ws->S, ws->WORK, lwork);
    assert_residual_below(result[0], THRESH);

    /* TEST 2: norm(A*P - Q*R) / (||A|| * eps * max(M,N)) */
    snprintf(ctx, sizeof(ctx), "m=%d n=%d type=%d nb=%d TEST 2 (factorization)", m, n, imode, nb);
    set_test_context(ctx);
    result[1] = dqpt01(m, n, mnmin, ws->COPYA, ws->A, lda, ws->TAU,
                       &ws->IWORK[n], ws->WORK, lwork);
    assert_residual_below(result[1], THRESH);

    /* TEST 3: norm(Q'*Q - I) / (eps * M) */
    snprintf(ctx, sizeof(ctx), "m=%d n=%d type=%d nb=%d TEST 3 (orthogonality)", m, n, imode, nb);
    set_test_context(ctx);
    result[2] = dqrt11(m, mnmin, ws->A, lda, ws->TAU, ws->WORK, lwork);
    assert_residual_below(result[2], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkq3_single based on prestate.
 */
static void test_dchkq3_case(void** state)
{
    dchkq3_params_t* params = *state;
    run_dchkq3_single(params->m, params->n, params->imode, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NTYPES * NNB = 7 * 7 * 6 * 5 = 1470 tests
 */

#define MAX_TESTS (NM * NN * NTYPES * NNB)

static dchkq3_params_t g_params[MAX_TESTS];
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

            for (INT imode = 1; imode <= NTYPES; imode++) {
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    dchkq3_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imode = imode;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchkq3_m%d_n%d_type%d_nb%d_%d",
                             m, n, imode, nb, inb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchkq3_case;
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
    return _cmocka_run_group_tests("dchkq3", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
