/**
 * @file test_dchktz.c
 * @brief Comprehensive test suite for DTZRZF (trapezoidal RZ factorization).
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchktz.f to C using CMocka.
 * Tests DTZRZF for various matrix types.
 *
 * Test structure from dchktz.f:
 *   TEST 1: norm(svd(R) - svd(A)) / (||svd(A)|| * eps * max(M,N)) via dqrt12
 *   TEST 2: norm(A - R*Q) / (||A|| * eps * max(M,N)) via drzt01
 *   TEST 3: norm(Q'*Q - I) / (eps * max(M,N)) via drzt02
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50 (only tests M <= N)
 *   Matrix types: 0-2 (zero, one small SV, exponential distribution)
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

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  3
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
    char name[64];
} dchktz_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Working matrix (NMAX x NMAX) */
    f64* COPYA;  /* Copy of original matrix (NMAX x NMAX) */
    f64* S;      /* Singular values (NMAX) */
    f64* TAU;    /* Scalar factors of elementary reflectors (NMAX) */
    f64* WORK;   /* General workspace */
} dchktz_workspace_t;

static dchktz_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchktz_workspace_t));
    if (!g_workspace) return -1;

    /* Workspace size: n*n + 4*m + n + m*n + 2*min(m,n) + 4*n */
    INT lwork = NMAX * NMAX * 2 + 10 * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->COPYA = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->S = malloc(NMAX * sizeof(f64));
    g_workspace->TAU = malloc(NMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));

    if (!g_workspace->A || !g_workspace->COPYA || !g_workspace->S ||
        !g_workspace->TAU || !g_workspace->WORK) {
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
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full dchktz test battery for a single (m, n, imode) combination.
 *
 * Matrix types (MODE = IMODE):
 *   0: zero matrix
 *   1: one small singular value
 *   2: exponential distribution of singular values
 */
static void run_dchktz_single(INT m, INT n, INT imode)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    dchktz_workspace_t* ws = g_workspace;

    INT info;
    INT lda = (m > 1) ? m : 1;
    INT lwork = NMAX * NMAX * 2 + 10 * NMAX;
    INT mnmin = (m < n) ? m : n;
    f64 eps = dlamch("E");
    f64 result[NTESTS];
    char ctx[128];

    /* Seed based on (m, n, imode) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imode));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Generate test matrix based on MODE */
    INT mode = imode;  /* 0, 1, or 2 */
    if (mode == 0) {
        /* Zero matrix */
        dlaset("F", m, n, ZERO, ZERO, ws->A, lda);
        for (INT i = 0; i < mnmin; i++) {
            ws->S[i] = ZERO;
        }
    } else {
        /* Generate matrix with specified singular value distribution */
        /* mode 1: one small singular value, mode 2: exponential distribution */
        dlatms(m, n, "U",
               "N", ws->S, mode,
               ONE / eps, ONE, m, n, "N", ws->A, lda, ws->WORK, &info,
               rng_state);

        /* Reduce to upper trapezoidal form using QR factorization */
        dgeqr2(m, n, ws->A, lda, ws->WORK, &ws->WORK[mnmin], &info);

        /* Zero out below the diagonal */
        if (m > 1) {
            dlaset("L", m - 1, n, ZERO, ZERO, &ws->A[1], lda);
        }

        /* Sort singular values in decreasing order */
        dlaord("D", mnmin, ws->S, 1);
    }

    /* Save A */
    dlacpy("A", m, n, ws->A, lda, ws->COPYA, lda);

    /* Call DTZRZF to reduce the upper trapezoidal matrix to upper triangular form */
    dtzrzf(m, n, ws->A, lda, ws->TAU, ws->WORK, lwork, &info);

    /* TEST 1: norm(svd(R) - svd(A)) / (||svd(A)|| * eps * max(M,N))
     * R is the m-by-m upper triangular factor from DTZRZF (stored in A(0:m-1, 0:m-1)) */
    snprintf(ctx, sizeof(ctx), "m=%d n=%d type=%d TEST 1 (SVD comparison)", m, n, imode);
    set_test_context(ctx);
    result[0] = dqrt12(m, m, ws->A, lda, ws->S, ws->WORK, lwork);
    assert_residual_below(result[0], THRESH);

    /* TEST 2: norm(A - R*Q) / (||A|| * eps * max(M,N)) */
    snprintf(ctx, sizeof(ctx), "m=%d n=%d type=%d TEST 2 (factorization)", m, n, imode);
    set_test_context(ctx);
    result[1] = drzt01(m, n, ws->COPYA, ws->A, lda, ws->TAU, ws->WORK, lwork);
    assert_residual_below(result[1], THRESH);

    /* TEST 3: norm(Q'*Q - I) / (eps * max(M,N)) */
    snprintf(ctx, sizeof(ctx), "m=%d n=%d type=%d TEST 3 (orthogonality)", m, n, imode);
    set_test_context(ctx);
    result[2] = drzt02(m, n, ws->A, lda, ws->TAU, ws->WORK, lwork);
    assert_residual_below(result[2], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchktz_single based on prestate.
 */
static void test_dchktz_case(void** state)
{
    dchktz_params_t* params = *state;
    run_dchktz_single(params->m, params->n, params->imode);
}

/*
 * Generate all parameter combinations where M <= N.
 */

#define MAX_TESTS (NM * NN * NTYPES)

static dchktz_params_t g_params[MAX_TESTS];
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

            /* DTZRZF only applies when M <= N (underdetermined systems) */
            if (m > n) {
                continue;
            }

            for (INT imode = 0; imode < NTYPES; imode++) {
                dchktz_params_t* p = &g_params[g_num_tests];
                p->m = m;
                p->n = n;
                p->imode = imode;
                snprintf(p->name, sizeof(p->name), "dchktz_m%d_n%d_type%d", m, n, imode);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dchktz_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }
}

int main(void)
{
    build_test_array();
    return _cmocka_run_group_tests("dchktz", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
