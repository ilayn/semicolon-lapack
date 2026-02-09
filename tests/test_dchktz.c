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
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  3
#define NTESTS  3
#define THRESH  30.0
#define NMAX    50

/* Routines under test */
extern void dtzrzf(const int m, const int n, double* A, const int lda,
                   double* tau, double* work, const int lwork, int* info);
extern void dgeqr2(const int m, const int n, double* A, const int lda,
                   double* tau, double* work, int* info);

/* Verification routines */
extern double dqrt12(const int m, const int n, const double* A, const int lda,
                     const double* S, double* work, const int lwork);
extern double drzt01(const int m, const int n, const double* A, const double* AF,
                     const int lda, const double* tau, double* work, const int lwork);
extern double drzt02(const int m, const int n, const double* AF, const int lda,
                     const double* tau, double* work, const int lwork);

/* Matrix generation */
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, double* d, const int mode, const double cond,
                   const double dmax, const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info,
                   uint64_t state[static 4]);
extern void dlaord(const char* job, const int n, double* X, const int incx);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern double dlamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int imode;
    char name[64];
} dchktz_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    double* A;      /* Working matrix (NMAX x NMAX) */
    double* COPYA;  /* Copy of original matrix (NMAX x NMAX) */
    double* S;      /* Singular values (NMAX) */
    double* TAU;    /* Scalar factors of elementary reflectors (NMAX) */
    double* WORK;   /* General workspace */
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
    int lwork = NMAX * NMAX * 2 + 10 * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->COPYA = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->S = malloc(NMAX * sizeof(double));
    g_workspace->TAU = malloc(NMAX * sizeof(double));
    g_workspace->WORK = malloc(lwork * sizeof(double));

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
static void run_dchktz_single(int m, int n, int imode)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    dchktz_workspace_t* ws = g_workspace;

    int info;
    int lda = (m > 1) ? m : 1;
    int lwork = NMAX * NMAX * 2 + 10 * NMAX;
    int mnmin = (m < n) ? m : n;
    double eps = dlamch("E");
    double result[NTESTS];
    char ctx[128];

    /* Seed based on (m, n, imode) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imode));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Generate test matrix based on MODE */
    int mode = imode;  /* 0, 1, or 2 */
    if (mode == 0) {
        /* Zero matrix */
        dlaset("F", m, n, ZERO, ZERO, ws->A, lda);
        for (int i = 0; i < mnmin; i++) {
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
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (int im = 0; im < (int)NM; im++) {
        int m = MVAL[im];

        for (int in = 0; in < (int)NN; in++) {
            int n = NVAL[in];

            /* DTZRZF only applies when M <= N (underdetermined systems) */
            if (m > n) {
                continue;
            }

            for (int imode = 0; imode < NTYPES; imode++) {
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
