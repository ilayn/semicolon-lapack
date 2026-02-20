/**
 * @file test_spotrf.c
 * @brief CMocka test suite for spotf2/spotrf2/spotrf (Cholesky factorization).
 *
 * Tests the Cholesky factorization routines which compute A = U'*U or A = L*L'.
 *
 * Each (n, uplo, imat, routine) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Verification: spot01 computes ||L*L' - A|| / (N * ||A|| * eps)
 *
 * Matrix types tested (from slatb4 for SPO path):
 *   1. Diagonal
 *   2. Random, well-conditioned
 *   3-5. Singular matrices (zero row/column)
 *   6. Random, ill-conditioned
 *   7. Random, very ill-conditioned
 *   8-9. Scaled matrices
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const char UPLOS[] = {'U', 'L'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9
#define NROUTINE 3   /* spotf2, spotrf2, spotrf */
#define THRESH  30.0f
#define NMAX    50

/* Routines under test */
extern void spotf2(const char* uplo, const int n, f32* A,
                   const int lda, int* info);
extern void spotrf2(const char* uplo, const int n, f32* A,
                    const int lda, int* info);
extern void spotrf(const char* uplo, const int n, f32* A,
                   const int lda, int* info);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    int iuplo;   /* 0='U', 1='L' */
    int iroutine; /* 0=spotf2, 1=spotrf2, 2=spotrf */
    char name[80];
} dpotrf_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* A;      /* Original matrix (NMAX x NMAX) */
    f32* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f32* D;      /* Singular values for slatms */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
} dpotrf_workspace_t;

static dpotrf_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dpotrf_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->WORK = malloc(3 * NMAX * sizeof(f32));
    g_workspace->RWORK = malloc(NMAX * sizeof(f32));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->D ||
        !g_workspace->WORK || !g_workspace->RWORK) {
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
        free(g_workspace->AFAC);
        free(g_workspace->D);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the spotrf test for a single (n, uplo, imat, routine) combination.
 */
static void run_dpotrf_single(int n, int iuplo, int imat, int iroutine)
{
    static const char* ROUTINE_NAMES[] = {"dpotf2", "dpotrf2", "dpotrf"};
    dpotrf_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info;
    int lda = (n > 1) ? n : 1;

    /* Get matrix parameters for this type */
    slatb4("SPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric positive definite test matrix */
    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, 5001500150015001ULL + (uint64_t)(n * 10000 + iuplo * 1000 + imat * 10 + iroutine));
    slatms(n, n, &dist, sym_str, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A to AFAC for factorization */
    slacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Factorize using the specified routine */
    switch (iroutine) {
    case 0:
        spotf2(uplo_str, n, ws->AFAC, lda, &info);
        break;
    case 1:
        spotrf2(uplo_str, n, ws->AFAC, lda, &info);
        break;
    case 2:
        spotrf(uplo_str, n, ws->AFAC, lda, &info);
        break;
    }
    (void)ROUTINE_NAMES;

    /* Types 3-5 are singular matrices where factorization should fail */
    int zerot = (imat >= 3 && imat <= 5);
    if (zerot) {
        /* info > 0 expected for singular matrices */
        assert_true(info >= 0);
        if (info > 0) {
            /* Factorization failed as expected for singular matrix */
            return;
        }
    } else {
        assert_info_success(info);
    }

    /* Verify factorization using spot01 */
    f32 resid;
    spot01(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->RWORK, &resid);

    assert_residual_below(resid, THRESH);
}

/**
 * CMocka test function - dispatches to run_dpotrf_single based on prestate.
 */
static void test_dpotrf_case(void** state)
{
    dpotrf_params_t* params = *state;
    run_dpotrf_single(params->n, params->iuplo, params->imat, params->iroutine);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NUPLO * NTYPES * NROUTINE = 7 * 2 * 9 * 3 = 378 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NROUTINE)

static dpotrf_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    static const char* ROUTINE_NAMES[] = {"dpotf2", "dpotrf2", "dpotrf"};

    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];

        int nimat = NTYPES;
        if (n <= 0) {
            nimat = 1;
        }

        for (int imat = 1; imat <= nimat; imat++) {
            /* Skip types 3, 4, or 5 if matrix size is too small */
            int zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (int iuplo = 0; iuplo < (int)NUPLO; iuplo++) {
                for (int iroutine = 0; iroutine < NROUTINE; iroutine++) {
                    /* Store parameters */
                    dpotrf_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->iroutine = iroutine;
                    snprintf(p->name, sizeof(p->name), "%s_n%d_%c_type%d",
                             ROUTINE_NAMES[iroutine], n, UPLOS[iuplo], imat);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dpotrf_case;
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

    /* Run all tests with shared workspace. */
    return _cmocka_run_group_tests("dpotrf", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
