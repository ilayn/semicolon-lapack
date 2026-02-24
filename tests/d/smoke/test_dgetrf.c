/**
 * @file test_dgetrf.c
 * @brief CMocka test suite for dgetrf (LU factorization with partial pivoting).
 *
 * Tests the blocked LU factorization routine dgetrf using LAPACK's
 * verification methodology with normalized residuals.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Verification: dget01 computes ||L*U - A|| / (N * ||A|| * eps)
 *
 * Matrix types tested (from dlatb4):
 *   1. Diagonal
 *   2. Upper triangular
 *   3. Lower triangular
 *   4. Random, well-conditioned (cond=2)
 *   5-7. Zero column matrices (singular)
 *   8. Random, ill-conditioned (cond ~ 3e7)
 *   9. Random, very ill-conditioned (cond ~ 9e15)
 *   10. Scaled near underflow
 *   11. Scaled near overflow
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
/* Test parameters from dtest.in */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  11
#define THRESH  30.0
#define NMAX    50

/* Routine under test */
/* Utilities */
/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT m;
    INT n;
    INT imat;
    char name[64];
} dgetrf_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Original matrix (NMAX x NMAX) */
    f64* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f64* D;      /* Singular values for dlatms */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace */
    INT* IPIV;      /* Pivot indices */
} dgetrf_workspace_t;

static dgetrf_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dgetrf_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->WORK = malloc(3 * NMAX * sizeof(f64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->IPIV = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->D ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->IPIV) {
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
        free(g_workspace->IPIV);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the dgetrf test for a single (m, n, imat) combination.
 */
static void run_dgetrf_single(INT m, INT n, INT imat)
{
    dgetrf_workspace_t* ws = g_workspace;

    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info;
    INT lda = (m > 1) ? m : 1;
    INT minmn = (m < n) ? m : n;

    /* Get matrix parameters for this type */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 10000 + n * 100 + imat));
    dlatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 5-7, zero one or more columns to create singular matrix */
    INT zerot = (imat >= 5 && imat <= 7);
    INT izero = 0;
    if (zerot) {
        if (imat == 5) {
            izero = 1;
        } else if (imat == 6) {
            izero = minmn;
        } else {
            izero = minmn / 2 + 1;
        }
        INT ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (INT i = 0; i < m; i++) {
                ws->A[ioff + i] = 0.0;
            }
        } else {
            dlaset("F", m, n - izero + 1, 0.0, 0.0, &ws->A[ioff], lda);
        }
    }

    /* Copy A to AFAC for factorization */
    dlacpy("F", m, n, ws->A, lda, ws->AFAC, lda);

    /* Factorize */
    dgetrf(m, n, ws->AFAC, lda, ws->IPIV, &info);

    /* For singular matrix types (5-7), info > 0 is expected and acceptable */
    if (zerot) {
        assert_true(info >= 0);
    } else {
        assert_int_equal(info, 0);
    }

    /* Verify factorization using dget01 */
    f64 resid;
    f64* AFAC_copy = malloc(lda * n * sizeof(f64));
    assert_non_null(AFAC_copy);
    memcpy(AFAC_copy, ws->AFAC, lda * n * sizeof(f64));

    dget01(m, n, ws->A, lda, AFAC_copy, lda, ws->IPIV, ws->RWORK, &resid);
    free(AFAC_copy);

    assert_residual_below(resid, THRESH);
}

/**
 * CMocka test function - dispatches to run_dgetrf_single based on prestate.
 */
static void test_dgetrf_case(void** state)
{
    dgetrf_params_t* params = *state;
    run_dgetrf_single(params->m, params->n, params->imat);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NTYPES = 7 * 7 * 11 = 539 tests
 * (minus skipped cases for small m,n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NM * NN * NTYPES)

static dgetrf_params_t g_params[MAX_TESTS];
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

            INT nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (INT imat = 1; imat <= nimat; imat++) {
                /* Skip types 5, 6, or 7 if matrix size is too small */
                INT zerot = (imat >= 5 && imat <= 7);
                INT minmn = (m < n) ? m : n;
                if (zerot && minmn < imat - 4) {
                    continue;
                }

                /* Store parameters */
                dgetrf_params_t* p = &g_params[g_num_tests];
                p->m = m;
                p->n = n;
                p->imat = imat;
                snprintf(p->name, sizeof(p->name), "dgetrf_m%d_n%d_type%d",
                         m, n, imat);

                /* Create CMocka test entry */
                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dgetrf_case;
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
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace. */
    return _cmocka_run_group_tests("dgetrf", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
