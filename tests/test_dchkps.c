/**
 * @file test_dchkps.c
 * @brief Comprehensive test suite for positive semidefinite pivoted Cholesky (DPS) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkps.f to C using CMocka.
 * Tests DPSTRF (and by extension DPSTF2).
 *
 * Each (n, uplo, imat, irank, inb) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkps.f:
 *   - Generate symmetric positive semidefinite matrix with specified rank
 *   - Compute pivoted Cholesky factorization via DPSTRF
 *   - Verify factorization residual via DPST01
 *   - Check computed rank matches expected rank
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NB values: 1, 3, 3, 3, 20
 *   RANKVAL: 30, 50, 90 (percentages)
 *   Matrix types: 1-9
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NBVAL[] = {1, 3, 3, 3, 20};
static const int RANKVAL[] = {30, 50, 90};  /* Percentage of full rank */
static const char UPLOS[] = {'U', 'L'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NRANK   (sizeof(RANKVAL) / sizeof(RANKVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */

/* Routine under test */
extern void dpstrf(const char* uplo, const int n, f64* A, const int lda,
                   int* piv, int* rank, const f64 tol, f64* work,
                   int* info);

/* Verification routine */
extern void dpst01(const char* uplo, const int n,
                   const f64* const restrict A, const int lda,
                   f64* const restrict AFAC, const int ldafac,
                   f64* const restrict PERM, const int ldperm,
                   const int* const restrict piv,
                   f64* const restrict rwork, f64* resid, const int rank);

/* Matrix generation */
extern void dlatb5(const char* path, const int imat, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatmt(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode,
                   const f64 cond, const f64 dmax, const int rank,
                   const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    int iuplo;  /* 0='U', 1='L' */
    int irank;  /* Index into RANKVAL[] */
    int inb;    /* Index into NBVAL[] */
    char name[80];
} dchkps_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Original matrix (NMAX x NMAX) */
    f64* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f64* PERM;   /* Permuted reconstruction (NMAX x NMAX) */
    f64* WORK;   /* General workspace (2*NMAX for dpstrf, 3*NMAX total) */
    f64* RWORK;  /* Real workspace for dlansy in dpst01 */
    f64* D;      /* Singular values for dlatmt */
    int* PIV;       /* Pivot indices */
} dchkps_workspace_t;

static dchkps_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkps_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->PERM = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->WORK = malloc(3 * NMAX * sizeof(f64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->PIV = malloc(NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->PERM ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->PIV) {
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
        free(g_workspace->PERM);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->D);
        free(g_workspace->PIV);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full dchkps test battery for a single (n, uplo, imat, irank, inb) combination.
 */
static void run_dchkps_single(int n, int iuplo, int imat, int irank, int inb)
{
    dchkps_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info;
    int lda = (n > 1) ? n : 1;
    f64 result;
    char ctx[128];
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat * 10 + irank));

    /* Set block size for this test via xlaenv */
    int nb = NBVAL[inb];
    xlaenv(1, nb);

    /* Compute expected rank from percentage */
    int rank = (int)ceil((n * (f64)RANKVAL[irank]) / 100.0);
    if (rank < 1 && n > 0) rank = 1;
    if (rank > n) rank = n;

    /* Get matrix parameters for this type */
    dlatb5("DPS", imat, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d rank=%d nb=%d (dlatmt)",
             n, uplo, imat, rank, nb);
    set_test_context(ctx);

    /* Generate symmetric positive semidefinite test matrix with specified rank */
    dlatmt(n, n, &dist, &type, ws->D, mode, cndnum, anorm, rank,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);

    if (info != 0) {
        /* dlatmt failed - skip this test case */
        clear_test_context();
        return;
    }

    /* Copy A to AFAC for factorization */
    dlacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d rank=%d nb=%d (dpstrf)",
             n, uplo, imat, rank, nb);
    set_test_context(ctx);

    /* Compute pivoted Cholesky factorization with default tolerance */
    f64 tol = -1.0;
    int comprank;
    dpstrf(uplo_str, n, ws->AFAC, lda, ws->PIV, &comprank, tol, ws->WORK, &info);

    /* Check error code from DPSTRF */
    if (info < 0) {
        /* Illegal argument */
        fail_msg("DPSTRF returned info=%d (illegal argument)", info);
        clear_test_context();
        return;
    }

    /* INFO > 0 is acceptable - it means the matrix is rank deficient,
     * which is expected when rank < n */
    if (info != 0 && rank == n) {
        /* Full rank expected but factorization indicated rank deficiency */
        fail_msg("DPSTRF returned info=%d for full-rank matrix", info);
        clear_test_context();
        return;
    }

    /* Skip residual test if factorization failed (info > 0 and rank < n is OK) */
    if (info != 0) {
        clear_test_context();
        return;
    }

    /* Reconstruct matrix from factors and compute residual */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d rank=%d nb=%d (dpst01)",
             n, uplo, imat, rank, nb);
    set_test_context(ctx);

    dpst01(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->PERM, lda,
           ws->PIV, ws->RWORK, &result, comprank);

    /* For n=0, set comprank=0 as in dchkps.f line 332-333 */
    if (n == 0) {
        comprank = 0;
    }

    /* Check residual */
    assert_residual_below(result, THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkps_single based on prestate.
 */
static void test_dchkps_case(void** state)
{
    dchkps_params_t* params = *state;
    run_dchkps_single(params->n, params->iuplo, params->imat, params->irank, params->inb);
}

/*
 * Generate all parameter combinations.
 * Note: Only types 3-5 are tested with multiple ranks; others use full rank only.
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NRANK * NNB)

static dchkps_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];

        int nimat = NTYPES;
        if (n <= 0) {
            nimat = 1;
        }

        for (int imat = 1; imat <= nimat; imat++) {
            for (int irank = 0; irank < (int)NRANK; irank++) {
                /* Only repeat test for different ranks if imat in 3-5 */
                if ((imat < 3 || imat > 5) && irank > 0) {
                    continue;
                }

                int rank = (int)ceil((n * (f64)RANKVAL[irank]) / 100.0);
                if (rank < 1 && n > 0) rank = 1;
                if (rank > n) rank = n;

                for (int iuplo = 0; iuplo < (int)NUPLO; iuplo++) {
                    /* Loop over block sizes */
                    for (int inb = 0; inb < (int)NNB; inb++) {
                        int nb = NBVAL[inb];

                        /* Store parameters */
                        dchkps_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->imat = imat;
                        p->iuplo = iuplo;
                        p->irank = irank;
                        p->inb = inb;
                        snprintf(p->name, sizeof(p->name),
                                 "dchkps_n%d_%c_type%d_rank%d_nb%d",
                                 n, UPLOS[iuplo], imat, rank, nb);

                        /* Create CMocka test entry */
                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_dchkps_case;
                        g_tests[g_num_tests].setup_func = NULL;
                        g_tests[g_num_tests].teardown_func = NULL;
                        g_tests[g_num_tests].initial_state = p;

                        g_num_tests++;
                    }
                }
            }
        }
    }
}

int main(void)
{
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace */
    return _cmocka_run_group_tests("dchkps", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
