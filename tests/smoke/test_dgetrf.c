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
#include <string.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NTYPES  11
#define THRESH  30.0
#define NMAX    50

/* Routine under test */
extern void dgetrf(const int m, const int n, double* A,
                   const int lda, int* ipiv, int* info);

/* Verification routine */
extern void dget01(const int m, const int n, const double* A,
                   const int lda, double* AFAC,
                   const int ldafac, const int* ipiv,
                   double* rwork, double* resid);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   uint64_t seed, const char* sym, double* d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int imat;
    char name[64];
} dgetrf_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    double* A;      /* Original matrix (NMAX x NMAX) */
    double* AFAC;   /* Factored matrix (NMAX x NMAX) */
    double* D;      /* Singular values for dlatms */
    double* WORK;   /* General workspace */
    double* RWORK;  /* Real workspace */
    int* IPIV;      /* Pivot indices */
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

    g_workspace->A = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->D = malloc(NMAX * sizeof(double));
    g_workspace->WORK = malloc(3 * NMAX * sizeof(double));
    g_workspace->RWORK = malloc(NMAX * sizeof(double));
    g_workspace->IPIV = malloc(NMAX * sizeof(int));

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
static void run_dgetrf_single(int m, int n, int imat)
{
    dgetrf_workspace_t* ws = g_workspace;

    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;
    int lda = (m > 1) ? m : 1;
    int minmn = (m < n) ? m : n;

    /* Seed based on (m, n, imat) for reproducibility */
    uint64_t seed = 1988198919901991ULL + (uint64_t)(m * 10000 + n * 100 + imat);

    /* Get matrix parameters for this type */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    dlatms(m, n, &dist, seed, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info);
    assert_int_equal(info, 0);

    /* For types 5-7, zero one or more columns to create singular matrix */
    int zerot = (imat >= 5 && imat <= 7);
    int izero = 0;
    if (zerot) {
        if (imat == 5) {
            izero = 1;
        } else if (imat == 6) {
            izero = minmn;
        } else {
            izero = minmn / 2 + 1;
        }
        int ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (int i = 0; i < m; i++) {
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
    double resid;
    double* AFAC_copy = malloc(lda * n * sizeof(double));
    assert_non_null(AFAC_copy);
    memcpy(AFAC_copy, ws->AFAC, lda * n * sizeof(double));

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

            int nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (int imat = 1; imat <= nimat; imat++) {
                /* Skip types 5, 6, or 7 if matrix size is too small */
                int zerot = (imat >= 5 && imat <= 7);
                int minmn = (m < n) ? m : n;
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
