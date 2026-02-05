/**
 * @file test_dchklq.c
 * @brief Comprehensive test suite for LQ factorization (DLQ) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchklq.f to C using CMocka.
 * Tests DGELQF, DORGLQ, and DORMLQ.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchklq.f:
 *   TEST 1-2: LQ factorization via dlqt01 (norm(L - A*Q'), norm(I - Q*Q'))
 *   TEST 3-6: DORMLQ tests via dlqt03 (applying Q from left/right with trans)
 *   TEST 7: Least squares solve via dgels and dget02
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   Matrix types: 1-8
 *   K values: MINMN, 0, 1, MINMN/2
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 15};  /* NRHS values for least squares */
static const int NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from dtest.in */
static const int NXVAL[] = {1, 0, 5, 9, 1};   /* Crossover points from dtest.in */

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPES  8
#define NTESTS  7   /* We test 1-2 (dlqt01), 3-6 (dlqt03), 7 (dgels) */
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void dgelqf(const int m, const int n, double* A, const int lda,
                   double* tau, double* work, const int lwork, int* info);
extern void dorglq(const int m, const int n, const int k,
                   double* A, const int lda, const double* tau,
                   double* work, const int lwork, int* info);
extern void dormlq(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const double* A, const int lda, const double* tau,
                   double* C, const int ldc, double* work, const int lwork,
                   int* info);
extern void dgels(const char* trans, const int m, const int n, const int nrhs,
                  double* A, const int lda, double* B, const int ldb,
                  double* work, const int lwork, int* info);

/* Verification routines */
extern void dlqt01(const int m, const int n, const double* A, double* AF,
                   double* Q, double* L, const int lda, double* tau,
                   double* work, const int lwork, double* rwork,
                   double* result);
extern void dlqt02(const int m, const int n, const int k,
                   const double* A, const double* AF, double* Q, double* L,
                   const int lda, const double* tau,
                   double* work, const int lwork, double* rwork,
                   double* result);
extern void dlqt03(const int m, const int n, const int k,
                   const double* AF, double* C, double* CC, double* Q,
                   const int lda, const double* tau,
                   double* work, const int lwork, double* rwork,
                   double* result);
extern void dget02(const char* trans, const int m, const int n, const int nrhs,
                   const double* A, const int lda, const double* X,
                   const int ldx, double* B, const int ldb,
                   double* rwork, double* resid);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist, uint64_t seed,
                   const char* sym, double* d, const int mode, const double cond,
                   const double dmax, const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const double* A, const int lda,
                   const double* XACT, const int ldxact, double* B,
                   const int ldb, uint64_t seed, int* info);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern double dlamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int imat;
    int inb;    /* Index into NBVAL[] */
    char name[64];
} dchklq_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    double* A;      /* Original matrix (NMAX x NMAX) */
    double* AF;     /* Factored matrix (NMAX x NMAX) */
    double* Q;      /* Orthogonal matrix Q (NMAX x NMAX) */
    double* L;      /* Lower triangular L (NMAX x NMAX) */
    double* C;      /* Workspace for dlqt03 (NMAX x NMAX) */
    double* CC;     /* Another workspace for dlqt03 (NMAX x NMAX) */
    double* B;      /* Right-hand side (NMAX x NSMAX) */
    double* X;      /* Solution (NMAX x NSMAX) */
    double* XACT;   /* Exact solution (NMAX x NSMAX) */
    double* TAU;    /* Scalar factors of elementary reflectors */
    double* WORK;   /* General workspace */
    double* RWORK;  /* Real workspace */
    double* D;      /* Singular values for dlatms */
    int* IWORK;     /* Integer workspace */
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

    int lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->AF = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->Q = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->L = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->C = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->CC = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->TAU = malloc(NMAX * sizeof(double));
    g_workspace->WORK = malloc(lwork * sizeof(double));
    g_workspace->RWORK = malloc(NMAX * sizeof(double));
    g_workspace->D = malloc(NMAX * sizeof(double));
    g_workspace->IWORK = malloc(NMAX * sizeof(int));

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
static void run_dchklq_single(int m, int n, int imat, int inb)
{
    const double ZERO = 0.0;
    dchklq_workspace_t* ws = g_workspace;

    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;
    int lda = NMAX;
    int lwork = NMAX * NMAX;
    int minmn = (m < n) ? m : n;
    double result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Set block size and crossover point for this test via xlaenv */
    int nb = NBVAL[inb];
    int nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Seed based on (m, n, imat) for reproducibility */
    uint64_t seed = 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imat);

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    dlatb4("DLQ", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    dlatms(m, n, &dist, seed++, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info);
    assert_int_equal(info, 0);

    /* Set K values to test: MINMN, 0, 1, MINMN/2 */
    int kval[4];
    kval[0] = minmn;
    kval[1] = 0;
    kval[2] = 1;
    kval[3] = minmn / 2;

    int nk;
    if (minmn == 0) {
        nk = 1;
    } else if (minmn == 1) {
        nk = 2;
    } else if (minmn <= 3) {
        nk = 3;
    } else {
        nk = 4;
    }

    for (int ik = 0; ik < nk; ik++) {
        int k = kval[ik];
        if (ik == 0) {
            /*
             * TEST 1-2: Test DGELQF via dlqt01
             * Computes:
             *   result[0] = norm(L - A*Q') / (N * norm(A) * eps)
             *   result[1] = norm(I - Q*Q') / (N * eps)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1 (LQ factorization L)", m, n, imat, k);
            set_test_context(ctx);
            dlqt01(m, n, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 2 (LQ factorization Q)", m, n, imat, k);
            set_test_context(ctx);
            assert_residual_below(result[1], THRESH);
        } else if (m <= n) {
            /*
             * TEST 1-2: Test DORGLQ via dlqt02
             * Uses factorization from dlqt01
             * Note: For LQ, we test when m <= n (wide/square matrices)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1-2 (DORGLQ)", m, n, imat, k);
            set_test_context(ctx);
            dlqt02(m, n, k, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            assert_residual_below(result[1], THRESH);
        }

        if (m >= k) {
            /*
             * TEST 3-6: Test DORMLQ via dlqt03
             * Tests Q from left/right with 'N'/'T' transpose options
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 3-6 (DORMLQ)", m, n, imat, k);
            set_test_context(ctx);
            dlqt03(m, n, k, ws->AF, ws->C, ws->CC, ws->Q, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);
            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);

            /*
             * TEST 7: If M <= N and K == M, test DGELS least squares solve
             * Note: For LQ, condition is K == M (underdetermined system)
             * Only test once per (M, N, IMAT) when INB == 1
             */
            if (k == m && m <= n && m > 0 && inb == 0) {
                int nrhs = NSVAL[0];  /* Use first NRHS value */

                snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d nrhs=%d TEST 7 (DGELS)", m, n, imat, k, nrhs);
                set_test_context(ctx);

                /* Generate right-hand side */
                dlarhs("DLQ", "N", "F", "N", m, n, 0, 0, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, seed++, &info);

                dlacpy("F", m, nrhs, ws->B, lda, ws->X, lda);
                /* Use ws->C instead of ws->AF for DGELS - DGELS modifies its input
                 * and we need to preserve AF for subsequent dlqt02/dlqt03 tests */
                dlacpy("F", m, n, ws->A, lda, ws->C, lda);

                dgels("N", m, n, nrhs, ws->C, lda, ws->X, lda,
                      ws->WORK, lwork, &info);
                assert_int_equal(info, 0);

                /* Check residual */
                dlacpy("F", m, nrhs, ws->B, lda, ws->WORK, lda);
                dget02("N", m, n, nrhs, ws->A, lda, ws->X, lda,
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

            for (int imat = 1; imat <= NTYPES; imat++) {
                /* Loop over block sizes */
                for (int inb = 0; inb < (int)NNB; inb++) {
                    int nb = NBVAL[inb];

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
