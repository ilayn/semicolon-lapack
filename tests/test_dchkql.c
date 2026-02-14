/**
 * @file test_dchkql.c
 * @brief Comprehensive test suite for QL factorization (DQL) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkql.f to C using CMocka.
 * Tests DGEQLF, DORGQL, and DORMQL.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkql.f:
 *   TEST 1-2: QL factorization via dqlt01 (norm(L - Q'*A), norm(I - Q'*Q))
 *   TEST 3-6: DORMQL tests via dqlt03 (applying Q from left/right with trans)
 *   TEST 7: Least squares solve via dgeqls and dget02
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   Matrix types: 1-8
 *   K values: MINMN, 0, 1, MINMN/2
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "test_rng.h"
#include "semicolon_lapack_double.h"
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
#define NTESTS  7   /* We test 1-2 (dqlt01), 3-6 (dqlt03), 7 (dgeqls) */
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Verification routines */
extern void dqlt01(const int m, const int n, const f64* A, f64* AF,
                   f64* Q, f64* L, const int lda, f64* tau,
                   f64* work, const int lwork, f64* rwork,
                   f64* result);
extern void dqlt02(const int m, const int n, const int k,
                   const f64* A, const f64* AF, f64* Q, f64* L,
                   const int lda, const f64* tau,
                   f64* work, const int lwork, f64* rwork,
                   f64* result);
extern void dqlt03(const int m, const int n, const int k,
                   const f64* AF, f64* C, f64* CC, f64* Q,
                   const int lda, const f64* tau,
                   f64* work, const int lwork, f64* rwork,
                   f64* result);
extern void dget02(const char* trans, const int m, const int n, const int nrhs,
                   const f64* A, const int lda, const f64* X,
                   const int ldx, f64* B, const int ldb,
                   f64* rwork, f64* resid);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode, const f64 cond,
                   const f64 dmax, const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f64* A, const int lda,
                   const f64* XACT, const int ldxact, f64* B,
                   const int ldb, int* info, uint64_t state[static 4]);

/* DGEQLS - solve using QL factorization (not a standard LAPACK routine,
 * but used in LAPACK testing). We implement it inline here. */
static void dgeqls(const int m, const int n, const int nrhs,
                   f64* A, const int lda, const f64* tau,
                   f64* B, const int ldb,
                   f64* work, const int lwork, int* info)
{
    *info = 0;
    if (m < 0) *info = -1;
    else if (n < 0 || n > m) *info = -2;
    else if (nrhs < 0) *info = -3;
    else if (lda < m || lda < 1) *info = -5;
    else if (ldb < m || ldb < 1) *info = -8;
    if (*info != 0) return;

    if (n == 0 || nrhs == 0) return;

    /* B := Q' * B */
    dormql("L", "T", m, nrhs, n, A, lda, tau, B, ldb, work, lwork, info);
    if (*info != 0) return;

    dtrtrs("L", "N", "N", n, nrhs, &A[(m - n) + 0 * lda], lda,
           &B[(m - n) + 0 * ldb], ldb, info);
}

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int imat;
    int inb;    /* Index into NBVAL[] */
    char name[64];
} dchkql_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Original matrix (NMAX x NMAX) */
    f64* AF;     /* Factored matrix (NMAX x NMAX) */
    f64* Q;      /* Orthogonal matrix Q (NMAX x NMAX) */
    f64* L;      /* Lower triangular L (NMAX x NMAX) */
    f64* C;      /* Workspace for dqlt03 (NMAX x NMAX) */
    f64* CC;     /* Another workspace for dqlt03 (NMAX x NMAX) */
    f64* B;      /* Right-hand side (NMAX x NSMAX) */
    f64* X;      /* Solution (NMAX x NSMAX) */
    f64* XACT;   /* Exact solution (NMAX x NSMAX) */
    f64* TAU;    /* Scalar factors of elementary reflectors */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace */
    f64* D;      /* Singular values for dlatms */
    int* IWORK;     /* Integer workspace */
} dchkql_workspace_t;

static dchkql_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkql_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AF = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->Q = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->L = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->C = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->CC = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->TAU = malloc(NMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
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
 * Run the full dchkql test battery for a single (m, n, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Unlike dchkge/dchkpo/dchksy, ALL tests run for each NB value because
 * the QL factorization and Q operations are all affected by blocking.
 */
static void run_dchkql_single(int m, int n, int imat, int inb)
{
    const f64 ZERO = 0.0;
    dchkql_workspace_t* ws = g_workspace;

    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info;
    int lda = NMAX;
    int lwork = NMAX * NMAX;
    int minmn = (m < n) ? m : n;
    f64 result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Set block size and crossover point for this test via xlaenv */
    int nb = NBVAL[inb];
    int nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Seed based on (m, n, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imat));

    /* Initialize results */
    for (int i = 0; i < NTESTS; i++) {
        result[i] = ZERO;
    }

    /* Get matrix parameters for this type */
    dlatb4("DQL", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    dlatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
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
             * TEST 1-2: Test DGEQLF via dqlt01
             * Computes:
             *   result[0] = norm(L - Q'*A) / (M * norm(A) * eps)
             *   result[1] = norm(I - Q'*Q) / (M * eps)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1 (QL factorization L)", m, n, imat, k);
            set_test_context(ctx);
            dqlt01(m, n, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 2 (QL factorization Q)", m, n, imat, k);
            set_test_context(ctx);
            assert_residual_below(result[1], THRESH);
        } else if (m >= n) {
            /*
             * TEST 1-2: Test DORGQL via dqlt02
             * Uses factorization from dqlt01
             * Note: For QL, we test when m >= n (tall/square matrices)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1-2 (DORGQL)", m, n, imat, k);
            set_test_context(ctx);
            dqlt02(m, n, k, ws->A, ws->AF, ws->Q, ws->L, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            assert_residual_below(result[1], THRESH);
        }

        if (m >= k) {
            /*
             * TEST 3-6: Test DORMQL via dqlt03
             * Tests Q from left/right with 'N'/'T' transpose options
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 3-6 (DORMQL)", m, n, imat, k);
            set_test_context(ctx);
            dqlt03(m, n, k, ws->AF, ws->C, ws->CC, ws->Q, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);
            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);

            /*
             * TEST 7: If M >= N and K == N, test DGEQLS least squares solve
             * Note: For QL, condition is K == N (overdetermined system)
             * Only test once per (M, N, IMAT) when INB == 1
             */
            if (k == n && m >= n && n > 0 && inb == 0) {
                int nrhs = NSVAL[0];  /* Use first NRHS value */

                snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d nrhs=%d TEST 7 (DGEQLS)", m, n, imat, k, nrhs);
                set_test_context(ctx);

                /* Generate right-hand side */
                dlarhs("DQL", "N", "F", "N", m, n, 0, 0, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

                dlacpy("F", m, nrhs, ws->B, lda, ws->X, lda);

                /* Use the factorization already in AF */
                dgeqls(m, n, nrhs, ws->AF, lda, ws->TAU, ws->X, lda,
                       ws->WORK, lwork, &info);
                assert_int_equal(info, 0);

                /* Check residual: X is in X(m-n:m-1, :) */
                dlacpy("F", m, nrhs, ws->B, lda, ws->WORK, lda);
                dget02("N", m, n, nrhs, ws->A, lda, &ws->X[(m - n)], lda,
                       ws->WORK, lda, ws->RWORK, &result[6]);

                assert_residual_below(result[6], THRESH);
            }
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkql_single based on prestate.
 */
static void test_dchkql_case(void** state)
{
    dchkql_params_t* params = *state;
    run_dchkql_single(params->m, params->n, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NTYPES * NNB = 7 * 7 * 8 * 5 = 1960 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NM * NN * NTYPES * NNB)

static dchkql_params_t g_params[MAX_TESTS];
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
                    dchkql_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imat = imat;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchkql_m%d_n%d_type%d_nb%d_%d",
                             m, n, imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchkql_case;
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
    return _cmocka_run_group_tests("dchkql", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
