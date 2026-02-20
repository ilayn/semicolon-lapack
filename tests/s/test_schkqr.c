/**
 * @file test_schkqr.c
 * @brief Comprehensive test suite for QR factorization (SQR) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkqr.f to C using CMocka.
 * Tests SGEQRF, SORGQR, and SORMQR.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkqr.f:
 *   TEST 1-2: QR factorization via sqrt01 (norm(R - Q'*A), norm(I - Q'*Q))
 *   TEST 3-6: SORMQR tests via sqrt03 (applying Q from left/right with trans)
 *   TEST 7: Least squares solve via sgels and sget02
 *   TEST 8-9: SGEQRFP tests (positive diagonal R)
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
#define NTESTS  9   /* Tests 1-2 (sqrt01), 3-6 (sqrt03), 7 (sgels), 8-9 (sqrt01p) */
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void sgeqrf(const int m, const int n, f32* A, const int lda,
                   f32* tau, f32* work, const int lwork, int* info);
extern void sorgqr(const int m, const int n, const int k,
                   f32* A, const int lda, const f32* tau,
                   f32* work, const int lwork, int* info);
extern void sormqr(const char* side, const char* trans,
                   const int m, const int n, const int k,
                   const f32* A, const int lda, const f32* tau,
                   f32* C, const int ldc, f32* work, const int lwork,
                   int* info);
extern void sgels(const char* trans, const int m, const int n, const int nrhs,
                  f32* A, const int lda, f32* B, const int ldb,
                  f32* work, const int lwork, int* info);

/* Verification routines */
extern void sqrt01(const int m, const int n, const f32* A, f32* AF,
                   f32* Q, f32* R, const int lda, f32* tau,
                   f32* work, const int lwork, f32* rwork,
                   f32* result);
extern void sqrt02(const int m, const int n, const int k,
                   const f32* A, f32* AF, f32* Q, f32* R,
                   const int lda, const f32* tau,
                   f32* work, const int lwork, f32* rwork,
                   f32* result);
extern void sqrt03(const int m, const int n, const int k,
                   const f32* AF, f32* C, f32* CC, f32* Q,
                   const int lda, const f32* tau,
                   f32* work, const int lwork, f32* rwork,
                   f32* result);
extern void sget02(const char* trans, const int m, const int n, const int nrhs,
                   const f32* A, const int lda, const f32* X,
                   const int ldx, f32* B, const int ldb,
                   f32* rwork, f32* resid);
extern void sqrt01p(const int m, const int n, const f32* A, f32* AF,
                    f32* Q, f32* R, const int lda, f32* tau,
                    f32* work, const int lwork, f32* rwork,
                    f32* result);
extern int sgennd(const int m, const int n, const f32* A, const int lda);

/* Matrix generation */
extern void slatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f32* anorm, int* mode,
                   f32* cndnum, char* dist);
extern void slatms(const int m, const int n, const char* dist,
                   const char* sym, f32* d, const int mode, const f32 cond,
                   const f32 dmax, const int kl, const int ku, const char* pack,
                   f32* A, const int lda, f32* work, int* info,
                   uint64_t state[static 4]);
extern void slarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f32* A, const int lda,
                   const f32* XACT, const int ldxact, f32* B,
                   const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern f32 slamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int imat;
    int inb;    /* Index into NBVAL[] */
    char name[64];
} dchkqr_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* A;      /* Original matrix (NMAX x NMAX) */
    f32* AF;     /* Factored matrix (NMAX x NMAX) */
    f32* Q;      /* Orthogonal matrix Q (NMAX x NMAX) */
    f32* R;      /* Upper triangular R (NMAX x NMAX) */
    f32* C;      /* Workspace for sqrt03 (NMAX x NMAX) */
    f32* CC;     /* Another workspace for sqrt03 (NMAX x NMAX) */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* TAU;    /* Scalar factors of elementary reflectors */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* D;      /* Singular values for slatms */
    int* IWORK;     /* Integer workspace */
} dchkqr_workspace_t;

static dchkqr_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkqr_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AF = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->Q = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->R = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->C = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->CC = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->TAU = malloc(NMAX * sizeof(f32));
    g_workspace->WORK = malloc(lwork * sizeof(f32));
    g_workspace->RWORK = malloc(NMAX * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->IWORK = malloc(NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->AF || !g_workspace->Q ||
        !g_workspace->R || !g_workspace->C || !g_workspace->CC ||
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
        free(g_workspace->R);
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
 * Run the full dchkqr test battery for a single (m, n, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Unlike dchkge/dchkpo/dchksy, ALL tests run for each NB value because
 * the QR factorization and Q operations are all affected by blocking.
 */
static void run_dchkqr_single(int m, int n, int imat, int inb_idx)
{
    const f32 ZERO = 0.0f;
    dchkqr_workspace_t* ws = g_workspace;

    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info;
    int lda = NMAX;
    int lwork = NMAX * NMAX;
    int minmn = (m < n) ? m : n;
    f32 result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Set block size and crossover point for this test via xlaenv */
    int nb = NBVAL[inb_idx];
    int nx = NXVAL[inb_idx];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Seed based on (m, n, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    slatb4("SQR", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    slatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
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
             * TEST 1-2: Test SGEQRF via sqrt01
             * Computes:
             *   result[0] = norm(R - Q'*A) / (M * norm(A) * eps)
             *   result[1] = norm(I - Q'*Q) / (M * eps)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1 (QR factorization R)", m, n, imat, k);
            set_test_context(ctx);
            sqrt01(m, n, ws->A, ws->AF, ws->Q, ws->R, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 2 (QR factorization Q)", m, n, imat, k);
            set_test_context(ctx);
            assert_residual_below(result[1], THRESH);

            /*
             * TEST 8-9: Test SGEQRFP via sqrt01p
             * SGEQRFP produces QR factorization with non-negative diagonal R.
             * Test 8: same as tests 1-2 but using SGEQRFP
             * Test 9: check that diagonal of R is non-negative (via sgennd)
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 8 (SGEQRFP factorization)", m, n, imat, k);
            set_test_context(ctx);
            sqrt01p(m, n, ws->A, ws->AF, ws->Q, ws->R, lda, ws->TAU,
                    ws->WORK, lwork, ws->RWORK, &result[7]);

            assert_residual_below(result[7], THRESH);
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 8b (SGEQRFP orthogonality)", m, n, imat, k);
            set_test_context(ctx);
            assert_residual_below(result[8], THRESH);

            /* Test 9: Check that AF has non-negative diagonal */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 9 (SGEQRFP non-negative diagonal)", m, n, imat, k);
            set_test_context(ctx);
            if (!sgennd(m, n, ws->AF, lda)) {
                result[8] = 2 * THRESH;  /* Fail the test */
            }
            assert_residual_below(result[8], THRESH);
        } else if (m >= n) {
            /*
             * TEST 1-2: Test SORGQR via sqrt02
             * Uses factorization from sqrt01
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 1-2 (SORGQR)", m, n, imat, k);
            set_test_context(ctx);
            sqrt02(m, n, k, ws->A, ws->AF, ws->Q, ws->R, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, result);

            assert_residual_below(result[0], THRESH);
            assert_residual_below(result[1], THRESH);
        }

        if (m >= k) {
            /*
             * TEST 3-6: Test SORMQR via sqrt03
             * Tests Q from left/right with 'N'/'T' transpose options
             */
            snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d TEST 3-6 (SORMQR)", m, n, imat, k);
            set_test_context(ctx);
            sqrt03(m, n, k, ws->AF, ws->C, ws->CC, ws->Q, lda, ws->TAU,
                   ws->WORK, lwork, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);
            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);

            /*
             * TEST 7: If M >= N and K == N and INB == 1, test SGELS least squares solve
             * Only run on first block size (INB.EQ.1 in Fortran, inb_idx == 0 in C)
             */
            if (k == n && inb_idx == 0) {
                int nrhs = NSVAL[0];  /* Use first NRHS value */

                snprintf(ctx, sizeof(ctx), "m=%d n=%d imat=%d k=%d nrhs=%d TEST 7 (SGELS)", m, n, imat, k, nrhs);
                set_test_context(ctx);

                /* Generate right-hand side */
                slarhs("SQR", "N", "F", "N", m, n, 0, 0, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

                slacpy("F", m, nrhs, ws->B, lda, ws->X, lda);
                /* Use ws->C instead of ws->AF for SGELS - SGELS modifies its input
                 * and we need to preserve AF for subsequent sqrt02/sqrt03 tests */
                slacpy("F", m, n, ws->A, lda, ws->C, lda);

                sgels("N", m, n, nrhs, ws->C, lda, ws->X, lda,
                      ws->WORK, lwork, &info);
                assert_int_equal(info, 0);

                /* Check residual */
                slacpy("F", m, nrhs, ws->B, lda, ws->WORK, lda);
                sget02("N", m, n, nrhs, ws->A, lda, ws->X, lda,
                       ws->WORK, lda, ws->RWORK, &result[6]);

                assert_residual_below(result[6], THRESH);
            }
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkqr_single based on prestate.
 */
static void test_dchkqr_case(void** state)
{
    dchkqr_params_t* params = *state;
    run_dchkqr_single(params->m, params->n, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NTYPES * NNB = 7 * 7 * 8 * 5 = 1960 tests
 */

/* Maximum number of test cases */
#define MAX_TESTS (NM * NN * NTYPES * NNB)

static dchkqr_params_t g_params[MAX_TESTS];
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
                    dchkqr_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imat = imat;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchkqr_m%d_n%d_type%d_nb%d_%d",
                             m, n, imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchkqr_case;
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
    return _cmocka_run_group_tests("dchkqr", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
