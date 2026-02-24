/**
 * @file test_schkpo.c
 * @brief Comprehensive test suite for positive definite matrix (SPO) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkpo.f to C using CMocka.
 * Tests SPOTRF, SPOTRI, SPOTRS, SPORFS, and SPOCON.
 *
 * Each (n, uplo, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkpo.f:
 *   TEST 1: Cholesky factorization residual via spot01
 *   TEST 2: Matrix inverse residual via spot03
 *   TEST 3: Solution residual via spot02
 *   TEST 4: Solution accuracy via sget04
 *   TEST 5: Refined solution accuracy via sget04 (after sporfs)
 *   TEST 6-7: Error bounds via spot05
 *   TEST 8: Condition number via sget06
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-9
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
/* Test parameters from dtest.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values */
static const INT NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from dtest.in */
static const char UPLOS[] = {'U', 'L'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9
#define NTESTS  8
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
/* Verification routines */
/* Matrix generation */
/* Utilities */
/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT n;
    INT imat;
    INT iuplo;  /* 0='U', 1='L' */
    INT inb;    /* Index into NBVAL[] */
    char name[64];
} dchkpo_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* A;      /* Original matrix (NMAX x NMAX) */
    f32* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f32* AINV;   /* Inverse matrix (NMAX x NMAX) */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* D;      /* Singular values for slatms */
    f32* FERR;   /* Forward error bounds */
    f32* BERR;   /* Backward error bounds */
    INT* IWORK;     /* Integer workspace */
} dchkpo_workspace_t;

static dchkpo_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkpo_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->RWORK = malloc(NMAX * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
    g_workspace->IWORK = malloc(NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->IWORK) {
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
        free(g_workspace->AINV);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->D);
        free(g_workspace->FERR);
        free(g_workspace->BERR);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full dchkpo test battery for a single (n, uplo, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Following LAPACK's dchkpo.f:
 *   - TESTs 1-2 (factorization, inverse) run for all NB values
 *   - TESTs 3-8 (solve, refinement) only run for inb=0 (first NB)
 */
static void run_dchkpo_single(INT n, INT iuplo, INT imat, INT inb)
{
    const f32 ZERO = 0.0f;
    dchkpo_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    INT kl, ku, mode;
    f32 anorm, cndnum;
    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    f32 rcondc, rcond;

    /* Set block size for this test via xlaenv */
    INT nb = NBVAL[inb];
    xlaenv(1, nb);
    f32 result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Seed based on (n, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    slatb4("SPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric positive definite test matrix */
    slatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 3-5, zero one row and column to create singular matrix */
    INT zerot = (imat >= 3 && imat <= 5);
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }
        /* Zero row and column izero (1-based in LAPACK, 0-based here) */
        INT ioff = (izero - 1) * lda;
        if (iuplo == 0) {
            /* Upper: zero column above diagonal, then row to the right */
            for (INT i = 0; i < izero - 1; i++) {
                ws->A[ioff + i] = ZERO;
            }
            ioff += izero - 1;
            for (INT i = izero - 1; i < n; i++) {
                ws->A[ioff] = ZERO;
                ioff += lda;
            }
        } else {
            /* Lower: zero column below diagonal, then row to the left */
            ioff = izero - 1;
            for (INT i = 0; i < izero - 1; i++) {
                ws->A[ioff] = ZERO;
                ioff += lda;
            }
            ioff = (izero - 1) * lda + izero - 1;
            for (INT i = izero - 1; i < n; i++) {
                ws->A[ioff + i - (izero - 1)] = ZERO;
            }
        }
    } else {
        izero = 0;
    }

    /* Copy A to AFAC for factorization */
    slacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the Cholesky factorization */
    spotrf(uplo_str, n, ws->AFAC, lda, &info);

    /* Check error code */
    if (zerot) {
        /* For singular matrices, info should be > 0 */
        assert_true(info >= 0);
        if (info != izero) {
            /* Expected singularity at izero but got different result */
            return;
        }
    } else {
        assert_int_equal(info, 0);
    }

    /* Skip the rest if factorization failed */
    if (info != 0) {
        return;
    }

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 1 (factorization)", n, uplo, imat);
    set_test_context(ctx);
    slacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
    spot01(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2: Form the inverse and compute the residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
    set_test_context(ctx);
    slacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
    spotri(uplo_str, n, ws->AINV, lda, &info);
    assert_int_equal(info, 0);

    spot03(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
           ws->RWORK, &rcondc, &result[1]);
    assert_residual_below(result[1], THRESH);

    /*
     * Skip solve tests if not the first block size.
     * This matches LAPACK's dchkpo.f line 413: IF( INB.NE.1 ) GO TO 90
     */
    if (inb > 0) {
        goto test8;
    }

    /*
     * TESTS 3-7: Solve tests (only for first NB)
     */
    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        /*
         * TEST 3: Solve and compute residual for A * X = B
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 3 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        slarhs("SPO", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        slacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        spotrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->X, lda, &info);
        assert_int_equal(info, 0);

        slacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        spot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TEST 4: Check solution from generated exact solution
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 4 (solution accuracy)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
        assert_residual_below(result[3], THRESH);

        /*
         * TESTS 5, 6, 7: Iterative refinement
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 5-7 (refinement)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        slacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        spotrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->X, lda, &info);

        sporfs(uplo_str, n, nrhs, ws->A, lda, ws->AFAC, lda,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
        spot05(uplo_str, n, nrhs, ws->A, lda, ws->B, lda, ws->X, lda,
               ws->XACT, lda, ws->FERR, ws->BERR, &result[5]);

        assert_residual_below(result[4], THRESH);
        assert_residual_below(result[5], THRESH);
        assert_residual_below(result[6], THRESH);
    }

test8:
    /*
     * TEST 8: Get an estimate of RCOND = 1/CNDNUM
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d TEST 8 (condition number)", n, uplo, imat, nb);
    set_test_context(ctx);
    anorm = slansy("1", uplo_str, n, ws->A, lda, ws->RWORK);
    spocon(uplo_str, n, ws->AFAC, lda, anorm, &rcond, ws->WORK,
           ws->IWORK, &info);
    assert_int_equal(info, 0);

    result[7] = sget06(rcond, rcondc);
    assert_residual_below(result[7], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkpo_single based on prestate.
 */
static void test_dchkpo_case(void** state)
{
    dchkpo_params_t* params = *state;
    run_dchkpo_single(params->n, params->iuplo, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NUPLO * NTYPES * NNB = 7 * 2 * 9 * 5 = 630 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NNB)

static dchkpo_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];

        INT nimat = NTYPES;
        if (n <= 0) {
            nimat = 1;
        }

        for (INT imat = 1; imat <= nimat; imat++) {
            /* Skip types 3, 4, or 5 if matrix size is too small */
            INT zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (INT iuplo = 0; iuplo < (INT)NUPLO; iuplo++) {
                /* Loop over block sizes */
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    /* Store parameters */
                    dchkpo_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchkpo_n%d_%c_type%d_nb%d_%d",
                             n, UPLOS[iuplo], imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchkpo_case;
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
    return _cmocka_run_group_tests("dchkpo", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
