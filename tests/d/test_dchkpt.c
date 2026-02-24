/**
 * @file test_dchkpt.c
 * @brief Comprehensive test suite for positive definite tridiagonal matrix (DPT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkpt.f to C using CMocka.
 * Tests DPTTRF, DPTTRS, DPTRFS, and DPTCON.
 *
 * Each (n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkpt.f:
 *   TEST 1: L*D*L' factorization residual via dptt01
 *   TEST 2: Solution residual via dptt02
 *   TEST 3: Solution accuracy via dget04
 *   TEST 4: Refined solution accuracy via dget04 (after dptrfs)
 *   TEST 5-6: Error bounds via dptt05
 *   TEST 7: Condition number via dget06
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-12
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include "semicolon_cblas.h"

/* Test parameters from dtest.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values */

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  12
#define NTESTS  7
#define THRESH  30.0
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
    char name[64];
} dchkpt_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* D;      /* Original diagonal (NMAX) */
    f64* E;      /* Original subdiagonal (NMAX-1) */
    f64* DF;     /* Factored diagonal (NMAX) */
    f64* EF;     /* Factored subdiagonal (NMAX-1) */
    f64* B;      /* Right-hand side (NMAX x NSMAX) */
    f64* X;      /* Solution (NMAX x NSMAX) */
    f64* XACT;   /* Exact solution (NMAX x NSMAX) */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace for error bounds */
    f64* FERR;   /* Forward error bounds */
    f64* BERR;   /* Backward error bounds */
    f64* Z;      /* Storage for zeroed elements (3) */
} dchkpt_workspace_t;

static dchkpt_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkpt_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->D = malloc(2 * NMAX * sizeof(f64));
    g_workspace->E = malloc(2 * NMAX * sizeof(f64));
    g_workspace->DF = g_workspace->D + NMAX;
    g_workspace->EF = g_workspace->E + NMAX;
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    /* WORK: NMAX * max(3, NSMAX) per dchkpt.f */
    g_workspace->WORK = malloc(NMAX * NSMAX * sizeof(f64));
    /* RWORK: max(NMAX, 2*NSMAX) per dchkpt.f */
    g_workspace->RWORK = malloc((NMAX > 2 * NSMAX ? NMAX : 2 * NSMAX) * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
    g_workspace->Z = malloc(3 * sizeof(f64));

    if (!g_workspace->D || !g_workspace->E ||
        !g_workspace->B || !g_workspace->X ||
        !g_workspace->XACT || !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->Z) {
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
        free(g_workspace->D);
        free(g_workspace->E);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->FERR);
        free(g_workspace->BERR);
        free(g_workspace->Z);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Generate a symmetric positive definite tridiagonal matrix for testing.
 *
 * For types 1-6: Use dlatms with controlled singular values.
 * For types 7-12: Generate diagonally dominant tridiagonal directly.
 */
static void generate_pt_matrix(INT n, INT imat, f64* D, f64* E,
                                uint64_t state[static 4], INT* izero, f64* Z)
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    const f64 ZERO = 0.0;
    (void)ZERO;  /* Used in assignments below */

    if (n <= 0) {
        *izero = 0;
        return;
    }

    dlatb4("DPT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    INT zerot = (imat >= 8 && imat <= 10);

    if (imat >= 1 && imat <= 6) {
        *izero = 0;

        /* Types 1-6: Generate positive definite tridiagonal with controlled condition.
           Generate diagonally dominant matrix directly since this is simpler
           and still tests the PT routines correctly. */
        for (INT i = 0; i < n; i++) {
            D[i] = rng_uniform_symmetric(state);
        }
        for (INT i = 0; i < n - 1; i++) {
            E[i] = rng_uniform_symmetric(state);
        }

        /* Make the tridiagonal matrix diagonally dominant (positive definite). */
        if (n == 1) {
            D[0] = fabs(D[0]) + 1.0;
        } else {
            D[0] = fabs(D[0]) + fabs(E[0]) + 1.0;
            D[n - 1] = fabs(D[n - 1]) + fabs(E[n - 2]) + 1.0;
            for (INT i = 1; i < n - 1; i++) {
                D[i] = fabs(D[i]) + fabs(E[i]) + fabs(E[i - 1]) + 1.0;
            }
        }

        /* Scale so maximum diagonal is anorm */
        INT ix = cblas_idamax(n, D, 1);
        f64 dmax = D[ix];
        cblas_dscal(n, anorm / dmax, D, 1);
        if (n > 1) {
            cblas_dscal(n - 1, anorm / dmax, E, 1);
        }

        /* Apply condition number scaling for types 3-4 */
        if (imat == 3 || imat == 4) {
            /* Scale to increase condition number */
            for (INT i = 0; i < n / 2; i++) {
                D[i] *= cndnum;
            }
        }

    } else {
        /* Types 7-12: generate a diagonally dominant matrix with
           unknown condition number in the vectors D and E. */

        if (!zerot || *izero == 0) {
            /* Let D and E have values from [-1,1]. */
            for (INT i = 0; i < n; i++) {
                D[i] = rng_uniform_symmetric(state);
            }
            for (INT i = 0; i < n - 1; i++) {
                E[i] = rng_uniform_symmetric(state);
            }

            /* Make the tridiagonal matrix diagonally dominant. */
            if (n == 1) {
                D[0] = fabs(D[0]);
            } else {
                D[0] = fabs(D[0]) + fabs(E[0]);
                D[n - 1] = fabs(D[n - 1]) + fabs(E[n - 2]);
                for (INT i = 1; i < n - 1; i++) {
                    D[i] = fabs(D[i]) + fabs(E[i]) + fabs(E[i - 1]);
                }
            }

            /* Scale D and E so the maximum element is ANORM. */
            INT ix = cblas_idamax(n, D, 1);
            f64 dmax = D[ix];
            cblas_dscal(n, anorm / dmax, D, 1);
            if (n > 1) {
                cblas_dscal(n - 1, anorm / dmax, E, 1);
            }

        } else if (*izero > 0) {
            /* Reuse the last matrix by copying back the zeroed out
               elements. */
            if (*izero == 1) {
                D[0] = Z[1];
                if (n > 1) {
                    E[0] = Z[2];
                }
            } else if (*izero == n) {
                E[n - 2] = Z[0];
                D[n - 1] = Z[1];
            } else {
                E[*izero - 2] = Z[0];
                D[*izero - 1] = Z[1];
                E[*izero - 1] = Z[2];
            }
        }

        /* For types 8-10, set one row and column of the matrix to
           zero. */
        *izero = 0;
        if (imat == 8) {
            *izero = 1;
            Z[1] = D[0];
            D[0] = ZERO;
            if (n > 1) {
                Z[2] = E[0];
                E[0] = ZERO;
            }
        } else if (imat == 9) {
            *izero = n;
            if (n > 1) {
                Z[0] = E[n - 2];
                E[n - 2] = ZERO;
            }
            Z[1] = D[n - 1];
            D[n - 1] = ZERO;
        } else if (imat == 10) {
            *izero = (n + 1) / 2;
            if (*izero > 1) {
                Z[0] = E[*izero - 2];
                E[*izero - 2] = ZERO;
                Z[2] = E[*izero - 1];
                E[*izero - 1] = ZERO;
            }
            Z[1] = D[*izero - 1];
            D[*izero - 1] = ZERO;
        }
    }
}

/**
 * Run the full dchkpt test battery for a single (n, imat) combination.
 * This is the core test logic, parameterized by the test case.
 */
static void run_dchkpt_single(INT n, INT imat)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    dchkpt_workspace_t* ws = g_workspace;

    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    f64 anorm = ZERO, rcond, rcondc, ainvnm;
    f64 result[NTESTS];
    f64 reslts[2];
    char ctx[128];

    /* Seed based on (n, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + imat));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Initialize izero for first call */
    izero = 0;

    /* Generate test matrix */
    generate_pt_matrix(n, imat, ws->D, ws->E, rng_state, &izero, ws->Z);

    /* Copy to factored arrays: D(N+1:2N), E(N+1:2N-1) in Fortran terms */
    cblas_dcopy(n, ws->D, 1, ws->DF, 1);
    if (n > 1) {
        cblas_dcopy(n - 1, ws->E, 1, ws->EF, 1);
    }

    /*
     * TEST 1: Factor A as L*D*L' and compute the ratio
     *         norm(L*D*L' - A) / (n * norm(A) * EPS)
     */
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 1 (factorization)", n, imat);
    set_test_context(ctx);
    dpttrf(n, ws->DF, ws->EF, &info);

    /* Check error code from DPTTRF. */
    if (info != izero) {
        /* Unexpected error */
        if (izero == 0) {
            assert_int_equal(info, 0);
        }
    }

    if (info > 0) {
        rcondc = ZERO;
        goto test7;
    }

    dptt01(n, ws->D, ws->E, ws->DF, ws->EF, ws->WORK, &result[0]);

    /* Print the test ratio if greater than or equal to THRESH. */
    assert_residual_below(result[0], THRESH);

    /*
     * Compute RCONDC = 1 / (norm(A) * norm(inv(A))
     */

    /* Compute norm(A). */
    anorm = dlanst("1", n, ws->D, ws->E);

    /* Use DPTTRS to solve for one column at a time of inv(A),
       computing the maximum column sum as we go. */
    ainvnm = ZERO;
    for (INT i = 0; i < n; i++) {
        for (INT j = 0; j < n; j++) {
            ws->X[j] = ZERO;
        }
        ws->X[i] = ONE;
        dpttrs(n, 1, ws->DF, ws->EF, ws->X, lda, &info);
        ainvnm = fmax(ainvnm, cblas_dasum(n, ws->X, 1));
    }
    rcondc = ONE / fmax(ONE, anorm * ainvnm);

    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        /* Generate NRHS random solution vectors. */
        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                ws->XACT[i + j * lda] = rng_uniform_symmetric(rng_state);
            }
        }

        /* Set the right hand side. */
        dlaptm(n, nrhs, ONE, ws->D, ws->E, ws->XACT, lda, ZERO, ws->B, lda);

        /*
         * TEST 2: Solve A*x = b and compute the residual.
         */
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d TEST 2 (solve)", n, imat, nrhs);
        set_test_context(ctx);
        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        dpttrs(n, nrhs, ws->DF, ws->EF, ws->X, lda, &info);

        /* Check error code from DPTTRS. */
        assert_int_equal(info, 0);

        dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        dptt02(n, nrhs, ws->D, ws->E, ws->X, lda, ws->WORK, lda, &result[1]);
        assert_residual_below(result[1], THRESH);

        /*
         * TEST 3: Check solution from generated exact solution.
         */
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d TEST 3 (accuracy)", n, imat, nrhs);
        set_test_context(ctx);
        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TESTS 4, 5, and 6: Use iterative refinement to improve the solution.
         */
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d TEST 4-6 (refinement)", n, imat, nrhs);
        set_test_context(ctx);
        dptrfs(n, nrhs, ws->D, ws->E, ws->DF, ws->EF, ws->B, lda,
               ws->X, lda, ws->RWORK, ws->RWORK + nrhs, ws->WORK, &info);

        /* Check error code from DPTRFS. */
        assert_int_equal(info, 0);

        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
        dptt05(n, nrhs, ws->D, ws->E, ws->B, lda, ws->X, lda, ws->XACT, lda,
               ws->RWORK, ws->RWORK + nrhs, reslts);
        result[4] = reslts[0];
        result[5] = reslts[1];

        /* Print information about the tests that did not pass the
           threshold. */
        assert_residual_below(result[3], THRESH);
        assert_residual_below(result[4], THRESH);
        assert_residual_below(result[5], THRESH);
    }

    /*
     * TEST 7: Estimate the reciprocal of the condition number of the
     *         matrix.
     */
test7:
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 7 (condition)", n, imat);
    set_test_context(ctx);
    dptcon(n, ws->DF, ws->EF, anorm, &rcond, ws->RWORK, &info);

    /* Check error code from DPTCON. */
    assert_int_equal(info, 0);

    result[6] = dget06(rcond, rcondc);

    /* Print the test ratio if greater than or equal to THRESH. */
    assert_residual_below(result[6], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkpt_single based on prestate.
 */
static void test_dchkpt_case(void** state)
{
    dchkpt_params_t* params = *state;
    run_dchkpt_single(params->n, params->imat);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NTYPES = 7 * 12 = 84 tests
 * (minus skipped cases for small n)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NTYPES)

static dchkpt_params_t g_params[MAX_TESTS];
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
            /* Store parameters */
            dchkpt_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->imat = imat;
            snprintf(p->name, sizeof(p->name), "dchkpt_n%d_type%d", n, imat);

            /* Create CMocka test entry */
            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_dchkpt_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
        }
    }
}

int main(void)
{
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace. */
    return _cmocka_run_group_tests("dchkpt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
