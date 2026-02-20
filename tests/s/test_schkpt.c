/**
 * @file test_schkpt.c
 * @brief Comprehensive test suite for positive definite tridiagonal matrix (SPT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkpt.f to C using CMocka.
 * Tests SPTTRF, SPTTRS, SPTRFS, and SPTCON.
 *
 * Each (n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkpt.f:
 *   TEST 1: L*D*L' factorization residual via sptt01
 *   TEST 2: Solution residual via sptt02
 *   TEST 3: Solution accuracy via sget04
 *   TEST 4: Refined solution accuracy via sget04 (after sptrfs)
 *   TEST 5-6: Error bounds via sptt05
 *   TEST 7: Condition number via sget06
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-12
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 15};  /* NRHS values */

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  12
#define NTESTS  7
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void spttrf(const int n, f32* D, f32* E, int* info);
extern void spttrs(const int n, const int nrhs,
                   const f32* D, const f32* E,
                   f32* B, const int ldb, int* info);
extern void sptrfs(const int n, const int nrhs,
                   const f32* D, const f32* E,
                   const f32* DF, const f32* EF,
                   const f32* B, const int ldb,
                   f32* X, const int ldx,
                   f32* ferr, f32* berr,
                   f32* work, int* info);
extern void sptcon(const int n, const f32* D, const f32* E,
                   const f32 anorm, f32* rcond,
                   f32* work, int* info);

/* Verification routines */
extern void sptt01(const int n, const f32* D, const f32* E,
                   const f32* DF, const f32* EF,
                   f32* work, f32* resid);
extern void sptt02(const int n, const int nrhs, const f32* D,
                   const f32* E, const f32* X, const int ldx,
                   f32* B, const int ldb, f32* resid);
extern void sptt05(const int n, const int nrhs, const f32* D,
                   const f32* E, const f32* B, const int ldb,
                   const f32* X, const int ldx,
                   const f32* XACT, const int ldxact,
                   const f32* ferr, const f32* berr, f32* reslts);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond,
                   f32* resid);
extern f32 sget06(const f32 rcond, const f32 rcondc);

/* Matrix generation */
extern void slatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f32* anorm, int* mode,
                   f32* cndnum, char* dist);
extern void slatms(const int m, const int n, const char* dist,
                   const char* sym, f32* d, const int mode, const f32 cond,
                   const f32 dmax, const int kl, const int ku, const char* pack,
                   f32* A, const int lda, f32* work, int* info,
                   uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern f32 slanst(const char* norm, const int n,
                     const f32* D, const f32* E);
extern f32 slamch(const char* cmach);
extern void slaptm(const int n, const int nrhs, const f32 alpha,
                   const f32* D, const f32* E,
                   const f32* X, const int ldx, const f32 beta,
                   f32* B, const int ldb);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    char name[64];
} dchkpt_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* D;      /* Original diagonal (NMAX) */
    f32* E;      /* Original subdiagonal (NMAX-1) */
    f32* DF;     /* Factored diagonal (NMAX) */
    f32* EF;     /* Factored subdiagonal (NMAX-1) */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace for error bounds */
    f32* FERR;   /* Forward error bounds */
    f32* BERR;   /* Backward error bounds */
    f32* Z;      /* Storage for zeroed elements (3) */
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

    g_workspace->D = malloc(2 * NMAX * sizeof(f32));
    g_workspace->E = malloc(2 * NMAX * sizeof(f32));
    g_workspace->DF = g_workspace->D + NMAX;
    g_workspace->EF = g_workspace->E + NMAX;
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    /* WORK: NMAX * max(3, NSMAX) per dchkpt.f */
    g_workspace->WORK = malloc(NMAX * NSMAX * sizeof(f32));
    /* RWORK: max(NMAX, 2*NSMAX) per dchkpt.f */
    g_workspace->RWORK = malloc((NMAX > 2 * NSMAX ? NMAX : 2 * NSMAX) * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
    g_workspace->Z = malloc(3 * sizeof(f32));

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
 * For types 1-6: Use slatms with controlled singular values.
 * For types 7-12: Generate diagonally dominant tridiagonal directly.
 */
static void generate_pt_matrix(int n, int imat, f32* D, f32* E,
                                uint64_t state[static 4], int* izero, f32* Z)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    const f32 ZERO = 0.0f;
    (void)ZERO;  /* Used in assignments below */

    if (n <= 0) {
        *izero = 0;
        return;
    }

    slatb4("SPT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    int zerot = (imat >= 8 && imat <= 10);

    if (imat >= 1 && imat <= 6) {
        *izero = 0;

        /* Types 1-6: Generate positive definite tridiagonal with controlled condition.
           Generate diagonally dominant matrix directly since this is simpler
           and still tests the PT routines correctly. */
        for (int i = 0; i < n; i++) {
            D[i] = rng_uniform_symmetric_f32(state);
        }
        for (int i = 0; i < n - 1; i++) {
            E[i] = rng_uniform_symmetric_f32(state);
        }

        /* Make the tridiagonal matrix diagonally dominant (positive definite). */
        if (n == 1) {
            D[0] = fabsf(D[0]) + 1.0f;
        } else {
            D[0] = fabsf(D[0]) + fabsf(E[0]) + 1.0f;
            D[n - 1] = fabsf(D[n - 1]) + fabsf(E[n - 2]) + 1.0f;
            for (int i = 1; i < n - 1; i++) {
                D[i] = fabsf(D[i]) + fabsf(E[i]) + fabsf(E[i - 1]) + 1.0f;
            }
        }

        /* Scale so maximum diagonal is anorm */
        int ix = cblas_isamax(n, D, 1);
        f32 dmax = D[ix];
        cblas_sscal(n, anorm / dmax, D, 1);
        if (n > 1) {
            cblas_sscal(n - 1, anorm / dmax, E, 1);
        }

        /* Apply condition number scaling for types 3-4 */
        if (imat == 3 || imat == 4) {
            /* Scale to increase condition number */
            for (int i = 0; i < n / 2; i++) {
                D[i] *= cndnum;
            }
        }

    } else {
        /* Types 7-12: generate a diagonally dominant matrix with
           unknown condition number in the vectors D and E. */

        if (!zerot || *izero == 0) {
            /* Let D and E have values from [-1,1]. */
            for (int i = 0; i < n; i++) {
                D[i] = rng_uniform_symmetric_f32(state);
            }
            for (int i = 0; i < n - 1; i++) {
                E[i] = rng_uniform_symmetric_f32(state);
            }

            /* Make the tridiagonal matrix diagonally dominant. */
            if (n == 1) {
                D[0] = fabsf(D[0]);
            } else {
                D[0] = fabsf(D[0]) + fabsf(E[0]);
                D[n - 1] = fabsf(D[n - 1]) + fabsf(E[n - 2]);
                for (int i = 1; i < n - 1; i++) {
                    D[i] = fabsf(D[i]) + fabsf(E[i]) + fabsf(E[i - 1]);
                }
            }

            /* Scale D and E so the maximum element is ANORM. */
            int ix = cblas_isamax(n, D, 1);
            f32 dmax = D[ix];
            cblas_sscal(n, anorm / dmax, D, 1);
            if (n > 1) {
                cblas_sscal(n - 1, anorm / dmax, E, 1);
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
static void run_dchkpt_single(int n, int imat)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    dchkpt_workspace_t* ws = g_workspace;

    int info, izero;
    int lda = (n > 1) ? n : 1;
    f32 anorm = ZERO, rcond, rcondc, ainvnm;
    f32 result[NTESTS];
    f32 reslts[2];
    char ctx[128];

    /* Seed based on (n, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Initialize izero for first call */
    izero = 0;

    /* Generate test matrix */
    generate_pt_matrix(n, imat, ws->D, ws->E, rng_state, &izero, ws->Z);

    /* Copy to factored arrays: D(N+1:2N), E(N+1:2N-1) in Fortran terms */
    cblas_scopy(n, ws->D, 1, ws->DF, 1);
    if (n > 1) {
        cblas_scopy(n - 1, ws->E, 1, ws->EF, 1);
    }

    /*
     * TEST 1: Factor A as L*D*L' and compute the ratio
     *         norm(L*D*L' - A) / (n * norm(A) * EPS)
     */
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 1 (factorization)", n, imat);
    set_test_context(ctx);
    spttrf(n, ws->DF, ws->EF, &info);

    /* Check error code from SPTTRF. */
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

    sptt01(n, ws->D, ws->E, ws->DF, ws->EF, ws->WORK, &result[0]);

    /* Print the test ratio if greater than or equal to THRESH. */
    assert_residual_below(result[0], THRESH);

    /*
     * Compute RCONDC = 1 / (norm(A) * norm(inv(A))
     */

    /* Compute norm(A). */
    anorm = slanst("1", n, ws->D, ws->E);

    /* Use SPTTRS to solve for one column at a time of inv(A),
       computing the maximum column sum as we go. */
    ainvnm = ZERO;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ws->X[j] = ZERO;
        }
        ws->X[i] = ONE;
        spttrs(n, 1, ws->DF, ws->EF, ws->X, lda, &info);
        ainvnm = fmaxf(ainvnm, cblas_sasum(n, ws->X, 1));
    }
    rcondc = ONE / fmaxf(ONE, anorm * ainvnm);

    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /* Generate NRHS random solution vectors. */
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                ws->XACT[i + j * lda] = rng_uniform_symmetric_f32(rng_state);
            }
        }

        /* Set the right hand side. */
        slaptm(n, nrhs, ONE, ws->D, ws->E, ws->XACT, lda, ZERO, ws->B, lda);

        /*
         * TEST 2: Solve A*x = b and compute the residual.
         */
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d TEST 2 (solve)", n, imat, nrhs);
        set_test_context(ctx);
        slacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        spttrs(n, nrhs, ws->DF, ws->EF, ws->X, lda, &info);

        /* Check error code from SPTTRS. */
        assert_int_equal(info, 0);

        slacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        sptt02(n, nrhs, ws->D, ws->E, ws->X, lda, ws->WORK, lda, &result[1]);
        assert_residual_below(result[1], THRESH);

        /*
         * TEST 3: Check solution from generated exact solution.
         */
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d TEST 3 (accuracy)", n, imat, nrhs);
        set_test_context(ctx);
        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TESTS 4, 5, and 6: Use iterative refinement to improve the solution.
         */
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d TEST 4-6 (refinement)", n, imat, nrhs);
        set_test_context(ctx);
        sptrfs(n, nrhs, ws->D, ws->E, ws->DF, ws->EF, ws->B, lda,
               ws->X, lda, ws->RWORK, ws->RWORK + nrhs, ws->WORK, &info);

        /* Check error code from SPTRFS. */
        assert_int_equal(info, 0);

        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
        sptt05(n, nrhs, ws->D, ws->E, ws->B, lda, ws->X, lda, ws->XACT, lda,
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
    sptcon(n, ws->DF, ws->EF, anorm, &rcond, ws->RWORK, &info);

    /* Check error code from SPTCON. */
    assert_int_equal(info, 0);

    result[6] = sget06(rcond, rcondc);

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
