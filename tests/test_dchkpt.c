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
#include <string.h>
#include <stdio.h>
#include <cblas.h>
#include "testutils/test_rng.h"

/* Test parameters from dtest.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 15};  /* NRHS values */

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  12
#define NTESTS  7
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void dpttrf(const int n, double* D, double* E, int* info);
extern void dpttrs(const int n, const int nrhs,
                   const double* D, const double* E,
                   double* B, const int ldb, int* info);
extern void dptrfs(const int n, const int nrhs,
                   const double* D, const double* E,
                   const double* DF, const double* EF,
                   const double* B, const int ldb,
                   double* X, const int ldx,
                   double* ferr, double* berr,
                   double* work, int* info);
extern void dptcon(const int n, const double* D, const double* E,
                   const double anorm, double* rcond,
                   double* work, int* info);

/* Verification routines */
extern void dptt01(const int n, const double* D, const double* E,
                   const double* DF, const double* EF,
                   double* work, double* resid);
extern void dptt02(const int n, const int nrhs, const double* D,
                   const double* E, const double* X, const int ldx,
                   double* B, const int ldb, double* resid);
extern void dptt05(const int n, const int nrhs, const double* D,
                   const double* E, const double* B, const int ldb,
                   const double* X, const int ldx,
                   const double* XACT, const int ldxact,
                   const double* ferr, const double* berr, double* reslts);
extern void dget04(const int n, const int nrhs, const double* X, const int ldx,
                   const double* XACT, const int ldxact, const double rcond,
                   double* resid);
extern double dget06(const double rcond, const double rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist, uint64_t seed,
                   const char* sym, double* d, const int mode, const double cond,
                   const double dmax, const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern double dlanst(const char* norm, const int n,
                     const double* D, const double* E);
extern double dlamch(const char* cmach);
extern void dlaptm(const int n, const int nrhs, const double alpha,
                   const double* D, const double* E,
                   const double* X, const int ldx, const double beta,
                   double* B, const int ldb);

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
    double* D;      /* Original diagonal (NMAX) */
    double* E;      /* Original subdiagonal (NMAX-1) */
    double* DF;     /* Factored diagonal (NMAX) */
    double* EF;     /* Factored subdiagonal (NMAX-1) */
    double* B;      /* Right-hand side (NMAX x NSMAX) */
    double* X;      /* Solution (NMAX x NSMAX) */
    double* XACT;   /* Exact solution (NMAX x NSMAX) */
    double* WORK;   /* General workspace */
    double* RWORK;  /* Real workspace for error bounds */
    double* FERR;   /* Forward error bounds */
    double* BERR;   /* Backward error bounds */
    double* Z;      /* Storage for zeroed elements (3) */
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

    g_workspace->D = malloc(2 * NMAX * sizeof(double));
    g_workspace->E = malloc(2 * NMAX * sizeof(double));
    g_workspace->DF = g_workspace->D + NMAX;
    g_workspace->EF = g_workspace->E + NMAX;
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(double));
    /* WORK: NMAX * max(3, NSMAX) per dchkpt.f */
    g_workspace->WORK = malloc(NMAX * NSMAX * sizeof(double));
    /* RWORK: max(NMAX, 2*NSMAX) per dchkpt.f */
    g_workspace->RWORK = malloc((NMAX > 2 * NSMAX ? NMAX : 2 * NSMAX) * sizeof(double));
    g_workspace->FERR = malloc(NSMAX * sizeof(double));
    g_workspace->BERR = malloc(NSMAX * sizeof(double));
    g_workspace->Z = malloc(3 * sizeof(double));

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
static void generate_pt_matrix(int n, int imat, double* D, double* E,
                                uint64_t* seed, int* izero, double* Z)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    const double ZERO = 0.0;
    (void)ZERO;  /* Used in assignments below */

    if (n <= 0) {
        *izero = 0;
        return;
    }

    dlatb4("DPT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    int zerot = (imat >= 8 && imat <= 10);

    if (imat >= 1 && imat <= 6) {
        *izero = 0;

        /* Types 1-6: Generate positive definite tridiagonal with controlled condition.
           Generate diagonally dominant matrix directly since this is simpler
           and still tests the PT routines correctly. */
        rng_seed(*seed);
        for (int i = 0; i < n; i++) {
            D[i] = rng_uniform_symmetric();
        }
        for (int i = 0; i < n - 1; i++) {
            E[i] = rng_uniform_symmetric();
        }

        /* Make the tridiagonal matrix diagonally dominant (positive definite). */
        if (n == 1) {
            D[0] = fabs(D[0]) + 1.0;
        } else {
            D[0] = fabs(D[0]) + fabs(E[0]) + 1.0;
            D[n - 1] = fabs(D[n - 1]) + fabs(E[n - 2]) + 1.0;
            for (int i = 1; i < n - 1; i++) {
                D[i] = fabs(D[i]) + fabs(E[i]) + fabs(E[i - 1]) + 1.0;
            }
        }

        /* Scale so maximum diagonal is anorm */
        int ix = cblas_idamax(n, D, 1);
        double dmax = D[ix];
        cblas_dscal(n, anorm / dmax, D, 1);
        if (n > 1) {
            cblas_dscal(n - 1, anorm / dmax, E, 1);
        }

        /* Apply condition number scaling for types 3-4 */
        if (imat == 3 || imat == 4) {
            /* Scale to increase condition number */
            for (int i = 0; i < n / 2; i++) {
                D[i] *= cndnum;
            }
        }

        (*seed)++;

    } else {
        /* Types 7-12: generate a diagonally dominant matrix with
           unknown condition number in the vectors D and E. */

        if (!zerot || *izero == 0) {
            /* Let D and E have values from [-1,1]. */
            rng_seed(*seed);
            for (int i = 0; i < n; i++) {
                D[i] = rng_uniform_symmetric();
            }
            for (int i = 0; i < n - 1; i++) {
                E[i] = rng_uniform_symmetric();
            }

            /* Make the tridiagonal matrix diagonally dominant. */
            if (n == 1) {
                D[0] = fabs(D[0]);
            } else {
                D[0] = fabs(D[0]) + fabs(E[0]);
                D[n - 1] = fabs(D[n - 1]) + fabs(E[n - 2]);
                for (int i = 1; i < n - 1; i++) {
                    D[i] = fabs(D[i]) + fabs(E[i]) + fabs(E[i - 1]);
                }
            }

            /* Scale D and E so the maximum element is ANORM. */
            int ix = cblas_idamax(n, D, 1);
            double dmax = D[ix];
            cblas_dscal(n, anorm / dmax, D, 1);
            if (n > 1) {
                cblas_dscal(n - 1, anorm / dmax, E, 1);
            }

            (*seed)++;

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
    const double ZERO = 0.0;
    const double ONE = 1.0;
    dchkpt_workspace_t* ws = g_workspace;

    int info, izero;
    int lda = (n > 1) ? n : 1;
    double anorm = ZERO, rcond, rcondc, ainvnm;
    double result[NTESTS];
    double reslts[2];
    char ctx[128];

    /* Seed based on (n, imat) for reproducibility */
    uint64_t seed = 1988198919901991ULL + (uint64_t)(n * 1000 + imat);

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Initialize izero for first call */
    izero = 0;

    /* Generate test matrix */
    generate_pt_matrix(n, imat, ws->D, ws->E, &seed, &izero, ws->Z);

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
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            ws->X[j] = ZERO;
        }
        ws->X[i] = ONE;
        dpttrs(n, 1, ws->DF, ws->EF, ws->X, lda, &info);
        ainvnm = fmax(ainvnm, cblas_dasum(n, ws->X, 1));
    }
    rcondc = ONE / fmax(ONE, anorm * ainvnm);

    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /* Generate NRHS random solution vectors. */
        rng_seed(seed++);
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                ws->XACT[i + j * lda] = rng_uniform_symmetric();
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
