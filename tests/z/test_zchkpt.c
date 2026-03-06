/**
 * @file test_zchkpt.c
 * @brief Comprehensive test suite for Hermitian positive definite tridiagonal (ZPT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkpt.f to C using CMocka.
 * Tests ZPTTRF, ZPTTRS, ZPTRFS, and ZPTCON.
 *
 * Each (n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkpt.f:
 *   TEST 1: L*D*L' factorization residual via zptt01
 *   TEST 2: Solution residual via zptt02
 *   TEST 3: Solution accuracy via zget04
 *   TEST 4: Refined solution accuracy via zget04 (after zptrfs)
 *   TEST 5-6: Error bounds via zptt05
 *   TEST 7: Condition number via dget06
 *
 * Parameters from ztest.in:
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

/* Test parameters from ztest.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values */

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  12
#define NTESTS  7
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT n;
    INT imat;
    char name[64];
} zchkpt_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    c128* A;      /* Band storage for zlatms output (2 x NMAX) */
    f64*  D;      /* Original diagonal (2*NMAX: D and DF share allocation) */
    c128* E;      /* Original subdiagonal (2*NMAX: E and EF share allocation) */
    f64*  DF;     /* Factored diagonal (points into D + NMAX) */
    c128* EF;     /* Factored subdiagonal (points into E + NMAX) */
    c128* B;      /* Right-hand side (NMAX x NSMAX) */
    c128* X;      /* Solution (NMAX x NSMAX) */
    c128* XACT;   /* Exact solution (NMAX x NSMAX) */
    c128* WORK;   /* General workspace */
    f64*  RWORK;  /* Real workspace for error bounds and zptrfs */
    c128* Z;      /* Storage for zeroed elements (3) */
} zchkpt_workspace_t;

static zchkpt_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkpt_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(2 * NMAX * sizeof(c128));
    g_workspace->D = malloc(2 * NMAX * sizeof(f64));
    g_workspace->E = malloc(2 * NMAX * sizeof(c128));
    g_workspace->DF = g_workspace->D + NMAX;
    g_workspace->EF = g_workspace->E + NMAX;
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c128));
    /* WORK: NMAX * max(3, NSMAX) per zchkpt.f */
    g_workspace->WORK = malloc(NMAX * NSMAX * sizeof(c128));
    /* RWORK: 2*NSMAX + NMAX (ferr + berr + zptrfs internal rwork) */
    g_workspace->RWORK = malloc((2 * NSMAX + NMAX) * sizeof(f64));
    g_workspace->Z = malloc(3 * sizeof(c128));

    if (!g_workspace->A || !g_workspace->D || !g_workspace->E ||
        !g_workspace->B || !g_workspace->X ||
        !g_workspace->XACT || !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->Z) {
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
        free(g_workspace->D);
        free(g_workspace->E);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->Z);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Generate a Hermitian positive definite tridiagonal matrix for testing.
 *
 * For types 1-6: Use zlatms with controlled singular values.
 * For types 7-12: Generate diagonally dominant tridiagonal directly.
 */
static void generate_pt_matrix(INT n, INT imat, f64* D, c128* E,
                                c128* A, c128* WORK, f64* RWORK,
                                uint64_t state[static 4], INT* izero, c128* Z)
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info;

    if (n <= 0) {
        *izero = 0;
        return;
    }

    zlatb4("ZPT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    INT zerot = (imat >= 8 && imat <= 10);

    if (imat >= 1 && imat <= 6) {
        *izero = 0;

        zlatms(n, n, &dist, &type, RWORK, mode, cndnum, anorm,
               kl, ku, "B", A, 2, WORK, &info, state);

        if (info != 0) {
            return;
        }

        for (INT i = 0; i < n - 1; i++) {
            D[i] = creal(A[i * 2]);
            E[i] = A[i * 2 + 1];
        }
        if (n > 0) {
            D[n - 1] = creal(A[(n - 1) * 2]);
        }

    } else {

        if (!zerot || *izero == 0) {

            for (INT i = 0; i < n; i++) {
                D[i] = rng_uniform_symmetric(state);
            }
            for (INT i = 0; i < n - 1; i++) {
                E[i] = CMPLX(rng_uniform_symmetric(state),
                             rng_uniform_symmetric(state));
            }

            if (n == 1) {
                D[0] = fabs(D[0]);
            } else {
                D[0] = fabs(D[0]) + cabs(E[0]);
                D[n - 1] = fabs(D[n - 1]) + cabs(E[n - 2]);
                for (INT i = 1; i < n - 1; i++) {
                    D[i] = fabs(D[i]) + cabs(E[i]) + cabs(E[i - 1]);
                }
            }

            INT ix = cblas_idamax(n, D, 1);
            f64 dmax = D[ix];
            cblas_dscal(n, anorm / dmax, D, 1);
            if (n > 1) {
                cblas_zdscal(n - 1, anorm / dmax, E, 1);
            }

        } else if (*izero > 0) {

            if (*izero == 1) {
                D[0] = creal(Z[1]);
                if (n > 1) {
                    E[0] = Z[2];
                }
            } else if (*izero == n) {
                E[n - 2] = Z[0];
                D[n - 1] = creal(Z[1]);
            } else {
                E[*izero - 2] = Z[0];
                D[*izero - 1] = creal(Z[1]);
                E[*izero - 1] = Z[2];
            }
        }

        *izero = 0;
        if (imat == 8) {
            *izero = 1;
            Z[1] = CMPLX(D[0], 0.0);
            D[0] = 0.0;
            if (n > 1) {
                Z[2] = E[0];
                E[0] = 0.0;
            }
        } else if (imat == 9) {
            *izero = n;
            if (n > 1) {
                Z[0] = E[n - 2];
                E[n - 2] = 0.0;
            }
            Z[1] = CMPLX(D[n - 1], 0.0);
            D[n - 1] = 0.0;
        } else if (imat == 10) {
            *izero = (n + 1) / 2;
            if (*izero > 1) {
                Z[0] = E[*izero - 2];
                E[*izero - 2] = 0.0;
                Z[2] = E[*izero - 1];
                E[*izero - 1] = 0.0;
            }
            Z[1] = CMPLX(D[*izero - 1], 0.0);
            D[*izero - 1] = 0.0;
        }
    }
}

/**
 * Run the full zchkpt test battery for a single (n, imat) combination.
 */
static void run_zchkpt_single(INT n, INT imat)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    static const char UPLOS[] = {'U', 'L'};
    zchkpt_workspace_t* ws = g_workspace;

    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    f64 anorm = ZERO, rcond, rcondc, ainvnm;
    f64 result[NTESTS];
    f64 reslts[2];
    char ctx[128];

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + imat));

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    izero = 0;

    generate_pt_matrix(n, imat, ws->D, ws->E, ws->A, ws->WORK,
                       ws->RWORK, rng_state, &izero, ws->Z);

    cblas_dcopy(n, ws->D, 1, ws->DF, 1);
    if (n > 1) {
        cblas_zcopy(n - 1, ws->E, 1, ws->EF, 1);
    }

    /*
     * TEST 1: Factor A as L*D*L' and compute the ratio
     *         norm(L*D*L' - A) / (n * norm(A) * EPS)
     */
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 1 (factorization)", n, imat);
    set_test_context(ctx);
    zpttrf(n, ws->DF, ws->EF, &info);

    if (info != izero) {
        if (izero == 0) {
            assert_int_equal(info, 0);
        }
    }

    if (info > 0) {
        rcondc = ZERO;
        goto test7;
    }

    zptt01(n, ws->D, ws->E, ws->DF, ws->EF, ws->WORK, &result[0]);

    assert_residual_below(result[0], THRESH);

    /*
     * Compute RCONDC = 1 / (norm(A) * norm(inv(A))
     */

    anorm = zlanht("1", n, ws->D, ws->E);

    ainvnm = ZERO;
    for (INT i = 0; i < n; i++) {
        for (INT j = 0; j < n; j++) {
            ws->X[j] = 0.0;
        }
        ws->X[i] = 1.0;
        zpttrs("L", n, 1, ws->DF, ws->EF, ws->X, lda, &info);
        ainvnm = fmax(ainvnm, cblas_dzasum(n, ws->X, 1));
    }
    rcondc = ONE / fmax(ONE, anorm * ainvnm);

    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                ws->XACT[i + j * lda] = CMPLX(rng_uniform_symmetric(rng_state),
                                               rng_uniform_symmetric(rng_state));
            }
        }

        for (INT iuplo = 0; iuplo < 2; iuplo++) {
            char uplo = UPLOS[iuplo];
            char uplo_str[2] = {uplo, '\0'};

            zlaptm(uplo_str, n, nrhs, ONE, ws->D, ws->E, ws->XACT, lda,
                   ZERO, ws->B, lda);

            /*
             * TEST 2: Solve A*x = b and compute the residual.
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d uplo=%c TEST 2 (solve)",
                     n, imat, nrhs, uplo);
            set_test_context(ctx);
            zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            zpttrs(uplo_str, n, nrhs, ws->DF, ws->EF, ws->X, lda, &info);

            assert_int_equal(info, 0);

            zlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
            zptt02(uplo_str, n, nrhs, ws->D, ws->E, ws->X, lda,
                   ws->WORK, lda, &result[1]);
            assert_residual_below(result[1], THRESH);

            /*
             * TEST 3: Check solution from generated exact solution.
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d uplo=%c TEST 3 (accuracy)",
                     n, imat, nrhs, uplo);
            set_test_context(ctx);
            zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            assert_residual_below(result[2], THRESH);

            /*
             * TESTS 4, 5, and 6: Use iterative refinement to improve the solution.
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d uplo=%c TEST 4-6 (refinement)",
                     n, imat, nrhs, uplo);
            set_test_context(ctx);
            zptrfs(uplo_str, n, nrhs, ws->D, ws->E, ws->DF, ws->EF,
                   ws->B, lda, ws->X, lda,
                   ws->RWORK, ws->RWORK + nrhs,
                   ws->WORK, ws->RWORK + 2 * nrhs, &info);

            assert_int_equal(info, 0);

            zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
            zptt05(n, nrhs, ws->D, ws->E, ws->B, lda, ws->X, lda, ws->XACT, lda,
                   ws->RWORK, ws->RWORK + nrhs, reslts);
            result[4] = reslts[0];
            result[5] = reslts[1];

            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);
        }
    }

    /*
     * TEST 7: Estimate the reciprocal of the condition number of the
     *         matrix.
     */
test7:
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 7 (condition)", n, imat);
    set_test_context(ctx);
    zptcon(n, ws->DF, ws->EF, anorm, &rcond, ws->RWORK, &info);

    assert_int_equal(info, 0);

    result[6] = dget06(rcond, rcondc);

    assert_residual_below(result[6], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_zchkpt_single based on prestate.
 */
static void test_zchkpt_case(void** state)
{
    zchkpt_params_t* params = *state;
    run_zchkpt_single(params->n, params->imat);
}

/*
 * Generate all parameter combinations.
 */

#define MAX_TESTS (NN * NTYPES)

static zchkpt_params_t g_params[MAX_TESTS];
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
            zchkpt_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->imat = imat;
            snprintf(p->name, sizeof(p->name), "zchkpt_n%d_type%d", n, imat);

            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_zchkpt_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
        }
    }
}

int main(void)
{
    build_test_array();

    (void)_cmocka_run_group_tests("zchkpt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
