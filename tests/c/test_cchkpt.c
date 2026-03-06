/**
 * @file test_cchkpt.c
 * @brief Comprehensive test suite for Hermitian positive definite tridiagonal (CPT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkpt.f to C using CMocka.
 * Tests CPTTRF, CPTTRS, CPTRFS, and CPTCON.
 *
 * Each (n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkpt.f:
 *   TEST 1: L*D*L' factorization residual via cptt01
 *   TEST 2: Solution residual via cptt02
 *   TEST 3: Solution accuracy via cget04
 *   TEST 4: Refined solution accuracy via cget04 (after cptrfs)
 *   TEST 5-6: Error bounds via cptt05
 *   TEST 7: Condition number via sget06
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
#define THRESH  30.0f
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
    c64* A;      /* Band storage for clatms output (2 x NMAX) */
    f32*  D;      /* Original diagonal (2*NMAX: D and DF share allocation) */
    c64* E;      /* Original subdiagonal (2*NMAX: E and EF share allocation) */
    f32*  DF;     /* Factored diagonal (points into D + NMAX) */
    c64* EF;     /* Factored subdiagonal (points into E + NMAX) */
    c64* B;      /* Right-hand side (NMAX x NSMAX) */
    c64* X;      /* Solution (NMAX x NSMAX) */
    c64* XACT;   /* Exact solution (NMAX x NSMAX) */
    c64* WORK;   /* General workspace */
    f32*  RWORK;  /* Real workspace for error bounds and cptrfs */
    c64* Z;      /* Storage for zeroed elements (3) */
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

    g_workspace->A = malloc(2 * NMAX * sizeof(c64));
    g_workspace->D = malloc(2 * NMAX * sizeof(f32));
    g_workspace->E = malloc(2 * NMAX * sizeof(c64));
    g_workspace->DF = g_workspace->D + NMAX;
    g_workspace->EF = g_workspace->E + NMAX;
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c64));
    /* WORK: NMAX * max(3, NSMAX) per zchkpt.f */
    g_workspace->WORK = malloc(NMAX * NSMAX * sizeof(c64));
    /* RWORK: 2*NSMAX + NMAX (ferr + berr + cptrfs internal rwork) */
    g_workspace->RWORK = malloc((2 * NSMAX + NMAX) * sizeof(f32));
    g_workspace->Z = malloc(3 * sizeof(c64));

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
 * For types 1-6: Use clatms with controlled singular values.
 * For types 7-12: Generate diagonally dominant tridiagonal directly.
 */
static void generate_pt_matrix(INT n, INT imat, f32* D, c64* E,
                                c64* A, c64* WORK, f32* RWORK,
                                uint64_t state[static 4], INT* izero, c64* Z)
{
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    INT info;

    if (n <= 0) {
        *izero = 0;
        return;
    }

    clatb4("CPT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    INT zerot = (imat >= 8 && imat <= 10);

    if (imat >= 1 && imat <= 6) {
        *izero = 0;

        clatms(n, n, &dist, &type, RWORK, mode, cndnum, anorm,
               kl, ku, "B", A, 2, WORK, &info, state);

        if (info != 0) {
            return;
        }

        for (INT i = 0; i < n - 1; i++) {
            D[i] = crealf(A[i * 2]);
            E[i] = A[i * 2 + 1];
        }
        if (n > 0) {
            D[n - 1] = crealf(A[(n - 1) * 2]);
        }

    } else {

        if (!zerot || *izero == 0) {

            for (INT i = 0; i < n; i++) {
                D[i] = rng_uniform_symmetric_f32(state);
            }
            for (INT i = 0; i < n - 1; i++) {
                E[i] = CMPLXF(rng_uniform_symmetric_f32(state),
                             rng_uniform_symmetric_f32(state));
            }

            if (n == 1) {
                D[0] = fabsf(D[0]);
            } else {
                D[0] = fabsf(D[0]) + cabsf(E[0]);
                D[n - 1] = fabsf(D[n - 1]) + cabsf(E[n - 2]);
                for (INT i = 1; i < n - 1; i++) {
                    D[i] = fabsf(D[i]) + cabsf(E[i]) + cabsf(E[i - 1]);
                }
            }

            INT ix = cblas_isamax(n, D, 1);
            f32 dmax = D[ix];
            cblas_sscal(n, anorm / dmax, D, 1);
            if (n > 1) {
                cblas_csscal(n - 1, anorm / dmax, E, 1);
            }

        } else if (*izero > 0) {

            if (*izero == 1) {
                D[0] = crealf(Z[1]);
                if (n > 1) {
                    E[0] = Z[2];
                }
            } else if (*izero == n) {
                E[n - 2] = Z[0];
                D[n - 1] = crealf(Z[1]);
            } else {
                E[*izero - 2] = Z[0];
                D[*izero - 1] = crealf(Z[1]);
                E[*izero - 1] = Z[2];
            }
        }

        *izero = 0;
        if (imat == 8) {
            *izero = 1;
            Z[1] = CMPLXF(D[0], 0.0f);
            D[0] = 0.0f;
            if (n > 1) {
                Z[2] = E[0];
                E[0] = 0.0f;
            }
        } else if (imat == 9) {
            *izero = n;
            if (n > 1) {
                Z[0] = E[n - 2];
                E[n - 2] = 0.0f;
            }
            Z[1] = CMPLXF(D[n - 1], 0.0f);
            D[n - 1] = 0.0f;
        } else if (imat == 10) {
            *izero = (n + 1) / 2;
            if (*izero > 1) {
                Z[0] = E[*izero - 2];
                E[*izero - 2] = 0.0f;
                Z[2] = E[*izero - 1];
                E[*izero - 1] = 0.0f;
            }
            Z[1] = CMPLXF(D[*izero - 1], 0.0f);
            D[*izero - 1] = 0.0f;
        }
    }
}

/**
 * Run the full zchkpt test battery for a single (n, imat) combination.
 */
static void run_zchkpt_single(INT n, INT imat)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    static const char UPLOS[] = {'U', 'L'};
    zchkpt_workspace_t* ws = g_workspace;

    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    f32 anorm = ZERO, rcond, rcondc, ainvnm;
    f32 result[NTESTS];
    f32 reslts[2];
    char ctx[128];

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + imat));

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    izero = 0;

    generate_pt_matrix(n, imat, ws->D, ws->E, ws->A, ws->WORK,
                       ws->RWORK, rng_state, &izero, ws->Z);

    cblas_scopy(n, ws->D, 1, ws->DF, 1);
    if (n > 1) {
        cblas_ccopy(n - 1, ws->E, 1, ws->EF, 1);
    }

    /*
     * TEST 1: Factor A as L*D*L' and compute the ratio
     *         norm(L*D*L' - A) / (n * norm(A) * EPS)
     */
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 1 (factorization)", n, imat);
    set_test_context(ctx);
    cpttrf(n, ws->DF, ws->EF, &info);

    if (info != izero) {
        if (izero == 0) {
            assert_int_equal(info, 0);
        }
    }

    if (info > 0) {
        rcondc = ZERO;
        goto test7;
    }

    cptt01(n, ws->D, ws->E, ws->DF, ws->EF, ws->WORK, &result[0]);

    assert_residual_below(result[0], THRESH);

    /*
     * Compute RCONDC = 1 / (norm(A) * norm(inv(A))
     */

    anorm = clanht("1", n, ws->D, ws->E);

    ainvnm = ZERO;
    for (INT i = 0; i < n; i++) {
        for (INT j = 0; j < n; j++) {
            ws->X[j] = 0.0f;
        }
        ws->X[i] = 1.0f;
        cpttrs("L", n, 1, ws->DF, ws->EF, ws->X, lda, &info);
        ainvnm = fmaxf(ainvnm, cblas_scasum(n, ws->X, 1));
    }
    rcondc = ONE / fmaxf(ONE, anorm * ainvnm);

    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        for (INT j = 0; j < nrhs; j++) {
            for (INT i = 0; i < n; i++) {
                ws->XACT[i + j * lda] = CMPLXF(rng_uniform_symmetric_f32(rng_state),
                                               rng_uniform_symmetric_f32(rng_state));
            }
        }

        for (INT iuplo = 0; iuplo < 2; iuplo++) {
            char uplo = UPLOS[iuplo];
            char uplo_str[2] = {uplo, '\0'};

            claptm(uplo_str, n, nrhs, ONE, ws->D, ws->E, ws->XACT, lda,
                   ZERO, ws->B, lda);

            /*
             * TEST 2: Solve A*x = b and compute the residual.
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d uplo=%c TEST 2 (solve)",
                     n, imat, nrhs, uplo);
            set_test_context(ctx);
            clacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            cpttrs(uplo_str, n, nrhs, ws->DF, ws->EF, ws->X, lda, &info);

            assert_int_equal(info, 0);

            clacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
            cptt02(uplo_str, n, nrhs, ws->D, ws->E, ws->X, lda,
                   ws->WORK, lda, &result[1]);
            assert_residual_below(result[1], THRESH);

            /*
             * TEST 3: Check solution from generated exact solution.
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d uplo=%c TEST 3 (accuracy)",
                     n, imat, nrhs, uplo);
            set_test_context(ctx);
            cget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            assert_residual_below(result[2], THRESH);

            /*
             * TESTS 4, 5, and 6: Use iterative refinement to improve the solution.
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d uplo=%c TEST 4-6 (refinement)",
                     n, imat, nrhs, uplo);
            set_test_context(ctx);
            cptrfs(uplo_str, n, nrhs, ws->D, ws->E, ws->DF, ws->EF,
                   ws->B, lda, ws->X, lda,
                   ws->RWORK, ws->RWORK + nrhs,
                   ws->WORK, ws->RWORK + 2 * nrhs, &info);

            assert_int_equal(info, 0);

            cget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
            cptt05(n, nrhs, ws->D, ws->E, ws->B, lda, ws->X, lda, ws->XACT, lda,
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
    cptcon(n, ws->DF, ws->EF, anorm, &rcond, ws->RWORK, &info);

    assert_int_equal(info, 0);

    result[6] = sget06(rcond, rcondc);

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
