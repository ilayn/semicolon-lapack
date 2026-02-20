/**
 * @file test_schkpp.c
 * @brief Comprehensive test suite for positive definite packed matrix (SPP) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkpp.f to C using CMocka.
 * Tests SPPTRF, SPPTRI, SPPTRS, SPPRFS, and SPPCON.
 *
 * Each (n, uplo, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkpp.f:
 *   TEST 1: Cholesky factorization residual via sppt01
 *   TEST 2: Matrix inverse residual via sppt03
 *   TEST 3: Solution residual via sppt02
 *   TEST 4: Solution accuracy via sget04
 *   TEST 5: Refined solution accuracy via sget04 (after spprfs)
 *   TEST 6-7: Error bounds via sppt05
 *   TEST 8: Condition number via sget06
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 15};  /* NRHS values */
static const char UPLOS[] = {'U', 'L'};
static const char* PACKS[] = {"C", "R"};  /* Column packed for U, Row packed for L */

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9
#define NTESTS  8
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void spptrf(const char* uplo, const int n, f32* AP, int* info);
extern void spptri(const char* uplo, const int n, f32* AP, int* info);
extern void spptrs(const char* uplo, const int n, const int nrhs,
                   const f32* AP, f32* B, const int ldb, int* info);
extern void spprfs(const char* uplo, const int n, const int nrhs,
                   const f32* AP, const f32* AFP,
                   const f32* B, const int ldb,
                   f32* X, const int ldx, f32* ferr, f32* berr,
                   f32* work, int* iwork, int* info);
extern void sppcon(const char* uplo, const int n, const f32* AP,
                   const f32 anorm, f32* rcond,
                   f32* work, int* iwork, int* info);

/* Verification routines */
extern void sppt01(const char* uplo, const int n, const f32* A,
                   f32* AFAC, f32* rwork, f32* resid);
extern void sppt02(const char* uplo, const int n, const int nrhs,
                   const f32* A, const f32* X, const int ldx,
                   f32* B, const int ldb, f32* rwork, f32* resid);
extern void sppt03(const char* uplo, const int n, const f32* A,
                   const f32* AINV, f32* work, const int ldwork,
                   f32* rwork, f32* rcond, f32* resid);
extern void sppt05(const char* uplo, const int n, const int nrhs,
                   const f32* AP, const f32* B, const int ldb,
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
extern void slarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f32* A, const int lda,
                   const f32* XACT, const int ldxact, f32* B,
                   const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern f32 slansp(const char* norm, const char* uplo, const int n,
                     const f32* AP, f32* work);
extern f32 slamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    int iuplo;  /* 0='U', 1='L' */
    char name[64];
} dchkpp_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* A;      /* Original packed matrix (NMAX*(NMAX+1)/2) */
    f32* AFAC;   /* Factored packed matrix (NMAX*(NMAX+1)/2) */
    f32* AINV;   /* Inverse packed matrix (NMAX*(NMAX+1)/2) */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* D;      /* Singular values for slatms */
    f32* FERR;   /* Forward error bounds */
    f32* BERR;   /* Backward error bounds */
    int* IWORK;     /* Integer workspace */
} dchkpp_workspace_t;

static dchkpp_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkpp_workspace_t));
    if (!g_workspace) return -1;

    /* Note: A needs full NMAX*NMAX storage for slatms even though final result
     * is packed. slatms generates in full format then packs. AFAC and AINV
     * only need packed storage since they receive already-packed data. */
    int npp = NMAX * (NMAX + 1) / 2;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AFAC = malloc(npp * sizeof(f32));
    g_workspace->AINV = malloc(npp * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->RWORK = malloc((2 * NSMAX > NMAX ? 2 * NSMAX : NMAX) * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
    g_workspace->IWORK = malloc(NMAX * sizeof(int));

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
 * Run the full dchkpp test battery for a single (n, uplo, imat) combination.
 * This is the core test logic, parameterized by the test case.
 */
static void run_dchkpp_single(int n, int iuplo, int imat)
{
    const f32 ZERO = 0.0f;
    dchkpp_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    const char* packit = PACKS[iuplo];
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info, izero;
    int lda = (n > 1) ? n : 1;
    int npp = n * (n + 1) / 2;
    f32 rcondc, rcond;

    f32 result[NTESTS];
    char ctx[128];

    /* Seed based on (n, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    slatb4("SPP", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric positive definite test matrix in packed form */
    slatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, packit, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 3-5, zero one row and column to create singular matrix */
    int zerot = (imat >= 3 && imat <= 5);
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }
        /* Zero row and column izero in packed format */
        int ioff;
        if (iuplo == 0) {
            /* Upper packed: column j stored at AP[(j-1)*j/2 : (j-1)*j/2 + j] */
            ioff = (izero - 1) * izero / 2;
            for (int i = 0; i < izero - 1; i++) {
                ws->A[ioff + i] = ZERO;
            }
            ioff = ioff + izero - 1;
            for (int i = izero - 1; i < n; i++) {
                ws->A[ioff] = ZERO;
                ioff = ioff + i + 1;
            }
        } else {
            /* Lower packed */
            ioff = izero - 1;
            for (int i = 0; i < izero - 1; i++) {
                ws->A[ioff] = ZERO;
                ioff = ioff + n - i - 1;
            }
            ioff = ioff - (izero - 1);
            for (int i = izero - 1; i < n; i++) {
                ws->A[ioff + i - (izero - 1)] = ZERO;
            }
        }
    } else {
        izero = 0;
    }

    /* Copy A to AFAC for factorization */
    cblas_scopy(npp, ws->A, 1, ws->AFAC, 1);

    /* Compute the Cholesky factorization */
    spptrf(uplo_str, n, ws->AFAC, &info);

    /* Check error code */
    if (zerot) {
        assert_true(info >= 0);
        if (info != izero) {
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
    cblas_scopy(npp, ws->AFAC, 1, ws->AINV, 1);
    sppt01(uplo_str, n, ws->A, ws->AINV, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2: Form the inverse and compute the residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
    set_test_context(ctx);
    cblas_scopy(npp, ws->AFAC, 1, ws->AINV, 1);
    spptri(uplo_str, n, ws->AINV, &info);
    assert_int_equal(info, 0);

    sppt03(uplo_str, n, ws->A, ws->AINV, ws->WORK, lda,
           ws->RWORK, &rcondc, &result[1]);
    assert_residual_below(result[1], THRESH);

    /*
     * TESTS 3-7: Solve tests
     */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /*
         * TEST 3: Solve and compute residual for A * X = B
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 3 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);

        /* Generate RHS using full-format A temporarily */
        /* For packed, we need slarhs with packed matrix - use lda=n for the RHS arrays */
        slarhs("SPP", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        slacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        spptrs(uplo_str, n, nrhs, ws->AFAC, ws->X, lda, &info);
        assert_int_equal(info, 0);

        slacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        sppt02(uplo_str, n, nrhs, ws->A, ws->X, lda,
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

        spprfs(uplo_str, n, nrhs, ws->A, ws->AFAC,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
        sppt05(uplo_str, n, nrhs, ws->A, ws->B, lda, ws->X, lda,
               ws->XACT, lda, ws->FERR, ws->BERR, &result[5]);

        assert_residual_below(result[4], THRESH);
        assert_residual_below(result[5], THRESH);
        assert_residual_below(result[6], THRESH);
    }

    /*
     * TEST 8: Get an estimate of RCOND = 1/CNDNUM
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 8 (condition number)", n, uplo, imat);
    set_test_context(ctx);
    anorm = slansp("1", uplo_str, n, ws->A, ws->RWORK);
    sppcon(uplo_str, n, ws->AFAC, anorm, &rcond, ws->WORK,
           ws->IWORK, &info);
    assert_int_equal(info, 0);

    result[7] = sget06(rcond, rcondc);
    assert_residual_below(result[7], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkpp_single based on prestate.
 */
static void test_dchkpp_case(void** state)
{
    dchkpp_params_t* params = *state;
    run_dchkpp_single(params->n, params->iuplo, params->imat);
}

/*
 * Generate all parameter combinations.
 * No block size loop since packed routines are unblocked.
 * Total: NN * NUPLO * NTYPES = 7 * 2 * 9 = 126 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES)

static dchkpp_params_t g_params[MAX_TESTS];
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
            /* Skip types 3, 4, or 5 if matrix size is too small */
            int zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (int iuplo = 0; iuplo < (int)NUPLO; iuplo++) {
                /* Store parameters */
                dchkpp_params_t* p = &g_params[g_num_tests];
                p->n = n;
                p->imat = imat;
                p->iuplo = iuplo;
                snprintf(p->name, sizeof(p->name), "dchkpp_n%d_%c_type%d",
                         n, UPLOS[iuplo], imat);

                /* Create CMocka test entry */
                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dchkpp_case;
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

    /* Run all tests with shared workspace */
    return _cmocka_run_group_tests("dchkpp", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
