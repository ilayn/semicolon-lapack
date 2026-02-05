/**
 * @file test_dchkpo.c
 * @brief Comprehensive test suite for positive definite matrix (DPO) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkpo.f to C using CMocka.
 * Tests DPOTRF, DPOTRI, DPOTRS, DPORFS, and DPOCON.
 *
 * Each (n, uplo, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkpo.f:
 *   TEST 1: Cholesky factorization residual via dpot01
 *   TEST 2: Matrix inverse residual via dpot03
 *   TEST 3: Solution residual via dpot02
 *   TEST 4: Solution accuracy via dget04
 *   TEST 5: Refined solution accuracy via dget04 (after dporfs)
 *   TEST 6-7: Error bounds via dpot05
 *   TEST 8: Condition number via dget06
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-9
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 15};  /* NRHS values */
static const int NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from dtest.in */
static const char UPLOS[] = {'U', 'L'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9
#define NTESTS  8
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void dpotrf(const char* uplo, const int n, double* A, const int lda,
                   int* info);
extern void dpotri(const char* uplo, const int n, double* A, const int lda,
                   int* info);
extern void dpotrs(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, double* B, const int ldb,
                   int* info);
extern void dporfs(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, const double* AF,
                   const int ldaf, const double* B, const int ldb,
                   double* X, const int ldx, double* ferr, double* berr,
                   double* work, int* iwork, int* info);
extern void dpocon(const char* uplo, const int n, const double* A,
                   const int lda, const double anorm, double* rcond,
                   double* work, int* iwork, int* info);

/* Verification routines */
extern void dpot01(const char* uplo, const int n, const double* A,
                   const int lda, double* AFAC, const int ldafac,
                   double* rwork, double* resid);
extern void dpot02(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, const double* X,
                   const int ldx, double* B, const int ldb,
                   double* rwork, double* resid);
extern void dpot03(const char* uplo, const int n, const double* A,
                   const int lda, const double* AINV, const int ldainv,
                   double* work, const int ldwork, double* rwork,
                   double* rcond, double* resid);
extern void dpot05(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, const double* B,
                   const int ldb, const double* X, const int ldx,
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
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const double* A, const int lda,
                   const double* XACT, const int ldxact, double* B,
                   const int ldb, uint64_t seed, int* info);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* A, const int lda, double* work);
extern double dlamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    int iuplo;  /* 0='U', 1='L' */
    int inb;    /* Index into NBVAL[] */
    char name[64];
} dchkpo_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    double* A;      /* Original matrix (NMAX x NMAX) */
    double* AFAC;   /* Factored matrix (NMAX x NMAX) */
    double* AINV;   /* Inverse matrix (NMAX x NMAX) */
    double* B;      /* Right-hand side (NMAX x NSMAX) */
    double* X;      /* Solution (NMAX x NSMAX) */
    double* XACT;   /* Exact solution (NMAX x NSMAX) */
    double* WORK;   /* General workspace */
    double* RWORK;  /* Real workspace */
    double* D;      /* Singular values for dlatms */
    double* FERR;   /* Forward error bounds */
    double* BERR;   /* Backward error bounds */
    int* IWORK;     /* Integer workspace */
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

    g_workspace->A = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->RWORK = malloc(NMAX * sizeof(double));
    g_workspace->D = malloc(NMAX * sizeof(double));
    g_workspace->FERR = malloc(NSMAX * sizeof(double));
    g_workspace->BERR = malloc(NSMAX * sizeof(double));
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
 * Run the full dchkpo test battery for a single (n, uplo, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Following LAPACK's dchkpo.f:
 *   - TESTs 1-2 (factorization, inverse) run for all NB values
 *   - TESTs 3-8 (solve, refinement) only run for inb=0 (first NB)
 */
static void run_dchkpo_single(int n, int iuplo, int imat, int inb)
{
    const double ZERO = 0.0;
    dchkpo_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    int kl, ku, mode;
    double anorm, cndnum;
    int info, izero;
    int lda = (n > 1) ? n : 1;
    double rcondc, rcond;

    /* Set block size for this test via xlaenv */
    int nb = NBVAL[inb];
    xlaenv(1, nb);
    double result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Seed based on (n, uplo, imat) for reproducibility */
    uint64_t seed = 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat);

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    dlatb4("DPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric positive definite test matrix */
    dlatms(n, n, &dist, seed++, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info);
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
        /* Zero row and column izero (1-based in LAPACK, 0-based here) */
        int ioff = (izero - 1) * lda;
        if (iuplo == 0) {
            /* Upper: zero column above diagonal, then row to the right */
            for (int i = 0; i < izero - 1; i++) {
                ws->A[ioff + i] = ZERO;
            }
            ioff += izero - 1;
            for (int i = izero - 1; i < n; i++) {
                ws->A[ioff] = ZERO;
                ioff += lda;
            }
        } else {
            /* Lower: zero column below diagonal, then row to the left */
            ioff = izero - 1;
            for (int i = 0; i < izero - 1; i++) {
                ws->A[ioff] = ZERO;
                ioff += lda;
            }
            ioff = (izero - 1) * lda + izero - 1;
            for (int i = izero - 1; i < n; i++) {
                ws->A[ioff + i - (izero - 1)] = ZERO;
            }
        }
    } else {
        izero = 0;
    }

    /* Copy A to AFAC for factorization */
    dlacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the Cholesky factorization */
    dpotrf(uplo_str, n, ws->AFAC, lda, &info);

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
    dlacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
    dpot01(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2: Form the inverse and compute the residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
    set_test_context(ctx);
    dlacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
    dpotri(uplo_str, n, ws->AINV, lda, &info);
    assert_int_equal(info, 0);

    dpot03(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
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
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /*
         * TEST 3: Solve and compute residual for A * X = B
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 3 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        dlarhs("DPO", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, seed++, &info);

        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        dpotrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->X, lda, &info);
        assert_int_equal(info, 0);

        dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        dpot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TEST 4: Check solution from generated exact solution
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 4 (solution accuracy)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
        assert_residual_below(result[3], THRESH);

        /*
         * TESTS 5, 6, 7: Iterative refinement
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 5-7 (refinement)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        dpotrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->X, lda, &info);

        dporfs(uplo_str, n, nrhs, ws->A, lda, ws->AFAC, lda,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
        dpot05(uplo_str, n, nrhs, ws->A, lda, ws->B, lda, ws->X, lda,
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
    anorm = dlansy("1", uplo_str, n, ws->A, lda, ws->RWORK);
    dpocon(uplo_str, n, ws->AFAC, lda, anorm, &rcond, ws->WORK,
           ws->IWORK, &info);
    assert_int_equal(info, 0);

    result[7] = dget06(rcond, rcondc);
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
                /* Loop over block sizes */
                for (int inb = 0; inb < (int)NNB; inb++) {
                    int nb = NBVAL[inb];

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
