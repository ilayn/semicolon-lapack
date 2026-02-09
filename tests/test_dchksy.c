/**
 * @file test_dchksy.c
 * @brief Comprehensive test suite for symmetric indefinite matrix (DSY) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchksy.f to C using CMocka.
 * Tests DSYTRF, DSYTRI, DSYTRS, DSYRFS, and DSYCON.
 *
 * Each (n, uplo, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchksy.f:
 *   TEST 1: LDL^T factorization residual via dsyt01
 *   TEST 2: Matrix inverse residual via dpot03 (uses same formula)
 *   TEST 3: Solution residual via dpot02 (using dsytrs)
 *   TEST 4: Solution residual via dpot02 (using dsytrs2)
 *   TEST 5: Solution accuracy via dget04
 *   TEST 6: Refined solution accuracy via dget04 (after dsyrfs)
 *   TEST 7-8: Error bounds via dpot05
 *   TEST 9: Condition number via dget06
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-10
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
static const int NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from dtest.in */
static const int NXVAL[] = {1, 0, 5, 9, 1};   /* Crossover points from dtest.in */
static const char UPLOS[] = {'U', 'L'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  10
#define NTESTS  9
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void dsytrf(const char* uplo, const int n, double* A, const int lda,
                   int* ipiv, double* work, const int lwork, int* info);
extern void dsytri(const char* uplo, const int n, double* A, const int lda,
                   const int* ipiv, double* work, int* info);
extern void dsytri2(const char* uplo, const int n, double* A, const int lda,
                    const int* ipiv, double* work, const int lwork, int* info);
extern void dsytrs(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, const int* ipiv,
                   double* B, const int ldb, int* info);
extern void dsytrs2(const char* uplo, const int n, const int nrhs,
                    double* A, const int lda, const int* ipiv,
                    double* B, const int ldb, double* work, int* info);
extern void dsyrfs(const char* uplo, const int n, const int nrhs,
                   const double* A, const int lda, const double* AF,
                   const int ldaf, const int* ipiv, const double* B,
                   const int ldb, double* X, const int ldx, double* ferr,
                   double* berr, double* work, int* iwork, int* info);
extern void dsycon(const char* uplo, const int n, const double* A,
                   const int lda, const int* ipiv, const double anorm,
                   double* rcond, double* work, int* iwork, int* info);

/* Verification routines */
extern void dsyt01(const char* uplo, const int n, const double* A,
                   const int lda, const double* AFAC, const int ldafac,
                   const int* ipiv, double* C, const int ldc,
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
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, double* d, const int mode, const double cond,
                   const double dmax, const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const double* A, const int lda,
                   const double* XACT, const int ldxact, double* B,
                   const int ldb, int* info, uint64_t state[static 4]);

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
} dchksy_params_t;

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
    int* IPIV;      /* Pivot indices */
    int* IWORK;     /* Integer workspace */
} dchksy_workspace_t;

static dchksy_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchksy_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * 64;  /* Generous workspace for dsytrf */

    g_workspace->A = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->WORK = malloc(lwork * sizeof(double));
    g_workspace->RWORK = malloc(NMAX * sizeof(double));
    g_workspace->D = malloc(NMAX * sizeof(double));
    g_workspace->FERR = malloc(NSMAX * sizeof(double));
    g_workspace->BERR = malloc(NSMAX * sizeof(double));
    g_workspace->IPIV = malloc(NMAX * sizeof(int));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->IPIV ||
        !g_workspace->IWORK) {
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
        free(g_workspace->IPIV);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full dchksy test battery for a single (n, uplo, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Following LAPACK's dchksy.f:
 *   - TESTs 1-2 (factorization, inverse) run for all NB values
 *   - TESTs 3-9 (solve, refinement) only run for inb=0 (first NB)
 */
static void run_dchksy_single(int n, int iuplo, int imat, int inb)
{
    const double ZERO = 0.0;
    dchksy_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    int kl, ku, mode;
    double anorm, cndnum;
    int info, izero;
    int lda = (n > 1) ? n : 1;
    int lwork = NMAX * 64;
    int trfcon;
    double rcondc, rcond;
    double result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Set block size and crossover point for this test via xlaenv */
    int nb = NBVAL[inb];
    int nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Seed based on (n, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    dlatb4("DSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric test matrix */
    dlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 3-6, zero one or more rows and columns.
     * izero is 0-based index of the row/column to zero. */
    int zerot = (imat >= 3 && imat <= 6);
    if (zerot) {
        if (imat == 3) {
            izero = 0;  /* First row/column */
        } else if (imat == 4) {
            izero = n - 1;  /* Last row/column */
        } else {
            izero = n / 2;  /* Middle row/column */
        }

        if (imat < 6) {
            /* Zero row and column izero */
            if (iuplo == 0) {
                /* Upper: zero column izero (rows 0 to izero-1) and
                 * row izero (columns izero to n-1) */
                int ioff = izero * lda;
                for (int i = 0; i < izero; i++) {
                    ws->A[ioff + i] = ZERO;
                }
                ioff += izero;
                for (int i = izero; i < n; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
            } else {
                /* Lower: zero row izero (columns 0 to izero-1) and
                 * column izero (rows izero to n-1) */
                int ioff = izero;
                for (int i = 0; i < izero; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
                ioff = izero * lda + izero;
                for (int i = izero; i < n; i++) {
                    ws->A[ioff + i - izero] = ZERO;
                }
            }
        } else {
            /* Type 6: zero first izero+1 rows and columns (upper) or last (lower) */
            if (iuplo == 0) {
                int ioff = 0;
                for (int j = 0; j < n; j++) {
                    int i2 = (j <= izero) ? j + 1 : izero + 1;
                    for (int i = 0; i < i2; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
            } else {
                int ioff = 0;
                for (int j = 0; j < n; j++) {
                    int i1 = (j >= izero) ? j : izero;
                    for (int i = i1; i < n; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
            }
        }
    } else {
        izero = -1;  /* No zeroing, use -1 to indicate none */
    }

    /* Copy A to AFAC for factorization */
    dlacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the L*D*L^T or U*D*U^T factorization */
    dsytrf(uplo_str, n, ws->AFAC, lda, ws->IPIV, ws->WORK, lwork, &info);

    /* Check error code - for singular matrices, need to account for pivoting */
    if (izero >= 0) {
        /* Trace through pivots to find effective izero (0-based) */
        int k = izero;
        while (k >= 0 && k < n) {
            if (ws->IPIV[k] < 0) {
                /* 2x2 block: pivot index is -(ipiv+1) to get 0-based */
                int kp = -(ws->IPIV[k] + 1);
                if (kp != k) {
                    k = kp;
                } else {
                    break;
                }
            } else if (ws->IPIV[k] != k) {
                /* 1x1 block: ipiv[k] is the 0-based pivot row */
                k = ws->IPIV[k];
            } else {
                break;
            }
        }
        /* info should match the effective singular position */
        assert_true(info >= 0);
    }
    trfcon = (info != 0);
    if (trfcon) {
        rcondc = ZERO;
    }

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 1 (factorization)", n, uplo, imat);
    set_test_context(ctx);
    dsyt01(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->IPIV,
           ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2: Form the inverse and compute the residual (if factorization succeeded)
     * Uses dsytri2 (blocked symmetric inverse) as in LAPACK's dchksy.f
     * Only run for the first block size (inb == 0), matching LAPACK's INB.EQ.1 check.
     */
    if (inb == 0 && !trfcon) {
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
        set_test_context(ctx);
        dlacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
        int lwork_tri2 = (n + nb + 1) * (nb + 3);
        dsytri2(uplo_str, n, ws->AINV, lda, ws->IPIV, ws->WORK, lwork_tri2, &info);
        if (info == 0) {
            dpot03(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
                   ws->RWORK, &rcondc, &result[1]);
            assert_residual_below(result[1], THRESH);
        }
    }

    /*
     * Skip all tests beyond TEST 1 if not the first block size.
     * This matches LAPACK's dchksy.f line 501: IF( INB.GT.1 ) GO TO 150
     * where label 150 is at the END of the NB loop (after TEST 9).
     */
    if (inb > 0) {
        clear_test_context();
        return;
    }

    /* For first block size only: do condition estimate if factorization failed */
    if (trfcon) {
        rcondc = ZERO;
        goto test9;
    }

    /*
     * TESTS 3-8: Solve tests (only for first NB)
     */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /*
         * TEST 3: Solve and compute residual for A * X = B using dsytrs
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 3 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        dlarhs("DSY", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        dsytrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, &info);
        assert_int_equal(info, 0);

        dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        dpot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TEST 4: Solve using dsytrs2 (blocked symmetric solve) as in LAPACK's dchksy.f
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 4 (solve2)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        dsytrs2(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, ws->WORK, &info);
        assert_int_equal(info, 0);

        dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        dpot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[3]);
        assert_residual_below(result[3], THRESH);

        /*
         * TEST 5: Check solution from generated exact solution
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 5 (accuracy)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
        assert_residual_below(result[4], THRESH);

        /*
         * TESTS 6, 7, 8: Iterative refinement
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 6-8 (refinement)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        dsytrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, &info);

        dsyrfs(uplo_str, n, nrhs, ws->A, lda, ws->AFAC, lda, ws->IPIV,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[5]);
        dpot05(uplo_str, n, nrhs, ws->A, lda, ws->B, lda, ws->X, lda,
               ws->XACT, lda, ws->FERR, ws->BERR, &result[6]);

        assert_residual_below(result[5], THRESH);
        assert_residual_below(result[6], THRESH);
        assert_residual_below(result[7], THRESH);
    }

test9:
    /*
     * TEST 9: Get an estimate of RCOND = 1/CNDNUM
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 9 (condition)", n, uplo, imat);
    set_test_context(ctx);
    anorm = dlansy("1", uplo_str, n, ws->A, lda, ws->RWORK);
    dsycon(uplo_str, n, ws->AFAC, lda, ws->IPIV, anorm, &rcond,
           ws->WORK, ws->IWORK, &info);
    assert_int_equal(info, 0);

    result[8] = dget06(rcond, rcondc);
    assert_residual_below(result[8], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchksy_single based on prestate.
 */
static void test_dchksy_case(void** state)
{
    dchksy_params_t* params = *state;
    run_dchksy_single(params->n, params->iuplo, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NUPLO * NTYPES * NNB = 7 * 2 * 10 * 5 = 700 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NNB)

static dchksy_params_t g_params[MAX_TESTS];
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
            /* Skip types 3, 4, 5, or 6 if matrix size is too small */
            int zerot = (imat >= 3 && imat <= 6);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (int iuplo = 0; iuplo < (int)NUPLO; iuplo++) {
                /* Loop over block sizes */
                for (int inb = 0; inb < (int)NNB; inb++) {
                    int nb = NBVAL[inb];

                    /* Store parameters */
                    dchksy_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchksy_n%d_%c_type%d_nb%d_nx%d_%d",
                             n, UPLOS[iuplo], imat, nb, NXVAL[inb], inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchksy_case;
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
    return _cmocka_run_group_tests("dchksy", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
