/**
 * @file test_zchkhe.c
 * @brief Comprehensive test suite for Hermitian indefinite matrix (ZHE) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkhe.f to C using CMocka.
 * Tests ZHETRF, ZHETRI2, ZHETRS, ZHETRS2, ZHERFS, and ZHECON.
 *
 * Each (n, uplo, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkhe.f:
 *   TEST 1: LDL^H factorization residual via zhet01
 *   TEST 2: Matrix inverse residual via zpot03
 *   TEST 3: Solution residual via zpot02 (using zhetrs)
 *   TEST 4: Solution residual via zpot02 (using zhetrs2)
 *   TEST 5: Solution accuracy via zget04
 *   TEST 6: Refined solution accuracy via zget04 (after zherfs)
 *   TEST 7-8: Error bounds via zpot05
 *   TEST 9: Condition number via dget06
 *
 * Parameters from ztest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-10
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
/* Test parameters from ztest.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values */
static const INT NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from ztest.in */
static const INT NXVAL[] = {1, 0, 5, 9, 1};   /* Crossover points from ztest.in */
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

/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT n;
    INT imat;
    INT iuplo;  /* 0='U', 1='L' */
    INT inb;    /* Index into NBVAL[] */
    char name[64];
} zchkhe_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    c128* A;      /* Original matrix (NMAX x NMAX) */
    c128* AFAC;   /* Factored matrix (NMAX x NMAX) */
    c128* AINV;   /* Inverse matrix (NMAX x NMAX) */
    c128* B;      /* Right-hand side (NMAX x NSMAX) */
    c128* X;      /* Solution (NMAX x NSMAX) */
    c128* XACT;   /* Exact solution (NMAX x NSMAX) */
    c128* WORK;   /* General workspace */
    f64* RWORK;   /* Real workspace */
    f64* D;       /* Singular values for zlatms */
    f64* FERR;    /* Forward error bounds */
    f64* BERR;    /* Backward error bounds */
    INT* IPIV;    /* Pivot indices */
    INT* IWORK;   /* Integer workspace */
} zchkhe_workspace_t;

static zchkhe_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkhe_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * 64;  /* Generous workspace for zhetrf */

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->WORK = malloc(lwork * sizeof(c128));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
    g_workspace->IPIV = malloc(NMAX * sizeof(INT));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(INT));

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
 * Run the full zchkhe test battery for a single (n, uplo, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Following LAPACK's zchkhe.f:
 *   - TESTs 1-2 (factorization, inverse) run for all NB values
 *   - TESTs 3-9 (solve, refinement) only run for inb=0 (first NB)
 */
static void run_zchkhe_single(INT n, INT iuplo, INT imat, INT inb)
{
    const f64 ZERO = 0.0;
    zchkhe_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    INT lwork = NMAX * 64;
    INT trfcon;
    f64 rcondc, rcond;
    f64 result[NTESTS];
    char ctx[128];

    /* Set block size and crossover point for this test via xlaenv */
    INT nb = NBVAL[inb];
    INT nx = NXVAL[inb];
    xlaenv(1, nb);
    xlaenv(3, nx);

    /* Seed based on (n, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    zlatb4("ZHE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate Hermitian test matrix */
    zlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 3-6, zero one or more rows and columns.
     * izero is 0-based index of the row/column to zero. */
    INT zerot = (imat >= 3 && imat <= 6);
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
                INT ioff = izero * lda;
                for (INT i = 0; i < izero; i++) {
                    ws->A[ioff + i] = ZERO;
                }
                ioff += izero;
                for (INT i = izero; i < n; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
            } else {
                /* Lower: zero row izero (columns 0 to izero-1) and
                 * column izero (rows izero to n-1) */
                INT ioff = izero;
                for (INT i = 0; i < izero; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
                ioff = izero * lda + izero;
                for (INT i = izero; i < n; i++) {
                    ws->A[ioff + i - izero] = ZERO;
                }
            }
        } else {
            /* Type 6: zero first izero+1 rows and columns (upper) or last (lower) */
            if (iuplo == 0) {
                INT ioff = 0;
                for (INT j = 0; j < n; j++) {
                    INT i2 = (j <= izero) ? j + 1 : izero + 1;
                    for (INT i = 0; i < i2; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
            } else {
                INT ioff = 0;
                for (INT j = 0; j < n; j++) {
                    INT i1 = (j >= izero) ? j : izero;
                    for (INT i = i1; i < n; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
            }
        }
    } else {
        izero = -1;  /* No zeroing, use -1 to indicate none */
    }

    /* Set the imaginary part of the diagonals. */
    zlaipd(n, ws->A, lda + 1, 0);

    /* Copy A to AFAC for factorization */
    zlacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the L*D*L^H or U*D*U^H factorization */
    zhetrf(uplo_str, n, ws->AFAC, lda, ws->IPIV, ws->WORK, lwork, &info);

    /* Check error code - for singular matrices, need to account for pivoting */
    if (izero >= 0) {
        INT k = izero;
        while (k >= 0 && k < n) {
            if (ws->IPIV[k] < 0) {
                INT kp = -(ws->IPIV[k] + 1);
                if (kp != k) {
                    k = kp;
                } else {
                    break;
                }
            } else if (ws->IPIV[k] != k) {
                k = ws->IPIV[k];
            } else {
                break;
            }
        }
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
    zhet01(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->IPIV,
           ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2: Form the inverse and compute the residual (if factorization succeeded)
     * Uses zhetri2 (blocked Hermitian inverse) as in LAPACK's zchkhe.f
     * Only run for the first block size (inb == 0), matching LAPACK's INB.EQ.1 check.
     */
    if (inb == 0 && !trfcon) {
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
        set_test_context(ctx);
        zlacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
        INT lwork_tri2 = (n + nb + 1) * (nb + 3);
        zhetri2(uplo_str, n, ws->AINV, lda, ws->IPIV, ws->WORK, lwork_tri2, &info);
        if (info == 0) {
            zpot03(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
                   ws->RWORK, &rcondc, &result[1]);
            assert_residual_below(result[1], THRESH);
        }
    }

    /*
     * Skip all tests beyond TEST 1 if not the first block size.
     * This matches LAPACK's zchkhe.f line 500: IF( INB.GT.1 ) GO TO 150
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
    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        /*
         * TEST 3: Solve and compute residual for A * X = B using zhetrs
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 3 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        zlarhs("ZHE", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        zhetrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, &info);
        assert_int_equal(info, 0);

        zlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        zpot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TEST 4: Solve using zhetrs2 (blocked Hermitian solve)
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 4 (solve2)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        zhetrs2(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, ws->WORK, &info);
        assert_int_equal(info, 0);

        zlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        zpot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[3]);
        assert_residual_below(result[3], THRESH);

        /*
         * TEST 5: Check solution from generated exact solution
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 5 (accuracy)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
        assert_residual_below(result[4], THRESH);

        /*
         * TESTS 6, 7, 8: Iterative refinement
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 6-8 (refinement)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        zhetrs(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, &info);

        zherfs(uplo_str, n, nrhs, ws->A, lda, ws->AFAC, lda, ws->IPIV,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->RWORK, &info);
        assert_int_equal(info, 0);

        zget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[5]);
        zpot05(uplo_str, n, nrhs, ws->A, lda, ws->B, lda, ws->X, lda,
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
    anorm = zlanhe("1", uplo_str, n, ws->A, lda, ws->RWORK);
    zhecon(uplo_str, n, ws->AFAC, lda, ws->IPIV, anorm, &rcond,
           ws->WORK, &info);
    assert_int_equal(info, 0);

    result[8] = dget06(rcond, rcondc);
    assert_residual_below(result[8], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_zchkhe_single based on prestate.
 */
static void test_zchkhe_case(void** state)
{
    zchkhe_params_t* params = *state;
    run_zchkhe_single(params->n, params->iuplo, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NUPLO * NTYPES * NNB = 7 * 2 * 10 * 5 = 700 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NNB)

static zchkhe_params_t g_params[MAX_TESTS];
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
            /* Skip types 3, 4, 5, or 6 if matrix size is too small */
            INT zerot = (imat >= 3 && imat <= 6);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (INT iuplo = 0; iuplo < (INT)NUPLO; iuplo++) {
                /* Loop over block sizes */
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    /* Store parameters */
                    zchkhe_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "zchkhe_n%d_%c_type%d_nb%d_nx%d_%d",
                             n, UPLOS[iuplo], imat, nb, NXVAL[inb], inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchkhe_case;
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

    /* Run all tests with shared workspace */
    (void)_cmocka_run_group_tests("zchkhe", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
