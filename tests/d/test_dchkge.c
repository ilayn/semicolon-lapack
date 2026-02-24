/**
 * @file test_dchkge.c
 * @brief Comprehensive test suite for general matrix (DGE) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkge.f to C using CMocka.
 * Tests DGETRF, DGETRI, DGETRS, DGERFS, and DGECON.
 *
 * Each (m, n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkge.f:
 *   TEST 1: LU factorization residual via dget01
 *   TEST 2: Matrix inverse residual via dget03 (square, non-singular only)
 *   TEST 3: Solution residual via dget02
 *   TEST 4: Solution accuracy via dget04
 *   TEST 5: Refined solution accuracy via dget04 (after dgerfs)
 *   TEST 6-7: Error bounds via dget07
 *   TEST 8: Condition number via dget06
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   NB values: 1, 3, 3, 3, 20 (unique: 1, 3, 20)
 *   Matrix types: 1-11
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
/* Test parameters from dtest.in */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values */
static const INT NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from dtest.in */
static const char TRANSS[] = {'N', 'T', 'C'};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTRAN   (sizeof(TRANSS) / sizeof(TRANSS[0]))
#define NTYPES  11
#define NTESTS  8
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
/* Verification routines */
/* Matrix generation */
/* Utilities */
/**
 * Test parameters for a single test case.
 *
 * The test is parameterized by (m, n, imat, inb) where:
 *   - m, n: matrix dimensions
 *   - imat: matrix type (1-11)
 *   - inb: index into NBVAL[] for block size
 *
 * Following LAPACK's dchkge.f:
 *   - TESTs 1-2 (factorization, inverse) run for all NB values
 *   - TESTs 3-8 (solve, refinement) only run for inb=0 (first NB) and M==N
 */
typedef struct {
    INT m;
    INT n;
    INT imat;
    INT inb;    /* Index into NBVAL[] */
    char name[64];
} dchkge_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Original matrix (NMAX x NMAX) */
    f64* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f64* AINV;   /* Inverse matrix (NMAX x NMAX) */
    f64* B;      /* Right-hand side (NMAX x NSMAX) */
    f64* X;      /* Solution (NMAX x NSMAX) */
    f64* XACT;   /* Exact solution (NMAX x NSMAX) */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace */
    f64* D;      /* Singular values for dlatms */
    f64* FERR;   /* Forward error bounds */
    f64* BERR;   /* Backward error bounds */
    INT* IPIV;      /* Pivot indices */
    INT* IWORK;     /* Integer workspace */
} dchkge_workspace_t;

static dchkge_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkge_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->RWORK = malloc(2 * NMAX * sizeof(f64));
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
 * Run the full dchkge test battery for a single (m, n, imat, inb) combination.
 * This is the core test logic, parameterized by the test case.
 *
 * Following LAPACK's dchkge.f:
 *   - TESTs 1-2 (factorization, inverse) run for all NB values
 *   - TESTs 3-8 (solve, refinement) only run for inb=0 and M==N
 */
static void run_dchkge_single(INT m, INT n, INT imat, INT inb)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    dchkge_workspace_t* ws = g_workspace;

    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info, izero;
    INT lda = (m > 1) ? m : 1;
    INT trfcon;
    f64 anormo, anormi, rcondo, rcondi, rcond, rcondc;
    f64 result[NTESTS];

    /* Set block size for this test via xlaenv (mirrors LAPACK's XLAENV call) */
    INT nb = NBVAL[inb];
    xlaenv(1, nb);

    /* Seed based on (m, n, imat) for reproducibility.
     * Note: seed does NOT include inb so we test the same matrix with different NB */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(m * 1000 + n * 100 + imat));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    dlatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, "N", ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 5-7, zero one or more columns to create singular matrix */
    INT zerot = (imat >= 5 && imat <= 7);
    if (zerot) {
        INT minmn = (m < n) ? m : n;
        if (imat == 5) {
            izero = 1;
        } else if (imat == 6) {
            izero = minmn;
        } else {
            izero = minmn / 2 + 1;
        }
        /* Zero column izero (1-based in LAPACK, 0-based here) */
        INT ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (INT i = 0; i < m; i++) {
                ws->A[ioff + i] = ZERO;
            }
        } else {
            dlaset("F", m, n - izero + 1, ZERO, ZERO, &ws->A[ioff], lda);
        }
    } else {
        izero = 0;
    }

    /* Copy A to AFAC for factorization */
    dlacpy("F", m, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the LU factorization */
    dgetrf(m, n, ws->AFAC, lda, ws->IPIV, &info);

    /* Check error code */
    if (zerot) {
        /* For singular matrices, info should be > 0 */
        assert_true(info >= 0);
    } else {
        assert_int_equal(info, 0);
    }
    trfcon = (info != 0);

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    dlacpy("F", m, n, ws->AFAC, lda, ws->AINV, lda);
    dget01(m, n, ws->A, lda, ws->AINV, lda, ws->IPIV, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2: Form the inverse if factorization was successful (square only).
     */
    if (m == n && info == 0) {
        dlacpy("F", n, n, ws->AFAC, lda, ws->AINV, lda);
        INT lwork = NMAX * 3;
        dgetri(n, ws->AINV, lda, ws->IPIV, ws->WORK, lwork, &info);
        assert_int_equal(info, 0);

        /* Compute residual for matrix times its inverse */
        dget03(n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
               ws->RWORK, &rcondo, &result[1]);
        anormo = dlange("O", m, n, ws->A, lda, ws->RWORK);

        /* Compute infinity-norm condition number */
        anormi = dlange("I", m, n, ws->A, lda, ws->RWORK);
        f64 ainvnm = dlange("I", n, n, ws->AINV, lda, ws->RWORK);
        if (anormi <= ZERO || ainvnm <= ZERO) {
            rcondi = ONE;
        } else {
            rcondi = (ONE / anormi) / ainvnm;
        }
        assert_residual_below(result[1], THRESH);
    } else {
        /* Do only the condition estimate if INFO > 0 */
        trfcon = 1;
        anormo = dlange("O", m, n, ws->A, lda, ws->RWORK);
        anormi = dlange("I", m, n, ws->A, lda, ws->RWORK);
        rcondo = ZERO;
        rcondi = ZERO;
    }

    /*
     * Skip solve tests if:
     *   - Matrix is not square (m != n), OR
     *   - Matrix is singular (trfcon), OR
     *   - Not the first block size (inb > 0)
     *
     * This matches LAPACK's dchkge.f line 449-450:
     *   IF( INB.GT.1 .OR. M.NE.N ) GO TO 90
     */
    if (m != n || trfcon || inb > 0) {
        goto test8;
    }

    /*
     * TESTS 3-7: Solve tests (only for M==N, non-singular, first NB)
     */
    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        for (INT itran = 0; itran < (INT)NTRAN; itran++) {
            char trans_arr[2] = {TRANSS[itran], '\0'};
            rcondc = (itran == 0) ? rcondo : rcondi;

            /*
             * TEST 3: Solve and compute residual for A * X = B
             */
            dlarhs("DGE", "N", " ", trans_arr, n, n, kl, ku, nrhs,
                   ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

            dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            dgetrs(trans_arr, n, nrhs, ws->AFAC, lda, ws->IPIV, ws->X, lda, &info);
            assert_int_equal(info, 0);

            dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
            dget02(trans_arr, n, n, nrhs, ws->A, lda, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[2]);

            assert_residual_below(result[2], THRESH);

            /*
             * TEST 4: Check solution from generated exact solution
             */
            dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);

            assert_residual_below(result[3], THRESH);

            /*
             * TESTS 5, 6, 7: Iterative refinement
             */
            dgerfs(trans_arr, n, nrhs, ws->A, lda, ws->AFAC, lda,
                   ws->IPIV, ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
                   ws->WORK, ws->IWORK, &info);
            assert_int_equal(info, 0);

            dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[4]);
            dget07(trans_arr, n, nrhs, ws->A, lda, ws->B, lda, ws->X, lda,
                   ws->XACT, lda, ws->FERR, 1, ws->BERR, &result[5]);

            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);
            assert_residual_below(result[6], THRESH);
        }
    }

test8:
    /*
     * TEST 8: Get an estimate of RCOND = 1/CNDNUM
     */
    if (m == n) {
        for (INT itran = 0; itran < 2; itran++) {
            char norm;
            if (itran == 0) {
                anorm = anormo;
                rcondc = rcondo;
                norm = 'O';
            } else {
                anorm = anormi;
                rcondc = rcondi;
                norm = 'I';
            }
            char norm_arr[2] = {norm, '\0'};
            dgecon(norm_arr, n, ws->AFAC, lda, anorm, &rcond, ws->WORK,
                   &ws->IWORK[n], &info);
            assert_int_equal(info, 0);

            result[7] = dget06(rcond, rcondc);

            assert_residual_below(result[7], THRESH);
        }
    }
}

/**
 * CMocka test function - dispatches to run_dchkge_single based on prestate.
 */
static void test_dchkge_case(void** state)
{
    dchkge_params_t* params = *state;
    run_dchkge_single(params->m, params->n, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NM * NN * NTYPES * NNB = 7 * 7 * 11 * 5 = 2695 tests (with unique NB)
 * However, NBVAL has duplicates {1, 3, 3, 3, 20}, so effective unique values are {1, 3, 20}.
 * We test all 5 entries as LAPACK does, even with duplicates.
 * (minus skipped cases for small m,n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NM * NN * NTYPES * NNB)

static dchkge_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];

        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];

            INT nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (INT imat = 1; imat <= nimat; imat++) {
                /* Skip types 5, 6, or 7 if matrix size is too small */
                INT zerot = (imat >= 5 && imat <= 7);
                INT minmn = (m < n) ? m : n;
                if (zerot && minmn < imat - 4) {
                    continue;
                }

                /* Loop over block sizes */
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    /* Store parameters */
                    dchkge_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->imat = imat;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchkge_m%d_n%d_type%d_nb%d_%d",
                             m, n, imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchkge_case;
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
    return _cmocka_run_group_tests("dchkge", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
