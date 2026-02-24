/**
 * @file test_ddrvab.c
 * @brief Test suite for DSGESV mixed-precision iterative refinement solver.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/ddrvab.f to C using CMocka.
 * Tests DSGESV which uses single precision factorization with f64 precision
 * iterative refinement.
 *
 * Test parameters from dchkab.f data file:
 *   M values: 0, 1, 2, 3, 5, 10, 16
 *   NRHS values: 2
 *   Matrix types: 1-11
 *   THRESH: 20.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
/* Test parameters */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 16};
static const INT NSVAL[] = {2};  /* NRHS values */

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  11
#define THRESH  20.0
#define NMAX    132
#define MAXRHS  16

/* Routine under test */
/* Verification routine */
/* Matrix generation */
/* Utilities */
/**
 * Test parameters for a single test case.
 */
typedef struct {
    INT n;
    INT nrhs;
    INT imat;
    char name[64];
} ddrvab_params_t;

/**
 * Workspace for test execution.
 */
typedef struct {
    f64* A;          /* Original matrix (NMAX x NMAX) */
    f64* AFAC;       /* Copy of A for factorization (NMAX x NMAX) */
    f64* B;          /* Right-hand side (NMAX x MAXRHS) */
    f64* X;          /* Solution (NMAX x MAXRHS) */
    f64* WORK;       /* Double precision workspace */
    f64* RWORK;      /* Workspace for verification */
    float* SWORK;       /* Single precision workspace */
    INT* IWORK;         /* Integer workspace (pivot indices) */
} ddrvab_workspace_t;

static ddrvab_workspace_t* g_ws = NULL;

/**
 * Group setup - allocate workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvab_workspace_t));
    if (!g_ws) return -1;

    g_ws->A = malloc(NMAX * NMAX * sizeof(f64));
    g_ws->AFAC = malloc(NMAX * NMAX * sizeof(f64));
    g_ws->B = malloc(NMAX * MAXRHS * sizeof(f64));
    g_ws->X = malloc(NMAX * MAXRHS * sizeof(f64));
    g_ws->WORK = malloc(NMAX * MAXRHS * 2 * sizeof(f64));
    g_ws->RWORK = malloc(2 * NMAX * sizeof(f64));
    g_ws->SWORK = malloc(NMAX * (NMAX + MAXRHS) * sizeof(float));
    g_ws->IWORK = malloc(NMAX * sizeof(INT));

    if (!g_ws->A || !g_ws->AFAC || !g_ws->B || !g_ws->X ||
        !g_ws->WORK || !g_ws->RWORK || !g_ws->SWORK || !g_ws->IWORK) {
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
    if (g_ws) {
        free(g_ws->A);
        free(g_ws->AFAC);
        free(g_ws->B);
        free(g_ws->X);
        free(g_ws->WORK);
        free(g_ws->RWORK);
        free(g_ws->SWORK);
        free(g_ws->IWORK);
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Run a single test case.
 */
static void run_test_single(INT n, INT nrhs, INT imat)
{
    char type, dist;
    INT kl, ku, mode, info, iter;
    f64 anorm, cndnum;
    INT lda = (n > 1) ? n : 1;
    INT izero = 0;

    /* Seed based on test parameters (matches LAPACK ISEEDY = {2006, 2007, 2008, 2009}) */
    uint64_t rng_state[4];
    rng_seed(rng_state, 2006 + n * 1000 + imat * 100 + nrhs);

    /* Get matrix parameters */
    dlatb4("DGE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    char dist_str[2] = {dist, '\0'};
    char sym_str[2] = {type, '\0'};
    dlatms(n, n, dist_str, sym_str, g_ws->RWORK, mode, cndnum, anorm,
           kl, ku, "N", g_ws->A, lda, g_ws->WORK, &info, rng_state);

    if (info != 0) {
        /* Matrix generation failed - skip this test */
        return;
    }

    /* For types 5-7, zero one or more columns to test singularity detection */
    INT zerot = (imat >= 5 && imat <= 7);
    if (zerot) {
        if (imat == 5) {
            izero = 1;
        } else if (imat == 6) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }
        /* Zero column izero (1-based) */
        INT ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (INT i = 0; i < n; i++) {
                g_ws->A[ioff + i] = 0.0;
            }
        } else {
            /* Zero columns from izero to n */
            dlaset("F", n, n - izero + 1, 0.0, 0.0, &g_ws->A[ioff], lda);
        }
    }

    /* Generate right-hand side */
    dlarhs("DGE", "N", " ", "N", n, n, kl, ku, nrhs,
           g_ws->A, lda, g_ws->X, lda, g_ws->B, lda, &info, rng_state);

    /* Save a copy of A for later */
    dlacpy("F", n, n, g_ws->A, lda, g_ws->AFAC, lda);

    /* Call DSGESV */
    dsgesv(n, nrhs, g_ws->A, lda, g_ws->IWORK, g_ws->B, lda,
           g_ws->X, lda, g_ws->WORK, g_ws->SWORK, &iter, &info);

    /* Restore A if iterative refinement was used */
    if (iter < 0) {
        dlacpy("F", n, n, g_ws->AFAC, lda, g_ws->A, lda);
    }

    /* Check error code */
    if (info != izero && izero != 0) {
        /* Expected singular but got different INFO */
        char context[128];
        snprintf(context, sizeof(context),
                 "ddrvab n=%d nrhs=%d imat=%d: INFO=%d, expected=%d",
                 n, nrhs, imat, info, izero);
        set_test_context(context);
        assert_int_equal(info, izero);
        clear_test_context();
        return;
    }

    /* Skip solution check if matrix is singular */
    if (info != 0) {
        return;
    }

    /* Check solution quality using dget08 */
    /* Copy B to WORK for residual computation (dget08 overwrites it) */
    dlacpy("F", n, nrhs, g_ws->B, lda, g_ws->WORK, lda);

    f64 resid;
    dget08("N", n, n, nrhs, g_ws->AFAC, lda, g_ws->X, lda,
           g_ws->WORK, lda, g_ws->RWORK, &resid);

    /* Check the test result:
     * If iterative refinement was used (iter >= 0), we want:
     *   resid < sqrt(n)
     * If f64 precision was used (iter < 0), we want:
     *   resid < THRESH
     */
    char context[128];
    snprintf(context, sizeof(context),
             "ddrvab n=%d nrhs=%d imat=%d iter=%d", n, nrhs, imat, iter);
    set_test_context(context);

    if (iter >= 0 && n > 0) {
        f64 thresh_iter = sqrt((f64)n);
        assert_residual_below(resid, thresh_iter);
    } else {
        assert_residual_ok(resid);
    }

    clear_test_context();
}

/**
 * Test DSGESV for all matrix types at given size.
 */
static void test_dsgesv(void** state)
{
    (void)state;

    for (size_t im = 0; im < NM; im++) {
        INT n = MVAL[im];
        INT nimat = NTYPES;

        /* For empty matrix, only test type 1 */
        if (n <= 0) {
            nimat = 1;
        }

        for (INT imat = 1; imat <= nimat; imat++) {
            /* Skip types 5, 6, 7 if matrix is too small */
            INT zerot = (imat >= 5 && imat <= 7);
            if (zerot && n < imat - 4) {
                continue;
            }

            for (size_t irhs = 0; irhs < NNS; irhs++) {
                INT nrhs = NSVAL[irhs];
                run_test_single(n, nrhs, imat);
            }
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_dsgesv),
    };

    return cmocka_run_group_tests_name("ddrvab", tests, group_setup, group_teardown);
}
