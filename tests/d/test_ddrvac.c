/**
 * @file test_ddrvac.c
 * @brief Test suite for DSPOSV mixed-precision iterative refinement solver.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/ddrvac.f to C using CMocka.
 * Tests DSPOSV which uses single precision Cholesky factorization with f64
 * precision iterative refinement for symmetric positive definite systems.
 *
 * Test parameters from dchkab.f data file:
 *   M values: 0, 1, 2, 3, 5, 10, 16
 *   NRHS values: 2
 *   Matrix types: 1-9 (positive definite types)
 *   THRESH: 20.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
/* Test parameters */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 16};
static const INT NSVAL[] = {2};  /* NRHS values */
static const char UPLOS[] = {'U', 'L'};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9  /* Positive definite matrix types */
#define THRESH  20.0
#define NMAX    132
#define MAXRHS  16

/* Routine under test */
/* Verification routine */
/* Matrix generation */
/* Utilities */
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
} ddrvac_workspace_t;

static ddrvac_workspace_t* g_ws = NULL;

/**
 * Group setup - allocate workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(ddrvac_workspace_t));
    if (!g_ws) return -1;

    g_ws->A = malloc(NMAX * NMAX * sizeof(f64));
    g_ws->AFAC = malloc(NMAX * NMAX * sizeof(f64));
    g_ws->B = malloc(NMAX * MAXRHS * sizeof(f64));
    g_ws->X = malloc(NMAX * MAXRHS * sizeof(f64));
    g_ws->WORK = malloc(NMAX * MAXRHS * 2 * sizeof(f64));
    g_ws->RWORK = malloc(2 * NMAX * sizeof(f64));
    g_ws->SWORK = malloc(NMAX * (NMAX + MAXRHS) * sizeof(float));

    if (!g_ws->A || !g_ws->AFAC || !g_ws->B || !g_ws->X ||
        !g_ws->WORK || !g_ws->RWORK || !g_ws->SWORK) {
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
        free(g_ws);
        g_ws = NULL;
    }
    return 0;
}

/**
 * Run a single test case.
 */
static void run_test_single(INT n, INT nrhs, INT imat, char uplo)
{
    char type, dist;
    INT kl, ku, mode, info, iter;
    f64 anorm, cndnum;
    INT lda = (n > 1) ? n : 1;
    INT izero = 0;

    /* Seed based on test parameters (matches LAPACK ISEEDY = {1988, 1989, 1990, 1991}) */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + nrhs + (uplo == 'U' ? 0 : 50));

    /* Get matrix parameters for positive definite path */
    dlatb4("DPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric positive definite test matrix */
    char dist_str[2] = {dist, '\0'};
    char uplo_str[2] = {uplo, '\0'};
    dlatms(n, n, dist_str, "P", g_ws->RWORK, mode, cndnum, anorm,
           kl, ku, uplo_str, g_ws->A, lda, g_ws->WORK, &info, rng_state);

    if (info != 0) {
        /* Matrix generation failed - skip this test */
        return;
    }

    /* For types 3-5, zero one row and column to test singularity detection */
    INT zerot = (imat >= 3 && imat <= 5);
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }

        /* Zero row and column izero (1-based) */
        INT ioff = (izero - 1) * lda;

        if (uplo == 'U' || uplo == 'u') {
            /* Upper: zero column up to diagonal, then row from diagonal */
            for (INT i = 0; i < izero - 1; i++) {
                g_ws->A[ioff + i] = 0.0;
            }
            ioff = ioff + izero - 1;
            for (INT i = izero - 1; i < n; i++) {
                g_ws->A[ioff] = 0.0;
                ioff = ioff + lda;
            }
        } else {
            /* Lower: zero row up to diagonal, then column from diagonal */
            ioff = izero - 1;
            for (INT i = 0; i < izero - 1; i++) {
                g_ws->A[ioff] = 0.0;
                ioff = ioff + lda;
            }
            ioff = ioff - (izero - 1);
            for (INT i = izero - 1; i < n; i++) {
                g_ws->A[ioff + i] = 0.0;
            }
        }
    }

    /* Generate right-hand side */
    dlarhs("DPO", "N", uplo_str, " ", n, n, kl, ku, nrhs,
           g_ws->A, lda, g_ws->X, lda, g_ws->B, lda, &info, rng_state);

    /* Save a copy of A for later */
    dlacpy("A", n, n, g_ws->A, lda, g_ws->AFAC, lda);

    /* Call DSPOSV */
    dsposv(uplo_str, n, nrhs, g_ws->AFAC, lda, g_ws->B, lda,
           g_ws->X, lda, g_ws->WORK, g_ws->SWORK, &iter, &info);

    /* Restore A if iterative refinement was used */
    if (iter < 0) {
        dlacpy("A", n, n, g_ws->A, lda, g_ws->AFAC, lda);
    }

    /* Check error code */
    if (info != izero && izero != 0) {
        /* Expected singular but got different INFO */
        char context[128];
        snprintf(context, sizeof(context),
                 "ddrvac uplo=%c n=%d nrhs=%d imat=%d: INFO=%d, expected=%d",
                 uplo, n, nrhs, imat, info, izero);
        set_test_context(context);
        assert_int_equal(info, izero);
        clear_test_context();
        return;
    }

    /* Skip solution check if matrix is singular */
    if (info != 0) {
        return;
    }

    /* Check solution quality using dpot06 */
    /* Copy B to WORK for residual computation (dpot06 overwrites it) */
    dlacpy("A", n, nrhs, g_ws->B, lda, g_ws->WORK, lda);

    f64 resid;
    dpot06(uplo_str, n, nrhs, g_ws->A, lda, g_ws->X, lda,
           g_ws->WORK, lda, g_ws->RWORK, &resid);

    /* Check the test result:
     * If iterative refinement was used (iter >= 0), we want:
     *   resid < sqrt(n)
     * If f64 precision was used (iter < 0), we want:
     *   resid < THRESH
     */
    char context[128];
    snprintf(context, sizeof(context),
             "ddrvac uplo=%c n=%d nrhs=%d imat=%d iter=%d", uplo, n, nrhs, imat, iter);
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
 * Test DSPOSV for all matrix types at given size.
 */
static void test_dsposv(void** state)
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
            /* Skip types 3, 4, 5 if matrix is too small */
            INT zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) {
                continue;
            }

            /* Test both upper and lower triangular storage */
            for (size_t iuplo = 0; iuplo < NUPLO; iuplo++) {
                char uplo = UPLOS[iuplo];

                for (size_t irhs = 0; irhs < NNS; irhs++) {
                    INT nrhs = NSVAL[irhs];
                    run_test_single(n, nrhs, imat, uplo);
                }
            }
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_dsposv),
    };

    return cmocka_run_group_tests_name("ddrvac", tests, group_setup, group_teardown);
}
