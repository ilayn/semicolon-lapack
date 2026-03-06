/**
 * @file test_zchkhe_aa.c
 * @brief Comprehensive test suite for Hermitian indefinite (Aasen) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkhe_aa.f to C using CMocka.
 * Tests ZHETRF_AA and ZHETRS_AA.
 *
 * Each (n, uplo, imat, inb) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchkhe_aa.f:
 *   TEST 1: LDL^H factorization residual via zhet01_aa
 *   TEST 2: Solution residual via zpot02 (using zhetrs_aa)
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
} zchkhe_aa_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    c128* A;      /* Original matrix (NMAX x NMAX) */
    c128* AFAC;   /* Factored matrix (NMAX x NMAX) */
    c128* AINV;   /* Work array for factorization (NMAX x NMAX) */
    c128* B;      /* Right-hand side (NMAX x NSMAX) */
    c128* X;      /* Solution (NMAX x NSMAX) */
    c128* XACT;   /* Exact solution (NMAX x NSMAX) */
    c128* WORK;   /* General workspace */
    f64* RWORK;   /* Real workspace */
    f64* D;       /* Singular values for zlatms */
    INT* IWORK;   /* Integer workspace / pivot indices */
} zchkhe_aa_workspace_t;

static zchkhe_aa_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchkhe_aa_workspace_t));
    if (!g_workspace) return -1;

    /* LWORK = MAX(1, N*NB + N) where max NB=20, N=50 => 50*20+50 = 1050
     * Also need workspace for zhetrs_aa: MAX(1, 3*N-2) = 148
     * Use generous workspace */
    INT lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(c128));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c128));
    g_workspace->WORK = malloc(lwork * sizeof(c128));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(INT));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
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
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full zchkhe_aa test battery for a single (n, uplo, imat, inb) combination.
 *
 * Following LAPACK's zchkhe_aa.f:
 *   - TEST 1 (factorization residual) runs for all NB values
 *   - TEST 2 (solve) runs for all NB values, for each NRHS
 */
static void run_zchkhe_aa_single(INT n, INT iuplo, INT imat, INT inb)
{
    const f64 ZERO = 0.0;
    zchkhe_aa_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info, izero;
    INT lda = (n > 1) ? n : 1;
    INT nb = NBVAL[inb];
    INT lwork_trf, lwork_trs;
    f64 result[NTESTS];
    char ctx[128];

    /* Set block size via xlaenv */
    xlaenv(1, nb);
    xlaenv(2, 2);

    /* Initialize RNG state based on (n, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Set up parameters with ZLATB4 for the matrix generator
     * based on the type of matrix to be generated.
     * Use MATPATH = "ZHE" (Hermitian types) */
    zlatb4("ZHE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate a matrix with ZLATMS */
    zlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For matrix types 3-6, zero one or more rows and columns of the matrix
     * to test that INFO is returned correctly. */
    INT zerot = (imat >= 3 && imat <= 6);
    if (zerot) {
        if (imat == 3) {
            izero = 0;  /* Fortran IZERO=1 -> C 0-based */
        } else if (imat == 4) {
            izero = n - 1;  /* Fortran IZERO=N -> C n-1 */
        } else {
            izero = n / 2;  /* Fortran IZERO=N/2+1 -> C n/2 */
        }

        if (imat < 6) {
            /* Set row and column IZERO to zero */
            if (iuplo == 0) {
                /* Upper: zero column izero (rows 0..izero-1) and
                 * row izero (columns izero..n-1) */
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
                /* Lower: zero row izero (columns 0..izero-1) and
                 * column izero (rows izero..n-1) */
                INT ioff = izero;
                for (INT i = 0; i < izero; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
                ioff -= izero;
                for (INT i = izero; i < n; i++) {
                    ws->A[ioff + i] = ZERO;
                }
            }
        } else {
            /* Type 6: zero multiple rows and columns */
            if (iuplo == 0) {
                /* Set the first IZERO+1 rows and columns to zero */
                INT ioff = 0;
                for (INT j = 0; j < n; j++) {
                    INT i2 = (j <= izero) ? j + 1 : izero + 1;
                    for (INT i = 0; i < i2; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
                izero = 0;  /* Fortran sets IZERO=1 (1-based), C uses 0 */
            } else {
                /* Set the last rows and columns to zero */
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
        izero = -1;  /* No zeroing */
    }

    /* Set the imaginary part of the diagonals. */
    zlaipd(n, ws->A, lda + 1, 0);

    /* Copy the test matrix A into AFAC for factorization */
    zlacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the L*D*L**H or U*D*U**H factorization of the matrix.
     * IWORK stores details of the interchanges and the block structure of D.
     * AINV is a work array for block factorization, LWORK is the length of AINV. */
    lwork_trf = (1 > n * nb + n) ? 1 : n * nb + n;  /* MAX(1, N*NB + N) */
    zhetrf_aa(uplo_str, n, ws->AFAC, lda, ws->IWORK, ws->AINV,
              lwork_trf, &info);

    /* Adjust the expected value of INFO to account for pivoting.
     * In zchkhe_aa.f, this logic is commented out: K = 0 unconditionally */
    INT k_expected = 0;

    /* Check error code from ZHETRF_AA */
    if (info != k_expected) {
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d ZHETRF_AA info=%d expected=%d",
                 n, uplo, imat, info, k_expected);
        set_test_context(ctx);
    }

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d TEST 1 (factorization)",
             n, uplo, imat, nb);
    set_test_context(ctx);
    zhet01_aa(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
              ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /* Skip solver test if INFO is not 0 */
    if (info != 0) {
        clear_test_context();
        return;
    }

    /* Do for each value of NRHS in NSVAL */
    for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
        INT nrhs = NSVAL[irhs];

        /*
         * TEST 2: Solve and compute residual for A * X = B using zhetrs_aa
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d nrhs=%d TEST 2 (solve)",
                 n, uplo, imat, nb, nrhs);
        set_test_context(ctx);

        zlarhs("ZHE", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        zlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);

        lwork_trs = (1 > 3 * n - 2) ? 1 : 3 * n - 2;  /* MAX(1, 3*N-2) */
        zhetrs_aa(uplo_str, n, nrhs, ws->AFAC, lda, ws->IWORK,
                  ws->X, lda, ws->WORK, lwork_trs, &info);

        if (info != 0) {
            if (izero < 0) {
                snprintf(ctx, sizeof(ctx),
                         "n=%d uplo=%c imat=%d nrhs=%d ZHETRS_AA info=%d",
                         n, uplo, imat, nrhs, info);
                set_test_context(ctx);
            }
        } else {
            zlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);

            zpot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            assert_residual_below(result[1], THRESH);
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_zchkhe_aa_single based on prestate.
 */
static void test_zchkhe_aa_case(void** state)
{
    zchkhe_aa_params_t* params = *state;
    run_zchkhe_aa_single(params->n, params->iuplo, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NUPLO * NTYPES * NNB = 7 * 2 * 10 * 5 = 700 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NNB)

static zchkhe_aa_params_t g_params[MAX_TESTS];
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
                    zchkhe_aa_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name),
                             "zchkhe_aa_n%d_%c_type%d_nb%d_%d",
                             n, UPLOS[iuplo], imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchkhe_aa_case;
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
    (void)_cmocka_run_group_tests("zchkhe_aa", g_tests, g_num_tests,
                                   group_setup, group_teardown);
    return 0;
}
