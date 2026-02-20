/**
 * @file test_dchksy_aa.c
 * @brief Comprehensive test suite for symmetric indefinite (Aasen) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchksy_aa.f to C using CMocka.
 * Tests DSYTRF_AA and DSYTRS_AA.
 *
 * Each (n, uplo, imat, inb) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchksy_aa.f:
 *   TEST 1: LDL^T factorization residual via dsyt01_aa
 *   TEST 2: Solution residual via dpot02 (using dsytrs_aa)
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
extern void dsytrf_aa(const char* uplo, const int n, f64* A, const int lda,
                      int* ipiv, f64* work, const int lwork, int* info);
extern void dsytrs_aa(const char* uplo, const int n, const int nrhs,
                      const f64* A, const int lda, const int* ipiv,
                      f64* B, const int ldb, f64* work, const int lwork,
                      int* info);

/* Verification routines */
extern void dsyt01_aa(const char* uplo, const int n, const f64* A,
                      const int lda, const f64* AFAC, const int ldafac,
                      const int* ipiv, f64* C, const int ldc,
                      f64* rwork, f64* resid);
extern void dpot02(const char* uplo, const int n, const int nrhs,
                   const f64* A, const int lda, const f64* X,
                   const int ldx, f64* B, const int ldb,
                   f64* rwork, f64* resid);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode, const f64 cond,
                   const f64 dmax, const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f64* A, const int lda,
                   f64* X, const int ldx, f64* B,
                   const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    int iuplo;  /* 0='U', 1='L' */
    int inb;    /* Index into NBVAL[] */
    char name[64];
} dchksy_aa_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Original matrix (NMAX x NMAX) */
    f64* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f64* AINV;   /* Work array for factorization (NMAX x NMAX) */
    f64* B;      /* Right-hand side (NMAX x NSMAX) */
    f64* X;      /* Solution (NMAX x NSMAX) */
    f64* XACT;   /* Exact solution (NMAX x NSMAX) */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace */
    f64* D;      /* Singular values for dlatms */
    int* IWORK;     /* Integer workspace / pivot indices */
} dchksy_aa_workspace_t;

static dchksy_aa_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchksy_aa_workspace_t));
    if (!g_workspace) return -1;

    /* LWORK = MAX(1, N*NB + N) where max NB=20, N=50 => 50*20+50 = 1050
     * Also need workspace for dsytrs_aa: MAX(1, 3*N-2) = 148
     * Use generous workspace */
    int lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(int));

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
 * Run the full dchksy_aa test battery for a single (n, uplo, imat, inb) combination.
 *
 * Following LAPACK's dchksy_aa.f:
 *   - TEST 1 (factorization residual) runs for all NB values
 *   - TEST 2 (solve) runs for all NB values, for each NRHS
 */
static void run_dchksy_aa_single(int n, int iuplo, int imat, int inb)
{
    const f64 ZERO = 0.0;
    dchksy_aa_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info, izero;
    int lda = (n > 1) ? n : 1;
    int nb = NBVAL[inb];
    int lwork_trf, lwork_trs;
    f64 result[NTESTS];
    char ctx[128];

    /* Set block size via xlaenv */
    xlaenv(1, nb);
    xlaenv(2, 2);

    /* Initialize RNG state based on (n, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Set up parameters with DLATB4 for the matrix generator
     * based on the type of matrix to be generated.
     * Use MATPATH = "DSY" (standard symmetric types) */
    dlatb4("DSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate a matrix with DLATMS */
    dlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For matrix types 3-6, zero one or more rows and columns of the matrix
     * to test that INFO is returned correctly. */
    int zerot = (imat >= 3 && imat <= 6);
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
                /* Lower: zero row izero (columns 0..izero-1) and
                 * column izero (rows izero..n-1) */
                int ioff = izero;
                for (int i = 0; i < izero; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
                ioff -= izero;
                for (int i = izero; i < n; i++) {
                    ws->A[ioff + i] = ZERO;
                }
            }
        } else {
            /* Type 6: zero multiple rows and columns */
            if (iuplo == 0) {
                /* Set the first IZERO+1 rows and columns to zero */
                int ioff = 0;
                for (int j = 0; j < n; j++) {
                    int i2 = (j <= izero) ? j + 1 : izero + 1;
                    for (int i = 0; i < i2; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
                izero = 0;  /* Fortran sets IZERO=1 (1-based), C uses 0 */
            } else {
                /* Set the last rows and columns to zero */
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
        izero = -1;  /* No zeroing */
    }

    /* Copy the test matrix A into AFAC for factorization */
    dlacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the L*D*L**T or U*D*U**T factorization of the matrix.
     * IWORK stores details of the interchanges and the block structure of D.
     * AINV is a work array for block factorization, LWORK is the length of AINV. */
    lwork_trf = (1 > n * nb + n) ? 1 : n * nb + n;  /* MAX(1, N*NB + N) */
    dsytrf_aa(uplo_str, n, ws->AFAC, lda, ws->IWORK, ws->AINV,
              lwork_trf, &info);

    /* Adjust the expected value of INFO to account for pivoting.
     * In dchksy_aa.f, this logic is commented out: K = 0 unconditionally */
    int k_expected = 0;

    /* Check error code from DSYTRF_AA */
    if (info != k_expected) {
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d DSYTRF_AA info=%d expected=%d",
                 n, uplo, imat, info, k_expected);
        set_test_context(ctx);
    }

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d TEST 1 (factorization)",
             n, uplo, imat, nb);
    set_test_context(ctx);
    dsyt01_aa(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->IWORK,
              ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /* Skip solver test if INFO is not 0 */
    if (info != 0) {
        clear_test_context();
        return;
    }

    /* Do for each value of NRHS in NSVAL */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /*
         * TEST 2: Solve and compute residual for A * X = B using dsytrs_aa
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d nrhs=%d TEST 2 (solve)",
                 n, uplo, imat, nb, nrhs);
        set_test_context(ctx);

        dlarhs("DSY", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);

        lwork_trs = (1 > 3 * n - 2) ? 1 : 3 * n - 2;  /* MAX(1, 3*N-2) */
        dsytrs_aa(uplo_str, n, nrhs, ws->AFAC, lda, ws->IWORK,
                  ws->X, lda, ws->WORK, lwork_trs, &info);

        if (info != 0) {
            if (izero < 0) {
                snprintf(ctx, sizeof(ctx),
                         "n=%d uplo=%c imat=%d nrhs=%d DSYTRS_AA info=%d",
                         n, uplo, imat, nrhs, info);
                set_test_context(ctx);
            }
        } else {
            dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);

            dpot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
                   ws->WORK, lda, ws->RWORK, &result[1]);

            assert_residual_below(result[1], THRESH);
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchksy_aa_single based on prestate.
 */
static void test_dchksy_aa_case(void** state)
{
    dchksy_aa_params_t* params = *state;
    run_dchksy_aa_single(params->n, params->iuplo, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NUPLO * NTYPES * NNB = 7 * 2 * 10 * 5 = 700 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NNB)

static dchksy_aa_params_t g_params[MAX_TESTS];
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
                    dchksy_aa_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name),
                             "dchksy_aa_n%d_%c_type%d_nb%d_%d",
                             n, UPLOS[iuplo], imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchksy_aa_case;
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
    return _cmocka_run_group_tests("dchksy_aa", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
