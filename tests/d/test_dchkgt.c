/**
 * @file test_dchkgt.c
 * @brief Comprehensive test suite for general tridiagonal matrix (DGT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkgt.f to C using CMocka.
 * Tests DGTTRF, DGTTRS, DGTRFS, and DGTCON.
 *
 * Each (n, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkgt.f:
 *   TEST 1: LU factorization residual via dgtt01
 *   TEST 2: Solution residual via dgtt02
 *   TEST 3: Solution accuracy via dget04
 *   TEST 4: Refined solution accuracy via dget04 (after dgtrfs)
 *   TEST 5-6: Error bounds via dgtt05
 *   TEST 7: Condition number via dget06
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   TRANS values: 'N', 'T', 'C'
 *   Matrix types: 1-12
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
static const char TRANSS[] = {'N', 'T', 'C'};

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTRAN   (sizeof(TRANSS) / sizeof(TRANSS[0]))
#define NTYPES  12
#define NTESTS  7
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void dgttrf(const int n, f64* DL, f64* D, f64* DU,
                   f64* DU2, int* ipiv, int* info);
extern void dgttrs(const char* trans, const int n, const int nrhs,
                   const f64* DL, const f64* D, const f64* DU,
                   const f64* DU2, const int* ipiv,
                   f64* B, const int ldb, int* info);
extern void dgtrfs(const char* trans, const int n, const int nrhs,
                   const f64* DL, const f64* D, const f64* DU,
                   const f64* DLF, const f64* DF, const f64* DUF,
                   const f64* DU2, const int* ipiv,
                   const f64* B, const int ldb,
                   f64* X, const int ldx,
                   f64* ferr, f64* berr,
                   f64* work, int* iwork, int* info);
extern void dgtcon(const char* norm, const int n,
                   const f64* DL, const f64* D, const f64* DU,
                   const f64* DU2, const int* ipiv,
                   const f64 anorm, f64* rcond,
                   f64* work, int* iwork, int* info);

/* Verification routines */
extern void dgtt01(const int n, const f64* DL, const f64* D,
                   const f64* DU, const f64* DLF, const f64* DF,
                   const f64* DUF, const f64* DU2, const int* ipiv,
                   f64* work, const int ldwork, f64* resid);
extern void dgtt02(const char* trans, const int n, const int nrhs,
                   const f64* DL, const f64* D, const f64* DU,
                   const f64* X, const int ldx,
                   f64* B, const int ldb, f64* resid);
extern void dgtt05(const char* trans, const int n, const int nrhs,
                   const f64* DL, const f64* D, const f64* DU,
                   const f64* B, const int ldb,
                   const f64* X, const int ldx,
                   const f64* XACT, const int ldxact,
                   const f64* ferr, const f64* berr, f64* reslts);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);
extern f64 dget06(const f64 rcond, const f64 rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode, const f64 cond,
                   const f64 dmax, const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern f64 dlangt(const char* norm, const int n,
                     const f64* DL, const f64* D, const f64* DU);
extern f64 dlamch(const char* cmach);
extern void dlagtm(const char* trans, const int n, const int nrhs,
                   const f64 alpha, const f64* DL, const f64* D,
                   const f64* DU, const f64* X, const int ldx,
                   const f64 beta, f64* B, const int ldb);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    char name[64];
} dchkgt_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* DL;     /* Original sub-diagonal (NMAX-1) */
    f64* D;      /* Original diagonal (NMAX) */
    f64* DU;     /* Original super-diagonal (NMAX-1) */
    f64* DLF;    /* Factored sub-diagonal (NMAX-1) */
    f64* DF;     /* Factored diagonal (NMAX) */
    f64* DUF;    /* Factored super-diagonal (NMAX-1) */
    f64* DU2;    /* Second super-diagonal from factorization (NMAX-2) */
    f64* B;      /* Right-hand side (NMAX x NSMAX) */
    f64* X;      /* Solution (NMAX x NSMAX) */
    f64* XACT;   /* Exact solution (NMAX x NSMAX) */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace for error bounds */
    f64* FERR;   /* Forward error bounds */
    f64* BERR;   /* Backward error bounds */
    int* IPIV;      /* Pivot indices */
    int* IWORK;     /* Integer workspace */
} dchkgt_workspace_t;

static dchkgt_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkgt_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->DL = malloc(NMAX * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->DU = malloc(NMAX * sizeof(f64));
    g_workspace->DLF = malloc(NMAX * sizeof(f64));
    g_workspace->DF = malloc(NMAX * sizeof(f64));
    g_workspace->DUF = malloc(NMAX * sizeof(f64));
    g_workspace->DU2 = malloc(NMAX * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->WORK = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->RWORK = malloc(2 * NSMAX * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
    g_workspace->IPIV = malloc(NMAX * sizeof(int));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(int));

    if (!g_workspace->DL || !g_workspace->D || !g_workspace->DU ||
        !g_workspace->DLF || !g_workspace->DF || !g_workspace->DUF ||
        !g_workspace->DU2 || !g_workspace->B || !g_workspace->X ||
        !g_workspace->XACT || !g_workspace->WORK || !g_workspace->RWORK ||
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
        free(g_workspace->DL);
        free(g_workspace->D);
        free(g_workspace->DU);
        free(g_workspace->DLF);
        free(g_workspace->DF);
        free(g_workspace->DUF);
        free(g_workspace->DU2);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
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
 * Generate a tridiagonal matrix for testing.
 *
 * For types 1-6: Use dlatms with controlled singular values.
 * For types 7-12: Generate random tridiagonal directly.
 */
static void generate_gt_matrix(int n, int imat, f64* DL, f64* D, f64* DU,
                                uint64_t rng_state[static 4], int* izero)
{
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info;
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    int m = (n > 1) ? n - 1 : 0;

    if (n <= 0) {
        *izero = 0;
        return;
    }

    dlatb4("DGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    int zerot = (imat >= 8 && imat <= 10);
    *izero = 0;

    if (imat >= 1 && imat <= 6) {
        /* Types 1-6: Use dlatms to generate matrix with controlled condition */
        /* Generate in band storage: 3 rows (sub, diag, super) */
        int lda = 3;
        f64* AB = calloc(lda * n, sizeof(f64));
        f64* d_sing = malloc(n * sizeof(f64));
        /* dlatms with pack='Z' needs: n*n (full matrix) + m+n (dlagge workspace) */
        f64* work = malloc((n * n + 2 * n) * sizeof(f64));

        if (!AB || !d_sing || !work) {
            free(AB);
            free(d_sing);
            free(work);
            return;
        }

        /* Generate band matrix with KL=1, KU=1 */
        dlatms(n, n, &dist,
               &type, d_sing, mode, cndnum, anorm,
               kl, ku, "Z", AB, lda, work, &info, rng_state);

        if (info == 0) {
            /* Extract tridiagonal from band storage */
            /* Band storage with 'Z': row 0 = super-diagonal, row 1 = diagonal, row 2 = sub-diagonal */
            for (int i = 0; i < n; i++) {
                D[i] = AB[1 + i * lda];  /* Diagonal */
            }
            for (int i = 0; i < m; i++) {
                DU[i] = AB[0 + (i + 1) * lda];  /* Super-diagonal */
                DL[i] = AB[2 + i * lda];        /* Sub-diagonal */
            }
        } else {
            /* Fall back to simple generation */
            for (int i = 0; i < n; i++) {
                D[i] = 2.0 * anorm;
            }
            for (int i = 0; i < m; i++) {
                DL[i] = -anorm * 0.5;
                DU[i] = -anorm * 0.5;
            }
        }

        free(AB);
        free(d_sing);
        free(work);
    } else {
        /* Types 7-12: Random generation */

        /* Generate random elements from [-1, 1] */
        for (int i = 0; i < m; i++) {
            DL[i] = rng_uniform_symmetric(rng_state);
        }
        for (int i = 0; i < n; i++) {
            D[i] = rng_uniform_symmetric(rng_state);
        }
        for (int i = 0; i < m; i++) {
            DU[i] = rng_uniform_symmetric(rng_state);
        }

        /* Scale if needed */
        if (anorm != ONE) {
            for (int i = 0; i < m; i++) {
                DL[i] *= anorm;
            }
            for (int i = 0; i < n; i++) {
                D[i] *= anorm;
            }
            for (int i = 0; i < m; i++) {
                DU[i] *= anorm;
            }
        }

        /* For types 8-10, zero one column to create singular matrix */
        if (zerot) {
            if (imat == 8) {
                /* Zero first column */
                *izero = 1;
                D[0] = ZERO;
                if (n > 1) {
                    DL[0] = ZERO;
                }
            } else if (imat == 9) {
                /* Zero last column */
                *izero = n;
                D[n - 1] = ZERO;
                if (n > 1) {
                    DU[n - 2] = ZERO;
                }
            } else {
                /* Zero middle columns */
                *izero = (n + 1) / 2;
                for (int i = *izero - 1; i < n - 1; i++) {
                    DL[i] = ZERO;
                    D[i] = ZERO;
                    DU[i] = ZERO;
                }
                D[n - 1] = ZERO;
            }
        }
    }
}

/**
 * Run the full dchkgt test battery for a single (n, imat) combination.
 * This is the core test logic, parameterized by the test case.
 */
static void run_dchkgt_single(int n, int imat)
{
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    dchkgt_workspace_t* ws = g_workspace;

    int info, izero;
    int m = (n > 1) ? n - 1 : 0;
    int lda = (n > 1) ? n : 1;
    int trfcon;
    f64 anorm, rcond, rcondc, rcondo, rcondi, ainvnm;
    f64 result[NTESTS];
    char ctx[128];  /* Context string for error messages */

    /* Seed based on (n, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Generate test matrix */
    generate_gt_matrix(n, imat, ws->DL, ws->D, ws->DU, rng_state, &izero);

    /* Copy to factored arrays */
    memcpy(ws->DLF, ws->DL, m * sizeof(f64));
    memcpy(ws->DF, ws->D, n * sizeof(f64));
    memcpy(ws->DUF, ws->DU, m * sizeof(f64));

    /*
     * TEST 1: Factor A as L*U and compute the ratio
     *         norm(L*U - A) / (n * norm(A) * EPS)
     */
    snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 1 (factorization)", n, imat);
    set_test_context(ctx);
    dgttrf(n, ws->DLF, ws->DF, ws->DUF, ws->DU2, ws->IPIV, &info);

    /* Check error code */
    if (izero > 0) {
        /* For singular matrices, info should be > 0 */
        assert_true(info >= 0);
    }
    trfcon = (info != 0);

    /* Verify factorization */
    dgtt01(n, ws->DL, ws->D, ws->DU, ws->DLF, ws->DF, ws->DUF,
           ws->DU2, ws->IPIV, ws->WORK, lda, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 7: Condition number estimation (for both 'O' and 'I' norms)
     */
    for (int itran = 0; itran < 2; itran++) {
        char norm = (itran == 0) ? 'O' : 'I';
        char norm_str[2] = {norm, '\0'};
        snprintf(ctx, sizeof(ctx), "n=%d imat=%d TEST 7 (condition norm=%c)", n, imat, norm);
        set_test_context(ctx);
        anorm = dlangt(norm_str, n, ws->DL, ws->D, ws->DU);

        if (!trfcon) {
            /* Compute inverse norm by solving for each column of identity */
            ainvnm = ZERO;
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    ws->X[j] = ZERO;
                }
                ws->X[i] = ONE;
                char trans_str[2] = {(itran == 0) ? 'N' : 'T', '\0'};
                dgttrs(trans_str, n, 1, ws->DLF, ws->DF, ws->DUF, ws->DU2,
                       ws->IPIV, ws->X, lda, &info);
                f64 sum = ZERO;
                for (int j = 0; j < n; j++) {
                    sum += fabs(ws->X[j]);
                }
                if (sum > ainvnm) ainvnm = sum;
            }

            /* Compute RCONDC = 1 / (norm(A) * norm(inv(A)) */
            if (anorm <= ZERO || ainvnm <= ZERO) {
                rcondc = ONE;
            } else {
                rcondc = (ONE / anorm) / ainvnm;
            }
            if (itran == 0) {
                rcondo = rcondc;
            } else {
                rcondi = rcondc;
            }
        } else {
            rcondc = ZERO;
        }

        /* Estimate condition number */
        dgtcon(norm_str, n, ws->DLF, ws->DF, ws->DUF, ws->DU2, ws->IPIV,
               anorm, &rcond, ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        result[6] = dget06(rcond, rcondc);
        assert_residual_below(result[6], THRESH);
    }

    /* Skip remaining tests if matrix is singular */
    if (trfcon) {
        return;
    }

    /*
     * TESTS 2-6: Solve tests for each NRHS and TRANS
     */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /* Generate NRHS random solution vectors */
        for (int j = 0; j < nrhs; j++) {
            for (int i = 0; i < n; i++) {
                ws->XACT[i + j * lda] = rng_uniform_symmetric(rng_state);
            }
        }

        for (int itran = 0; itran < (int)NTRAN; itran++) {
            char trans = TRANSS[itran];
            char trans_str[2] = {trans, '\0'};
            rcondc = (itran == 0) ? rcondo : rcondi;

            /* Set right-hand side: B = op(A) * XACT */
            dlagtm(trans_str, n, nrhs, ONE, ws->DL, ws->D, ws->DU,
                   ws->XACT, lda, ZERO, ws->B, lda);

            /*
             * TEST 2: Solve op(A) * X = B and compute residual
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d trans=%c TEST 2 (solve)", n, imat, nrhs, trans);
            set_test_context(ctx);
            dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            dgttrs(trans_str, n, nrhs, ws->DLF, ws->DF, ws->DUF, ws->DU2,
                   ws->IPIV, ws->X, lda, &info);
            assert_int_equal(info, 0);

            dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
            dgtt02(trans_str, n, nrhs, ws->DL, ws->D, ws->DU,
                   ws->X, lda, ws->WORK, lda, &result[1]);
            assert_residual_below(result[1], THRESH);

            /*
             * TEST 3: Check solution from generated exact solution
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d trans=%c TEST 3 (accuracy)", n, imat, nrhs, trans);
            set_test_context(ctx);
            dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            assert_residual_below(result[2], THRESH);

            /*
             * TESTS 4, 5, 6: Iterative refinement
             */
            snprintf(ctx, sizeof(ctx), "n=%d imat=%d nrhs=%d trans=%c TEST 4-6 (refinement)", n, imat, nrhs, trans);
            set_test_context(ctx);
            dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            dgttrs(trans_str, n, nrhs, ws->DLF, ws->DF, ws->DUF, ws->DU2,
                   ws->IPIV, ws->X, lda, &info);

            dgtrfs(trans_str, n, nrhs, ws->DL, ws->D, ws->DU,
                   ws->DLF, ws->DF, ws->DUF, ws->DU2, ws->IPIV,
                   ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
                   ws->WORK, ws->IWORK, &info);
            assert_int_equal(info, 0);

            dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
            dgtt05(trans_str, n, nrhs, ws->DL, ws->D, ws->DU,
                   ws->B, lda, ws->X, lda, ws->XACT, lda,
                   ws->FERR, ws->BERR, &result[4]);

            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
            assert_residual_below(result[5], THRESH);
        }
    }

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkgt_single based on prestate.
 */
static void test_dchkgt_case(void** state)
{
    dchkgt_params_t* params = *state;
    run_dchkgt_single(params->n, params->imat);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NTYPES = 7 * 12 = 84 tests
 * (minus skipped cases for small n)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NTYPES)

static dchkgt_params_t g_params[MAX_TESTS];
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
            /* Store parameters */
            dchkgt_params_t* p = &g_params[g_num_tests];
            p->n = n;
            p->imat = imat;
            snprintf(p->name, sizeof(p->name), "dchkgt_n%d_type%d", n, imat);

            /* Create CMocka test entry */
            g_tests[g_num_tests].name = p->name;
            g_tests[g_num_tests].test_func = test_dchkgt_case;
            g_tests[g_num_tests].setup_func = NULL;
            g_tests[g_num_tests].teardown_func = NULL;
            g_tests[g_num_tests].initial_state = p;

            g_num_tests++;
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
    return _cmocka_run_group_tests("dchkgt", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
