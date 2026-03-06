/**
 * @file test_cchktr.c
 * @brief Comprehensive test suite for triangular matrix routines (CTR).
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchktr.f to C using CMocka.
 * Tests CTRTRI, CTRTRS, CTRCON, CTRRFS, and CLATRS.
 *
 * Each (n, imat, uplo) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from zchktr.f:
 *   IMAT 1-10: Non-pathological triangular matrices
 *     TEST 1: CTRTRI - | A*AINV - I | / (n * |A| * |AINV| * eps)
 *     TEST 2: CTRTRS - | b - A*x | / (|A| * |x| * eps)
 *     TEST 3: Compare solution to exact
 *     TEST 4-6: CTRRFS iterative refinement bounds
 *     TEST 7: CTRCON condition number estimate
 *
 *   IMAT 11-18: Pathological matrices for CLATRS
 *     TEST 8: CLATRS with NORMIN='N'
 *     TEST 9: CLATRS with NORMIN='Y'
 *     TEST 10: CLATRS3 (blocked version)
 *
 * Parameters from ztest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NB values: 1, 3, 3, 3, 20
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-10 (standard), 11-18 (pathological)
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include "semicolon_cblas.h"

/* Test parameters from ztest.in */
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NSVAL[] = {1, 2, 15};  /* NRHS values */
static const INT NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes */

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPE1  10      /* Number of non-pathological types */
#define NTYPES  18      /* Total matrix types (including pathological) */
#define NTESTS  10      /* Number of tests per matrix */
#define THRESH  30.0f
#define NMAX    50      /* Maximum matrix dimension */
#define NSMAX   15      /* Maximum NRHS */

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    c64* A;      /* Original matrix (NMAX x NMAX) */
    c64* AINV;   /* Inverse matrix (NMAX x NMAX) */
    c64* B;      /* Right-hand side (NMAX x NSMAX) */
    c64* X;      /* Solution (NMAX x NSMAX) */
    c64* XACT;   /* Exact solution (NMAX x NSMAX) */
    c64* WORK;   /* General workspace */
    f32* RWORK;   /* Real workspace */
    f32* FERR;    /* Forward error bounds (NSMAX) */
    f32* BERR;    /* Backward error bounds (NSMAX) */
    f32* CNORM;   /* Column norms (NMAX) */
} zchktr_workspace_t;

static zchktr_workspace_t* g_workspace = NULL;

/* =========================================================================
 * Parameterized test infrastructure
 * ========================================================================= */

/* Test parameters for standard tests (IMAT 1-10) */
typedef struct {
    INT n;
    INT imat;
    INT iuplo;      /* 0='U', 1='L' */
    char name[64];
} standard_params_t;

/* Test parameters for pathological tests (IMAT 11-18) */
typedef struct {
    INT n;
    INT imat;
    INT iuplo;      /* 0='U', 1='L' */
    char name[64];
} latrs_params_t;

/* Maximum number of tests */
#define MAX_STANDARD_TESTS (NN * NTYPE1 * 2)      /* 7 * 10 * 2 = 140 */
#define MAX_LATRS_TESTS    (NN * 8 * 2)           /* 7 * 8 * 2 = 112 */
#define MAX_TESTS          (MAX_STANDARD_TESTS + MAX_LATRS_TESTS)

static standard_params_t g_standard_params[MAX_STANDARD_TESTS];
static latrs_params_t g_latrs_params[MAX_LATRS_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static const char* UPLOS[] = {"U", "L"};

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(zchktr_workspace_t));
    if (!g_workspace) return -1;

    INT lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(c64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(c64));
    g_workspace->WORK = malloc(lwork * sizeof(c64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
    g_workspace->CNORM = malloc(NMAX * sizeof(f32));

    if (!g_workspace->A || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->FERR || !g_workspace->BERR ||
        !g_workspace->CNORM) {
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
        free(g_workspace->AINV);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->FERR);
        free(g_workspace->BERR);
        free(g_workspace->CNORM);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run tests for non-pathological matrices (IMAT 1-10).
 * Tests CTRTRI, CTRTRS, CTRCON, CTRRFS.
 */
static void test_standard(void** state)
{
    standard_params_t* p = *state;
    zchktr_workspace_t* ws = g_workspace;

    INT n = p->n;
    INT imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f32 result[NTESTS];
    char diag;
    INT info, lda;
    f32 rcondo, rcondi, rcond, rcondc, anorm, ainvnm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10);
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    lda = (n > 1) ? n : 1;

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Generate triangular test matrix */
    clattr(imat, uplo, "N", &diag, n, ws->A, lda, ws->X, ws->WORK, ws->RWORK, &info, rng_state);
    assert_info_success(info);

    INT idiag = (diag == 'N' || diag == 'n') ? 1 : 2;

    /* Loop over block sizes (only test 1 uses multiple block sizes) */
    for (INT inb = 0; inb < (INT)NNB; inb++) {
        /* TEST 1: Form the inverse of A */
        clacpy(uplo, n, n, ws->A, lda, ws->AINV, lda);
        ctrtri(uplo, &diag, n, ws->AINV, lda, &info);

        /* Compute the infinity-norm condition number of A */
        anorm = clantr("I", uplo, &diag, n, n, ws->A, lda, ws->RWORK);
        ainvnm = clantr("I", uplo, &diag, n, n, ws->AINV, lda, ws->RWORK);
        if (anorm <= ZERO || ainvnm <= ZERO) {
            rcondi = ONE;
        } else {
            rcondi = (ONE / anorm) / ainvnm;
        }

        if (info == 0) {
            ctrt01(uplo, &diag, n, ws->A, lda, ws->AINV, lda,
                   &rcondo, ws->RWORK, &result[0]);

            if (result[0] >= THRESH) {
                print_message("TEST 1 failed: n=%d, imat=%d, uplo=%s, diag=%c, nb=%d, resid=%.3e\n",
                             n, imat, uplo, diag, NBVAL[inb], (double)result[0]);
            }
            assert_residual_ok(result[0]);
        }

        /* Skip remaining tests if not first block size */
        if (inb != 0) continue;

        /* Loop over NRHS values */
        for (INT irhs = 0; irhs < (INT)NNS; irhs++) {
            INT nrhs = NSVAL[irhs];

            /* Loop over TRANS = 'N', 'T', 'C' */
            const char* transs[] = {"N", "T", "C"};
            for (INT itran = 0; itran < 3; itran++) {
                const char* trans = transs[itran];
                if (itran == 0) {
                    rcondc = rcondo;
                } else {
                    rcondc = rcondi;
                }

                /* TEST 2: Solve and compute residual for op(A)*x = b */
                clarhs("CTR", "N", uplo, trans, n, n, 0, idiag, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

                clacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
                ctrtrs(uplo, trans, &diag, n, nrhs, ws->A, lda, ws->X, lda, &info);

                /* Skip tests 2-6 for singular matrices (IMAT 5, 6) or if CTRTRS failed */
                if (info != 0 || imat == 5 || imat == 6) {
                    continue;
                }

                ctrt02(uplo, trans, &diag, n, nrhs, ws->A, lda,
                       ws->X, lda, ws->B, lda, ws->WORK, ws->RWORK, &result[1]);

                /* TEST 3: Check solution from generated exact solution */
                cget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

                /* TESTS 4, 5, 6: Use iterative refinement */
                ctrrfs(uplo, trans, &diag, n, nrhs, ws->A, lda,
                       ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
                       ws->WORK, ws->RWORK, &info);

                cget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
                ctrt05(uplo, trans, &diag, n, nrhs, ws->A, lda,
                       ws->B, lda, ws->X, lda, ws->XACT, lda,
                       ws->FERR, ws->BERR, &result[4]);

                /* Check results */
                for (INT k = 1; k < 6; k++) {
                    if (result[k] >= THRESH) {
                        print_message("TEST %d failed: n=%d, imat=%d, uplo=%s, trans=%s, nrhs=%d, resid=%.3e\n",
                                     k + 1, n, imat, uplo, trans, nrhs, (double)result[k]);
                    }
                    assert_residual_ok(result[k]);
                }
            }
        }

        /* TEST 7: Get estimate of RCOND = 1/CNDNUM */
        if (imat != 5 && imat != 6) {
            for (INT itran = 0; itran < 2; itran++) {
                const char* norm;
                if (itran == 0) {
                    norm = "O";
                    rcondc = rcondo;
                } else {
                    norm = "I";
                    rcondc = rcondi;
                }

                ctrcon(norm, uplo, &diag, n, ws->A, lda, &rcond,
                       ws->WORK, ws->RWORK, &info);
                if (info != 0) {
                    print_message("CTRCON failed: info=%d\n", info);
                }

                ctrt06(rcond, rcondc, uplo, &diag, n, ws->A, lda, ws->RWORK, &result[6]);

                if (result[6] >= THRESH) {
                    print_message("TEST 7 failed: n=%d, imat=%d, uplo=%s, norm=%s, resid=%.3e\n",
                                 n, imat, uplo, norm, (double)result[6]);
                }
                assert_residual_ok(result[6]);
            }
        }
    }
}

/**
 * Run tests for pathological matrices (IMAT 11-18).
 * Tests CLATRS and CLATRS3 with scaling.
 */
static void test_latrs(void** state)
{
    latrs_params_t* p = *state;
    zchktr_workspace_t* ws = g_workspace;

    INT n = p->n;
    INT imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f32 result[NTESTS];
    char diag;
    INT info, lda;
    f32 scale;
    f32 scale3[2];
    f32 res;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10);
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    f32 bignum = slamch("O") / slamch("P");

    lda = (n > 1) ? n : 1;
    INT ldb = (n > 1) ? n : 1;

    for (INT k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Loop over TRANS = 'N', 'T', 'C' */
    const char* transs[] = {"N", "T", "C"};
    for (INT itran = 0; itran < 3; itran++) {
        const char* trans = transs[itran];

        /* Generate triangular test matrix */
        clattr(imat, uplo, trans, &diag, n, ws->A, lda, ws->X, ws->WORK, ws->RWORK, &info, rng_state);
        if (info != 0) {
            print_message("CLATTR failed: info=%d, imat=%d\n", info, imat);
        }

        /* TEST 8: Solve op(A)*x = b with NORMIN='N' */
        cblas_ccopy(n, ws->X, 1, ws->B, 1);
        char normin = 'N';
        clatrs(uplo, trans, &diag, &normin, n, ws->A, lda, ws->B, &scale,
               ws->CNORM, &info);
        if (info != 0) {
            print_message("CLATRS failed: info=%d, imat=%d, uplo=%s, trans=%s\n",
                         info, imat, uplo, trans);
        }

        ctrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale,
               ws->CNORM, ONE, ws->B, lda, ws->X, lda, ws->WORK, &result[7]);

        /* TEST 9: Solve again with NORMIN='Y' */
        cblas_ccopy(n, ws->X, 1, &ws->B[n], 1);
        normin = 'Y';
        clatrs(uplo, trans, &diag, &normin, n, ws->A, lda, &ws->B[n], &scale,
               ws->CNORM, &info);
        if (info != 0) {
            print_message("CLATRS (NORMIN=Y) failed: info=%d\n", info);
        }

        ctrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale,
               ws->CNORM, ONE, &ws->B[n], lda, ws->X, lda, ws->WORK, &result[8]);

        /* TEST 10: Solve op(A)*X = B with CLATRS3 (blocked, 2 RHS) */
        cblas_ccopy(n, ws->X, 1, ws->B, 1);
        cblas_ccopy(n, ws->X, 1, &ws->B[n], 1);
        cblas_csscal(n, bignum, &ws->B[n], 1);

        normin = 'N';
        INT lwork_latrs3 = NMAX;
        clatrs3(uplo, trans, &diag, &normin, n, 2, ws->A, lda,
                ws->B, ldb, scale3, ws->CNORM, ws->RWORK, lwork_latrs3, &info);
        if (info != 0) {
            print_message("CLATRS3 failed: info=%d, imat=%d, uplo=%s, trans=%s\n",
                         info, imat, uplo, trans);
        }

        /* Verify first column */
        ctrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale3[0],
               ws->CNORM, ONE, ws->B, lda, ws->X, lda, ws->WORK, &result[9]);

        /* Verify second column */
        cblas_csscal(n, bignum, ws->X, 1);
        ctrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale3[1],
               ws->CNORM, ONE, &ws->B[n], lda, ws->X, lda, ws->WORK, &res);
        if (res > result[9]) {
            result[9] = res;
        }

        /* Check results */
        if (result[7] >= THRESH) {
            print_message("TEST 8 failed: n=%d, imat=%d, uplo=%s, trans=%s, resid=%.3e\n",
                         n, imat, uplo, trans, (double)result[7]);
        }
        if (result[8] >= THRESH) {
            print_message("TEST 9 failed: n=%d, imat=%d, uplo=%s, trans=%s, resid=%.3e\n",
                         n, imat, uplo, trans, (double)result[8]);
        }
        if (result[9] >= THRESH) {
            print_message("TEST 10 failed: n=%d, imat=%d, uplo=%s, trans=%s, resid=%.3e\n",
                         n, imat, uplo, trans, (double)result[9]);
        }
        assert_residual_ok(result[7]);
        assert_residual_ok(result[8]);
        assert_residual_ok(result[9]);
    }
}

/**
 * Build the parameterized test array.
 */
static void build_test_array(void)
{
    g_num_tests = 0;
    INT standard_idx = 0;
    INT latrs_idx = 0;

    /* Standard tests: N x IMAT(1-10) x UPLO(U,L) */
    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];

        for (INT imat = 1; imat <= NTYPE1; imat++) {
            for (INT iuplo = 0; iuplo < 2; iuplo++) {
                standard_params_t* p = &g_standard_params[standard_idx];
                p->n = n;
                p->imat = imat;
                p->iuplo = iuplo;
                snprintf(p->name, sizeof(p->name), "n%d_type%d_%s",
                         n, imat, UPLOS[iuplo]);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_standard;
                g_tests[g_num_tests].initial_state = p;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_num_tests++;
                standard_idx++;
            }
        }
    }

    /* Pathological tests (CLATRS): N x IMAT(11-18) x UPLO(U,L) */
    for (INT in = 0; in < (INT)NN; in++) {
        INT n = NVAL[in];

        for (INT imat = NTYPE1 + 1; imat <= NTYPES; imat++) {
            for (INT iuplo = 0; iuplo < 2; iuplo++) {
                latrs_params_t* p = &g_latrs_params[latrs_idx];
                p->n = n;
                p->imat = imat;
                p->iuplo = iuplo;
                snprintf(p->name, sizeof(p->name), "latrs_n%d_type%d_%s",
                         n, imat, UPLOS[iuplo]);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_latrs;
                g_tests[g_num_tests].initial_state = p;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_num_tests++;
                latrs_idx++;
            }
        }
    }
}

/**
 * Main entry point.
 */
int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("zchktr", g_tests, (size_t)g_num_tests,
                                   group_setup, group_teardown);
}
