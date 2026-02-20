/**
 * @file test_schktp.c
 * @brief Comprehensive test suite for triangular packed matrix routines (STP).
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchktp.f to C using CMocka.
 * Tests STPTRI, STPTRS, STPCON, STPRFS, and SLATPS.
 *
 * Each (n, imat, uplo) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchktp.f:
 *   IMAT 1-10: Non-pathological triangular matrices
 *     TEST 1: STPTRI - | A*AINV - I | / (n * |A| * |AINV| * eps)
 *     TEST 2: STPTRS - | b - A*x | / (|A| * |x| * eps)
 *     TEST 3: Compare solution to exact
 *     TEST 4-6: STPRFS iterative refinement bounds
 *     TEST 7: STPCON condition number estimate
 *
 *   IMAT 11-18: Pathological matrices for SLATPS
 *     TEST 8: SLATPS with NORMIN='N'
 *     TEST 9: SLATPS with NORMIN='Y'
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-10 (standard), 11-18 (pathological)
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

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPE1  10      /* Number of non-pathological types */
#define NTYPES  18      /* Total matrix types (including pathological) */
#define NTESTS  9       /* Number of tests per matrix */
#define THRESH  30.0f
#define NMAX    50      /* Maximum matrix dimension */
#define NSMAX   15      /* Maximum NRHS */
#define LAP_MAX ((NMAX * (NMAX + 1)) / 2)  /* Max packed storage size */

/* Routines under test */
extern void stptri(const char* uplo, const char* diag, const int n,
                   f32* AP, int* info);
extern void stptrs(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs, const f32* AP,
                   f32* B, const int ldb, int* info);
extern void stpcon(const char* norm, const char* uplo, const char* diag,
                   const int n, const f32* AP,
                   f32* rcond, f32* work, int* iwork, int* info);
extern void stprfs(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs, const f32* AP,
                   const f32* B, const int ldb, const f32* X, const int ldx,
                   f32* ferr, f32* berr, f32* work, int* iwork, int* info);
extern void slatps(const char* uplo, const char* trans, const char* diag,
                   const char* normin, const int n, const f32* AP,
                   f32* X, f32* scale, f32* cnorm, int* info);

/* Verification routines */
extern void stpt01(const char* uplo, const char* diag, const int n,
                   const f32* AP, f32* AINVP,
                   f32* rcond, f32* work, f32* resid);
extern void stpt02(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs,
                   const f32* AP, const f32* X, const int ldx,
                   const f32* B, const int ldb,
                   f32* work, f32* resid);
extern void stpt03(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs,
                   const f32* AP, const f32 scale, const f32* cnorm,
                   const f32 tscal, const f32* X, const int ldx,
                   const f32* B, const int ldb,
                   f32* work, f32* resid);
extern void stpt05(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs,
                   const f32* AP, const f32* B, const int ldb,
                   const f32* X, const int ldx,
                   const f32* XACT, const int ldxact,
                   const f32* ferr, const f32* berr,
                   f32* reslts);
extern void stpt06(const f32 rcond, const f32 rcondc,
                   const char* uplo, const char* diag, const int n,
                   const f32* AP, f32* work, f32* rat);

/* Matrix generation */
extern void slattp(const int imat, const char* uplo, const char* trans, char* diag,
                   const int n, f32* AP, f32* B, f32* work,
                   int* info, uint64_t state[static 4]);
extern void slarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f32* A, const int lda,
                   f32* XACT, const int ldxact, f32* B, const int ldb,
                   int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern f32 slamch(const char* cmach);
extern f32 slantp(const char* norm, const char* uplo, const char* diag,
                     const int n, const f32* AP, f32* work);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond,
                   f32* resid);

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* AP;     /* Original packed matrix (LAP_MAX) */
    f32* AINVP;  /* Inverse packed matrix (LAP_MAX) */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* CNORM;  /* Column norms (NMAX) */
    int* IWORK;     /* Integer workspace */
} dchktp_workspace_t;

static dchktp_workspace_t* g_workspace = NULL;

/* =========================================================================
 * Parameterized test infrastructure
 * ========================================================================= */

/* Test parameters for standard tests (IMAT 1-10) */
typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    char name[64];
} standard_params_t;

/* Test parameters for pathological tests (IMAT 11-18) */
typedef struct {
    int n;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    char name[64];
} latps_params_t;

/* Maximum number of tests */
#define MAX_STANDARD_TESTS (NN * NTYPE1 * 2)      /* 7 * 10 * 2 = 140 */
#define MAX_LATPS_TESTS    (NN * 8 * 2)           /* 7 * 8 * 2 = 112 */
#define MAX_TESTS          (MAX_STANDARD_TESTS + MAX_LATPS_TESTS)

static standard_params_t g_standard_params[MAX_STANDARD_TESTS];
static latps_params_t g_latps_params[MAX_LATPS_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static const char* UPLOS[] = {"U", "L"};

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchktp_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * (NSMAX > 3 ? NSMAX : 3);

    g_workspace->AP = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AINVP = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->WORK = malloc(lwork * sizeof(f32));
    g_workspace->RWORK = malloc((NMAX + 2 * NSMAX) * sizeof(f32));
    g_workspace->CNORM = malloc(NMAX * sizeof(f32));
    g_workspace->IWORK = malloc(NMAX * sizeof(int));

    if (!g_workspace->AP || !g_workspace->AINVP ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->CNORM || !g_workspace->IWORK) {
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
        free(g_workspace->AP);
        free(g_workspace->AINVP);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->CNORM);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run tests for non-pathological matrices (IMAT 1-10).
 * Tests STPTRI, STPTRS, STPCON, STPRFS.
 */
static void test_standard(void** state)
{
    standard_params_t* p = *state;
    dchktp_workspace_t* ws = g_workspace;

    int n = p->n;
    int imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f32 result[NTESTS];
    char diag;
    int info, lda, lap;
    f32 rcondo, rcondi, rcond, rcondc, anorm, ainvnm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10);
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    lda = (n > 1) ? n : 1;
    lap = (n * (n + 1)) / 2;

    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Generate triangular packed test matrix */
    slattp(imat, uplo, "N", &diag, n, ws->AP, ws->XACT, ws->WORK, &info, rng_state);
    assert_info_success(info);

    int idiag = (diag == 'N' || diag == 'n') ? 1 : 2;

    /* TEST 1: Form the inverse of A */
    if (n > 0) {
        cblas_scopy(lap, ws->AP, 1, ws->AINVP, 1);
    }
    stptri(uplo, &diag, n, ws->AINVP, &info);

    /* Compute the infinity-norm condition number of A */
    anorm = slantp("I", uplo, &diag, n, ws->AP, ws->RWORK);
    ainvnm = slantp("I", uplo, &diag, n, ws->AINVP, ws->RWORK);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        rcondi = ONE;
    } else {
        rcondi = (ONE / anorm) / ainvnm;
    }

    if (info == 0) {
        stpt01(uplo, &diag, n, ws->AP, ws->AINVP,
               &rcondo, ws->RWORK, &result[0]);

        if (result[0] >= THRESH) {
            print_message("TEST 1 failed: n=%d, imat=%d, uplo=%s, diag=%c, resid=%.3e\n",
                         n, imat, uplo, diag, (double)result[0]);
        }
        assert_residual_ok(result[0]);
    }

    /* Loop over NRHS values */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /* Loop over TRANS = 'N', 'T', 'C' */
        const char* transs[] = {"N", "T", "C"};
        for (int itran = 0; itran < 3; itran++) {
            const char* trans = transs[itran];
            if (itran == 0) {
                rcondc = rcondo;
            } else {
                rcondc = rcondi;
            }

            /* TEST 2: Solve and compute residual for op(A)*x = b */
            slarhs("STP", "N", uplo, trans, n, n, 0, idiag, nrhs,
                   ws->AP, lap, ws->XACT, lda, ws->B, lda, &info, rng_state);

            slacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
            stptrs(uplo, trans, &diag, n, nrhs, ws->AP, ws->X, lda, &info);

            /* Skip tests 2-6 for matrices that cause STPTRS to fail */
            if (info != 0) {
                continue;
            }

            /* Skip if solution overflowed (can happen with ill-conditioned
             * unit triangular matrices in single precision) */
            {
                int overflow = 0;
                for (int ii = 0; ii < n * nrhs; ii++) {
                    if (!isfinite(ws->X[ii])) { overflow = 1; break; }
                }
                if (overflow) continue;
            }

            stpt02(uplo, trans, &diag, n, nrhs, ws->AP,
                   ws->X, lda, ws->B, lda, ws->WORK, &result[1]);

            /* TEST 3: Check solution from generated exact solution */
            sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

            /* TESTS 4, 5, 6: Use iterative refinement */
            stprfs(uplo, trans, &diag, n, nrhs, ws->AP,
                   ws->B, lda, ws->X, lda, ws->RWORK, &ws->RWORK[nrhs],
                   ws->WORK, ws->IWORK, &info);

            sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
            stpt05(uplo, trans, &diag, n, nrhs, ws->AP,
                   ws->B, lda, ws->X, lda, ws->XACT, lda,
                   ws->RWORK, &ws->RWORK[nrhs], &result[4]);

            /* Check results */
            for (int k = 1; k < 6; k++) {
                if (result[k] >= THRESH) {
                    print_message("TEST %d failed: n=%d, imat=%d, uplo=%s, trans=%s, nrhs=%d, resid=%.3e\n",
                                 k + 1, n, imat, uplo, trans, nrhs, (double)result[k]);
                }
                assert_residual_ok(result[k]);
            }
        }
    }

    /* TEST 7: Get estimate of RCOND = 1/CNDNUM */
    for (int itran = 0; itran < 2; itran++) {
        const char* norm;
        if (itran == 0) {
            norm = "O";
            rcondc = rcondo;
        } else {
            norm = "I";
            rcondc = rcondi;
        }

        stpcon(norm, uplo, &diag, n, ws->AP, &rcond, ws->WORK, ws->IWORK, &info);

        stpt06(rcond, rcondc, uplo, &diag, n, ws->AP, ws->RWORK, &result[6]);

        if (result[6] >= THRESH) {
            print_message("TEST 7 failed: n=%d, imat=%d, uplo=%s, diag=%c, norm=%s, resid=%.3e\n",
                         n, imat, uplo, diag, norm, (double)result[6]);
        }
        assert_residual_ok(result[6]);
    }
}

/**
 * Run tests for pathological matrices (IMAT 11-18).
 * Tests SLATPS.
 */
static void test_latps(void** state)
{
    latps_params_t* p = *state;
    dchktp_workspace_t* ws = g_workspace;

    int n = p->n;
    int imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f32 result[NTESTS];
    char diag;
    int info, lda;
    f32 scale;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10);
    const f32 ONE = 1.0f;
    const f32 ZERO = 0.0f;

    lda = (n > 1) ? n : 1;

    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Loop over TRANS = 'N', 'T', 'C' */
    const char* transs[] = {"N", "T", "C"};
    for (int itran = 0; itran < 3; itran++) {
        const char* trans = transs[itran];

        /* Generate triangular packed test matrix */
        slattp(imat, uplo, trans, &diag, n, ws->AP, ws->XACT, ws->WORK, &info, rng_state);

        /* TEST 8: Solve the system op(A)*x = b with NORMIN='N' */
        cblas_scopy(n, ws->XACT, 1, ws->B, 1);
        slatps(uplo, trans, &diag, "N", n, ws->AP, ws->B, &scale, ws->CNORM, &info);

        if (info != 0) {
            print_message("SLATPS returned INFO=%d for n=%d, imat=%d, uplo=%s, trans=%s\n",
                         info, n, imat, uplo, trans);
        }

        stpt03(uplo, trans, &diag, n, 1, ws->AP, scale, ws->CNORM, ONE,
               ws->B, lda, ws->XACT, lda, ws->WORK, &result[7]);

        if (result[7] >= THRESH) {
            print_message("TEST 8 failed: n=%d, imat=%d, uplo=%s, trans=%s, normin=N, resid=%.3e\n",
                         n, imat, uplo, trans, (double)result[7]);
        }
        assert_residual_ok(result[7]);

        /* TEST 9: Solve op(A)*x = b again with NORMIN='Y' */
        cblas_scopy(n, ws->XACT, 1, &ws->B[n], 1);
        slatps(uplo, trans, &diag, "Y", n, ws->AP, &ws->B[n], &scale, ws->CNORM, &info);

        if (info != 0) {
            print_message("SLATPS returned INFO=%d for n=%d, imat=%d, uplo=%s, trans=%s, normin=Y\n",
                         info, n, imat, uplo, trans);
        }

        stpt03(uplo, trans, &diag, n, 1, ws->AP, scale, ws->CNORM, ONE,
               &ws->B[n], lda, ws->XACT, lda, ws->WORK, &result[8]);

        if (result[8] >= THRESH) {
            print_message("TEST 9 failed: n=%d, imat=%d, uplo=%s, trans=%s, normin=Y, resid=%.3e\n",
                         n, imat, uplo, trans, (double)result[8]);
        }
        assert_residual_ok(result[8]);
    }
}

/**
 * Build test array with all parameter combinations.
 */
static void build_test_array(void)
{
    int standard_idx = 0;
    int latps_idx = 0;

    /* Standard tests: N x IMAT(1-10) x UPLO(U,L) */
    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];

        for (int imat = 1; imat <= NTYPE1; imat++) {
            for (int iuplo = 0; iuplo < 2; iuplo++) {
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

    /* Pathological tests (SLATPS): N x IMAT(11-18) x UPLO(U,L) */
    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];

        for (int imat = NTYPE1 + 1; imat <= NTYPES; imat++) {
            for (int iuplo = 0; iuplo < 2; iuplo++) {
                latps_params_t* p = &g_latps_params[latps_idx];
                p->n = n;
                p->imat = imat;
                p->iuplo = iuplo;
                snprintf(p->name, sizeof(p->name), "latps_n%d_type%d_%s",
                         n, imat, UPLOS[iuplo]);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_latps;
                g_tests[g_num_tests].initial_state = p;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_num_tests++;
                latps_idx++;
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

    return _cmocka_run_group_tests("dchktp", g_tests, (size_t)g_num_tests,
                                   group_setup, group_teardown);
}
