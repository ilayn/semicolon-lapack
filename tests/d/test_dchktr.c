/**
 * @file test_dchktr.c
 * @brief Comprehensive test suite for triangular matrix routines (DTR).
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchktr.f to C using CMocka.
 * Tests DTRTRI, DTRTRS, DTRCON, DTRRFS, and DLATRS.
 *
 * Each (n, imat, uplo) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchktr.f:
 *   IMAT 1-10: Non-pathological triangular matrices
 *     TEST 1: DTRTRI - | A*AINV - I | / (n * |A| * |AINV| * eps)
 *     TEST 2: DTRTRS - | b - A*x | / (|A| * |x| * eps)
 *     TEST 3: Compare solution to exact
 *     TEST 4-6: DTRRFS iterative refinement bounds
 *     TEST 7: DTRCON condition number estimate
 *
 *   IMAT 11-18: Pathological matrices for DLATRS
 *     TEST 8: DLATRS with NORMIN='N'
 *     TEST 9: DLATRS with NORMIN='Y'
 *     TEST 10: DLATRS3 (blocked version)
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NB values: 1, 3, 3, 3, 20
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
static const int NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes */

#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTYPE1  10      /* Number of non-pathological types */
#define NTYPES  18      /* Total matrix types (including pathological) */
#define NTESTS  10      /* Number of tests per matrix */
#define THRESH  30.0
#define NMAX    50      /* Maximum matrix dimension */
#define NSMAX   15      /* Maximum NRHS */

/* Routines under test */
extern void dtrtri(const char* uplo, const char* diag, const int n,
                   f64* A, const int lda, int* info);
extern void dtrtrs(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs, const f64* A, const int lda,
                   f64* B, const int ldb, int* info);
extern void dtrcon(const char* norm, const char* uplo, const char* diag,
                   const int n, const f64* A, const int lda,
                   f64* rcond, f64* work, int* iwork, int* info);
extern void dtrrfs(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs, const f64* A, const int lda,
                   const f64* B, const int ldb, const f64* X, const int ldx,
                   f64* ferr, f64* berr, f64* work, int* iwork, int* info);
extern void dlatrs(const char* uplo, const char* trans, const char* diag,
                   const char* normin, const int n, const f64* A, const int lda,
                   f64* X, f64* scale, f64* cnorm, int* info);
extern void dlatrs3(const char* uplo, const char* trans, const char* diag,
                    const char* normin, const int n, const int nrhs,
                    const f64* A, const int lda, f64* X, const int ldx,
                    f64* scale, f64* cnorm, f64* work, const int lwork, int* info);

/* Verification routines */
extern void dtrt01(const char* uplo, const char* diag, const int n,
                   const f64* A, const int lda, f64* AINV, const int ldainv,
                   f64* rcond, f64* work, f64* resid);
extern void dtrt02(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs, const f64* A, const int lda,
                   const f64* X, const int ldx, const f64* B, const int ldb,
                   f64* work, f64* resid);
extern void dtrt03(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs, const f64* A, const int lda,
                   const f64 scale, const f64* cnorm, const f64 tscal,
                   const f64* X, const int ldx, const f64* B, const int ldb,
                   f64* work, f64* resid);
extern void dtrt05(const char* uplo, const char* trans, const char* diag,
                   const int n, const int nrhs, const f64* A, const int lda,
                   const f64* B, const int ldb, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact,
                   const f64* ferr, const f64* berr, f64* reslts);
extern void dtrt06(const f64 rcond, const f64 rcondc,
                   const char* uplo, const char* diag, const int n,
                   const f64* A, const int lda, f64* work, f64* rat);

/* Matrix generation */
extern void dlattr(const int imat, const char* uplo, const char* trans, char* diag,
                   const int n, f64* A, const int lda,
                   f64* B, f64* work, int* info, uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f64* A, const int lda,
                   f64* XACT, const int ldxact, f64* B, const int ldb,
                   int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern f64 dlamch(const char* cmach);
extern f64 dlantr(const char* norm, const char* uplo, const char* diag,
                     const int m, const int n, const f64* A, const int lda,
                     f64* work);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* A;      /* Original matrix (NMAX x NMAX) */
    f64* AINV;   /* Inverse matrix (NMAX x NMAX) */
    f64* B;      /* Right-hand side (NMAX x NSMAX) */
    f64* X;      /* Solution (NMAX x NSMAX) */
    f64* XACT;   /* Exact solution (NMAX x NSMAX) */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace */
    f64* FERR;   /* Forward error bounds (NSMAX) */
    f64* BERR;   /* Backward error bounds (NSMAX) */
    f64* CNORM;  /* Column norms (NMAX) */
    int* IWORK;     /* Integer workspace */
} dchktr_workspace_t;

static dchktr_workspace_t* g_workspace = NULL;

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
} latrs_params_t;

/* Maximum number of tests */
#define MAX_STANDARD_TESTS (NN * NTYPE1 * 2)      /* 7 * 10 * 2 = 140 */
#define MAX_LATRS_TESTS    (NN * 8 * 2)           /* 7 * 8 * 2 = 112 */
#define MAX_TESTS          (MAX_STANDARD_TESTS + MAX_LATRS_TESTS)

static standard_params_t g_standard_params[MAX_STANDARD_TESTS];
static latrs_params_t g_latrs_params[MAX_LATRS_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static const char* UPLOS[] = {"U", "L"};

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchktr_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * NMAX;

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));
    g_workspace->RWORK = malloc(NMAX * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
    g_workspace->CNORM = malloc(NMAX * sizeof(f64));
    g_workspace->IWORK = malloc(NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK ||
        !g_workspace->FERR || !g_workspace->BERR ||
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
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run tests for non-pathological matrices (IMAT 1-10).
 * Tests DTRTRI, DTRTRS, DTRCON, DTRRFS.
 */
static void test_standard(void** state)
{
    standard_params_t* p = *state;
    dchktr_workspace_t* ws = g_workspace;

    int n = p->n;
    int imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f64 result[NTESTS];
    char diag;
    int info, lda;
    f64 rcondo, rcondi, rcond, rcondc, anorm, ainvnm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10);
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    lda = (n > 1) ? n : 1;

    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Generate triangular test matrix */
    dlattr(imat, uplo, "N", &diag, n, ws->A, lda, ws->X, ws->WORK, &info, rng_state);
    assert_info_success(info);

    int idiag = (diag == 'N' || diag == 'n') ? 1 : 2;

    /* Loop over block sizes (only test 1 uses multiple block sizes) */
    for (int inb = 0; inb < (int)NNB; inb++) {
        /* TEST 1: Form the inverse of A */
        dlacpy(uplo, n, n, ws->A, lda, ws->AINV, lda);
        dtrtri(uplo, &diag, n, ws->AINV, lda, &info);

        /* Compute the infinity-norm condition number of A */
        anorm = dlantr("I", uplo, &diag, n, n, ws->A, lda, ws->RWORK);
        ainvnm = dlantr("I", uplo, &diag, n, n, ws->AINV, lda, ws->RWORK);
        if (anorm <= ZERO || ainvnm <= ZERO) {
            rcondi = ONE;
        } else {
            rcondi = (ONE / anorm) / ainvnm;
        }

        if (info == 0) {
            dtrt01(uplo, &diag, n, ws->A, lda, ws->AINV, lda,
                   &rcondo, ws->RWORK, &result[0]);

            if (result[0] >= THRESH) {
                print_message("TEST 1 failed: n=%d, imat=%d, uplo=%s, diag=%c, nb=%d, resid=%.3e\n",
                             n, imat, uplo, diag, NBVAL[inb], result[0]);
            }
            assert_residual_ok(result[0]);
        }

        /* Skip remaining tests if not first block size */
        if (inb != 0) continue;

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
                dlarhs("DTR", "N", uplo, trans, n, n, 0, idiag, nrhs,
                       ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

                dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
                dtrtrs(uplo, trans, &diag, n, nrhs, ws->A, lda, ws->X, lda, &info);

                /* Skip tests 2-6 for singular matrices (IMAT 5, 6) or if DTRTRS failed */
                if (info != 0 || imat == 5 || imat == 6) {
                    continue;
                }

                dtrt02(uplo, trans, &diag, n, nrhs, ws->A, lda,
                       ws->X, lda, ws->B, lda, ws->WORK, &result[1]);

                /* TEST 3: Check solution from generated exact solution */
                dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);

                /* TESTS 4, 5, 6: Use iterative refinement */
                dtrrfs(uplo, trans, &diag, n, nrhs, ws->A, lda,
                       ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
                       ws->WORK, ws->IWORK, &info);

                dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
                dtrt05(uplo, trans, &diag, n, nrhs, ws->A, lda,
                       ws->B, lda, ws->X, lda, ws->XACT, lda,
                       ws->FERR, ws->BERR, &result[4]);

                /* Check results */
                for (int k = 1; k < 6; k++) {
                    if (result[k] >= THRESH) {
                        print_message("TEST %d failed: n=%d, imat=%d, uplo=%s, trans=%s, nrhs=%d, resid=%.3e\n",
                                     k + 1, n, imat, uplo, trans, nrhs, result[k]);
                    }
                    assert_residual_ok(result[k]);
                }
            }
        }

        /* TEST 7: Get estimate of RCOND = 1/CNDNUM */
        if (imat != 5 && imat != 6) {
            for (int itran = 0; itran < 2; itran++) {
                const char* norm;
                if (itran == 0) {
                    norm = "O";
                    rcondc = rcondo;
                } else {
                    norm = "I";
                    rcondc = rcondi;
                }

                dtrcon(norm, uplo, &diag, n, ws->A, lda, &rcond,
                       ws->WORK, ws->IWORK, &info);
                if (info != 0) {
                    print_message("DTRCON failed: info=%d\n", info);
                }

                dtrt06(rcond, rcondc, uplo, &diag, n, ws->A, lda, ws->RWORK, &result[6]);

                if (result[6] >= THRESH) {
                    print_message("TEST 7 failed: n=%d, imat=%d, uplo=%s, norm=%s, resid=%.3e\n",
                                 n, imat, uplo, norm, result[6]);
                }
                assert_residual_ok(result[6]);
            }
        }
    }
}

/**
 * Run tests for pathological matrices (IMAT 11-18).
 * Tests DLATRS and DLATRS3 with scaling.
 */
static void test_latrs(void** state)
{
    latrs_params_t* p = *state;
    dchktr_workspace_t* ws = g_workspace;

    int n = p->n;
    int imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f64 result[NTESTS];
    char diag;
    int info, lda;
    f64 scale;
    f64 scale3[2];
    f64 res;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10);
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    f64 bignum = dlamch("O") / dlamch("P");

    lda = (n > 1) ? n : 1;
    int ldb = (n > 1) ? n : 1;

    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Loop over TRANS = 'N', 'T', 'C' */
    const char* transs[] = {"N", "T", "C"};
    for (int itran = 0; itran < 3; itran++) {
        const char* trans = transs[itran];

        /* Generate triangular test matrix */
        dlattr(imat, uplo, trans, &diag, n, ws->A, lda, ws->X, ws->WORK, &info, rng_state);
        if (info != 0) {
            print_message("DLATTR failed: info=%d, imat=%d\n", info, imat);
        }

        /* TEST 8: Solve op(A)*x = b with NORMIN='N' */
        cblas_dcopy(n, ws->X, 1, ws->B, 1);
        char normin = 'N';
        dlatrs(uplo, trans, &diag, &normin, n, ws->A, lda, ws->B, &scale,
               ws->CNORM, &info);
        if (info != 0) {
            print_message("DLATRS failed: info=%d, imat=%d, uplo=%s, trans=%s\n",
                         info, imat, uplo, trans);
        }

        dtrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale,
               ws->CNORM, ONE, ws->B, lda, ws->X, lda, ws->WORK, &result[7]);

        /* TEST 9: Solve again with NORMIN='Y' */
        cblas_dcopy(n, ws->X, 1, &ws->B[n], 1);
        normin = 'Y';
        dlatrs(uplo, trans, &diag, &normin, n, ws->A, lda, &ws->B[n], &scale,
               ws->CNORM, &info);
        if (info != 0) {
            print_message("DLATRS (NORMIN=Y) failed: info=%d\n", info);
        }

        dtrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale,
               ws->CNORM, ONE, &ws->B[n], lda, ws->X, lda, ws->WORK, &result[8]);

        /* TEST 10: Solve op(A)*X = B with DLATRS3 (blocked, 2 RHS) */
        cblas_dcopy(n, ws->X, 1, ws->B, 1);
        cblas_dcopy(n, ws->X, 1, &ws->B[n], 1);
        cblas_dscal(n, bignum, &ws->B[n], 1);

        normin = 'N';
        int lwork_latrs3 = NMAX;
        dlatrs3(uplo, trans, &diag, &normin, n, 2, ws->A, lda,
                ws->B, ldb, scale3, ws->CNORM, ws->WORK, lwork_latrs3, &info);
        if (info != 0) {
            print_message("DLATRS3 failed: info=%d, imat=%d, uplo=%s, trans=%s\n",
                         info, imat, uplo, trans);
        }

        /* Verify first column */
        dtrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale3[0],
               ws->CNORM, ONE, ws->B, lda, ws->X, lda, ws->WORK, &result[9]);

        /* Verify second column */
        cblas_dscal(n, bignum, ws->X, 1);
        dtrt03(uplo, trans, &diag, n, 1, ws->A, lda, scale3[1],
               ws->CNORM, ONE, &ws->B[n], lda, ws->X, lda, ws->WORK, &res);
        if (res > result[9]) {
            result[9] = res;
        }

        /* Check results */
        if (result[7] >= THRESH) {
            print_message("TEST 8 failed: n=%d, imat=%d, uplo=%s, trans=%s, resid=%.3e\n",
                         n, imat, uplo, trans, result[7]);
        }
        if (result[8] >= THRESH) {
            print_message("TEST 9 failed: n=%d, imat=%d, uplo=%s, trans=%s, resid=%.3e\n",
                         n, imat, uplo, trans, result[8]);
        }
        if (result[9] >= THRESH) {
            print_message("TEST 10 failed: n=%d, imat=%d, uplo=%s, trans=%s, resid=%.3e\n",
                         n, imat, uplo, trans, result[9]);
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
    int standard_idx = 0;
    int latrs_idx = 0;

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

    /* Pathological tests (DLATRS): N x IMAT(11-18) x UPLO(U,L) */
    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];

        for (int imat = NTYPE1 + 1; imat <= NTYPES; imat++) {
            for (int iuplo = 0; iuplo < 2; iuplo++) {
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

    return _cmocka_run_group_tests("dchktr", g_tests, (size_t)g_num_tests,
                                   group_setup, group_teardown);
}
