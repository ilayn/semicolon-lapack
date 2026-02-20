/**
 * @file test_dchktb.c
 * @brief Comprehensive test suite for triangular banded matrix routines (DTB).
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchktb.f to C using CMocka.
 * Tests DTBTRS, DTBRFS, DTBCON, and DLATBS.
 *
 * Each (n, kd, imat, uplo) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchktb.f:
 *   IMAT 1-9: Non-pathological triangular banded matrices
 *     TEST 1: DTBTRS - | b - op(A)*x | / (|op(A)| * |x| * eps)
 *     TEST 2: Compare solution to exact (DGET04)
 *     TESTS 3-5: DTBRFS iterative refinement + error bounds
 *     TEST 6: DTBCON condition number estimate
 *
 *   IMAT 10-17: Pathological matrices for DLATBS
 *     TEST 7: DLATBS with NORMIN='N'
 *     TEST 8: DLATBS with NORMIN='Y'
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-9 (standard), 10-17 (pathological)
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
#define NTYPE1  9       /* Number of non-pathological types */
#define NTYPES  17      /* Total matrix types (including pathological) */
#define NTESTS  8       /* Number of tests per matrix */
#define THRESH  30.0
#define NMAX    50      /* Maximum matrix dimension */
#define NSMAX   15      /* Maximum NRHS */

/* Routines under test */
extern void dtbtrs(const char* uplo, const char* trans, const char* diag,
                   const int n, const int kd, const int nrhs,
                   const f64* AB, const int ldab,
                   f64* B, const int ldb, int* info);
extern void dtbrfs(const char* uplo, const char* trans, const char* diag,
                   const int n, const int kd, const int nrhs,
                   const f64* AB, const int ldab,
                   const f64* B, const int ldb,
                   const f64* X, const int ldx,
                   f64* ferr, f64* berr,
                   f64* work, int* iwork, int* info);
extern void dtbcon(const char* norm, const char* uplo, const char* diag,
                   const int n, const int kd, const f64* AB, const int ldab,
                   f64* rcond, f64* work, int* iwork, int* info);
extern void dlatbs(const char* uplo, const char* trans, const char* diag,
                   const char* normin, const int n, const int kd,
                   const f64* AB, const int ldab,
                   f64* X, f64* scale, f64* cnorm, int* info);

/* Verification routines */
extern void dtbt02(const char* uplo, const char* trans, const char* diag,
                   const int n, const int kd, const int nrhs,
                   const f64* AB, const int ldab,
                   const f64* X, const int ldx,
                   const f64* B, const int ldb,
                   f64* work, f64* resid);
extern void dtbt03(const char* uplo, const char* trans, const char* diag,
                   const int n, const int kd, const int nrhs,
                   const f64* AB, const int ldab,
                   const f64 scale, const f64* cnorm, const f64 tscal,
                   const f64* X, const int ldx,
                   const f64* B, const int ldb,
                   f64* work, f64* resid);
extern void dtbt05(const char* uplo, const char* trans, const char* diag,
                   const int n, const int kd, const int nrhs,
                   const f64* AB, const int ldab,
                   const f64* B, const int ldb,
                   const f64* X, const int ldx,
                   const f64* XACT, const int ldxact,
                   const f64* ferr, const f64* berr,
                   f64* reslts);
extern void dtbt06(const f64 rcond, const f64 rcondc,
                   const char* uplo, const char* diag, const int n, const int kd,
                   const f64* AB, const int ldab, f64* work, f64* rat);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);

/* Matrix generation */
extern void dlattb(const int imat, const char* uplo, const char* trans,
                   char* diag, const int n, const int kd,
                   f64* AB, const int ldab, f64* B, f64* work,
                   int* info, uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f64* A, const int lda,
                   f64* XACT, const int ldxact, f64* B, const int ldb,
                   int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern f64 dlamch(const char* cmach);
extern f64 dlantb(const char* norm, const char* uplo, const char* diag,
                     const int n, const int kd, const f64* AB, const int ldab,
                     f64* work);
extern f64 dlantr(const char* norm, const char* uplo, const char* diag,
                     const int m, const int n, const f64* A, const int lda,
                     f64* work);

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f64* AB;     /* Banded matrix: (NMAX+1) * NMAX */
    f64* AINV;   /* Full inverse: NMAX * NMAX */
    f64* B;      /* Right-hand side: NMAX * NSMAX */
    f64* X;      /* Solution: NMAX * NSMAX */
    f64* XACT;   /* Exact solution: NMAX * NSMAX */
    f64* WORK;   /* General workspace: NMAX * max(3, NSMAX) */
    f64* RWORK;  /* Real workspace: max(NMAX, 2*NSMAX) */
    int* IWORK;     /* Integer workspace: NMAX */
} dchktb_workspace_t;

static dchktb_workspace_t* g_workspace = NULL;

/* =========================================================================
 * Parameterized test infrastructure
 * ========================================================================= */

/* Test parameters for standard tests (IMAT 1-9) */
typedef struct {
    int n;
    int kd;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    char name[80];
} standard_params_t;

/* Test parameters for pathological tests (IMAT 10-17) */
typedef struct {
    int n;
    int kd;
    int imat;
    int iuplo;      /* 0='U', 1='L' */
    char name[80];
} latbs_params_t;

/* Maximum number of tests: NN * 4(kd) * (NTYPE1 + NTYPES-NTYPE1) * 2(uplo) */
#define MAX_STANDARD_TESTS (NN * 4 * NTYPE1 * 2)
#define MAX_LATBS_TESTS    (NN * 4 * (NTYPES - NTYPE1) * 2)
#define MAX_TESTS          (MAX_STANDARD_TESTS + MAX_LATBS_TESTS)

static standard_params_t g_standard_params[MAX_STANDARD_TESTS];
static latbs_params_t g_latbs_params[MAX_LATBS_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static const char* UPLOS[] = {"U", "L"};

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchktb_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * (3 > NSMAX ? 3 : NSMAX);
    int lrwork = (NMAX > 2 * NSMAX) ? NMAX : 2 * NSMAX;

    g_workspace->AB = malloc(((NMAX + 1) * NMAX + 1) * sizeof(f64));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->WORK = malloc(lwork * sizeof(f64));
    g_workspace->RWORK = malloc(lrwork * sizeof(f64));
    g_workspace->IWORK = malloc(NMAX * sizeof(int));

    if (!g_workspace->AB || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->IWORK) {
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
        free(g_workspace->AB);
        free(g_workspace->AINV);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run tests for non-pathological matrices (IMAT 1-9).
 * Tests DTBTRS, DTBRFS, DTBCON.
 */
static void test_standard(void** state)
{
    standard_params_t* p = *state;
    dchktb_workspace_t* ws = g_workspace;

    int n = p->n;
    int kd = p->kd;
    int imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f64 result[NTESTS];
    char diag;
    int info, lda, ldab;
    f64 rcondo, rcondi, rcond, rcondc, anorm, ainvnm;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10 + kd);
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    lda = (n > 1) ? n : 1;
    ldab = kd + 1;

    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Generate triangular banded test matrix */
    dlattb(imat, uplo, "N", &diag, n, kd, ws->AB, ldab,
           ws->XACT, ws->WORK, &info, rng_state);
    assert_info_success(info);

    int idiag = (diag == 'N' || diag == 'n') ? 1 : 2;
    char diag_str[2] = {diag, '\0'};

    /*
     * Form the inverse of A so we can get a good estimate of
     * RCONDC = 1/(norm(A) * norm(inv(A))).
     */
    dlaset("F", n, n, ZERO, ONE, ws->AINV, lda);
    if (uplo[0] == 'U' || uplo[0] == 'u') {
        for (int j = 0; j < n; j++) {
            cblas_dtbsv(CblasColMajor, CblasUpper, CblasNoTrans,
                        (diag == 'N' || diag == 'n') ? CblasNonUnit : CblasUnit,
                        j + 1, kd, ws->AB, ldab,
                        &ws->AINV[j * lda], 1);
        }
    } else {
        for (int j = 0; j < n; j++) {
            cblas_dtbsv(CblasColMajor, CblasLower, CblasNoTrans,
                        (diag == 'N' || diag == 'n') ? CblasNonUnit : CblasUnit,
                        n - j, kd, &ws->AB[j * ldab], ldab,
                        &ws->AINV[j * lda + j], 1);
        }
    }

    /* Compute the 1-norm condition number of A */
    anorm = dlantb("1", uplo, diag_str, n, kd, ws->AB, ldab, ws->RWORK);
    ainvnm = dlantr("1", uplo, diag_str, n, n, ws->AINV, lda, ws->RWORK);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        rcondo = ONE;
    } else {
        rcondo = (ONE / anorm) / ainvnm;
    }

    /* Compute the infinity-norm condition number of A */
    anorm = dlantb("I", uplo, diag_str, n, kd, ws->AB, ldab, ws->RWORK);
    ainvnm = dlantr("I", uplo, diag_str, n, n, ws->AINV, lda, ws->RWORK);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        rcondi = ONE;
    } else {
        rcondi = (ONE / anorm) / ainvnm;
    }

    /* Loop over NRHS values */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];
        char xtype_str[2] = "N";

        /* Loop over TRANS = 'N', 'T', 'C' */
        const char* transs[] = {"N", "T", "C"};
        for (int itran = 0; itran < 3; itran++) {
            const char* trans = transs[itran];
            if (itran == 0) {
                rcondc = rcondo;
            } else {
                rcondc = rcondi;
            }

            char ctx[128];

            /*
             * TEST 1: Solve and compute residual for op(A)*x = b.
             */
            snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%s trans=%s diag=%c imat=%d nrhs=%d TEST 1",
                     n, kd, uplo, trans, diag, imat, nrhs);
            set_test_context(ctx);

            dlarhs("DTB", xtype_str, uplo, trans, n, n, kd, idiag, nrhs,
                   ws->AB, ldab, ws->XACT, lda, ws->B, lda, &info, rng_state);
            xtype_str[0] = 'C';

            dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);

            dtbtrs(uplo, trans, diag_str, n, kd, nrhs, ws->AB, ldab,
                   ws->X, lda, &info);
            assert_info_success(info);

            dtbt02(uplo, trans, diag_str, n, kd, nrhs, ws->AB, ldab,
                   ws->X, lda, ws->B, lda, ws->WORK, &result[0]);
            assert_residual_below(result[0], THRESH);

            /*
             * TEST 2: Check solution from generated exact solution.
             */
            snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%s trans=%s diag=%c imat=%d nrhs=%d TEST 2",
                     n, kd, uplo, trans, diag, imat, nrhs);
            set_test_context(ctx);

            dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[1]);
            assert_residual_below(result[1], THRESH);

            /*
             * TESTS 3, 4, and 5: Iterative refinement.
             */
            snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%s trans=%s diag=%c imat=%d nrhs=%d TESTS 3-5",
                     n, kd, uplo, trans, diag, imat, nrhs);
            set_test_context(ctx);

            dtbrfs(uplo, trans, diag_str, n, kd, nrhs, ws->AB, ldab,
                   ws->B, lda, ws->X, lda,
                   ws->RWORK, &ws->RWORK[nrhs],
                   ws->WORK, ws->IWORK, &info);
            assert_info_success(info);

            dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
            dtbt05(uplo, trans, diag_str, n, kd, nrhs, ws->AB, ldab,
                   ws->B, lda, ws->X, lda, ws->XACT, lda,
                   ws->RWORK, &ws->RWORK[nrhs], &result[3]);

            assert_residual_below(result[2], THRESH);
            assert_residual_below(result[3], THRESH);
            assert_residual_below(result[4], THRESH);
        }
    }

    /*
     * TEST 6: Get an estimate of RCOND = 1/CNDNUM.
     */
    for (int itran = 0; itran < 2; itran++) {
        const char* norm;
        if (itran == 0) {
            norm = "O";
            rcondc = rcondo;
        } else {
            norm = "I";
            rcondc = rcondi;
        }

        char ctx[128];
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%s diag=%c imat=%d norm=%s TEST 6",
                 n, kd, uplo, diag, imat, norm);
        set_test_context(ctx);

        dtbcon(norm, uplo, diag_str, n, kd, ws->AB, ldab,
               &rcond, ws->WORK, ws->IWORK, &info);
        assert_info_success(info);

        dtbt06(rcond, rcondc, uplo, diag_str, n, kd, ws->AB, ldab,
               ws->RWORK, &result[5]);
        assert_residual_below(result[5], THRESH);
    }

    clear_test_context();
}

/**
 * Run tests for pathological matrices (IMAT 10-17).
 * Tests DLATBS.
 */
static void test_latbs(void** state)
{
    latbs_params_t* p = *state;
    dchktb_workspace_t* ws = g_workspace;

    int n = p->n;
    int kd = p->kd;
    int imat = p->imat;
    const char* uplo = UPLOS[p->iuplo];

    f64 result[NTESTS];
    char diag;
    int info, lda, ldab;
    f64 scale;
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + imat * 1000 + n * 100 + p->iuplo * 10 + kd);
    const f64 ONE = 1.0;
    const f64 ZERO = 0.0;

    lda = (n > 1) ? n : 1;
    ldab = kd + 1;

    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Loop over TRANS = 'N', 'T', 'C' */
    const char* transs[] = {"N", "T", "C"};
    for (int itran = 0; itran < 3; itran++) {
        const char* trans = transs[itran];

        /* Generate triangular banded test matrix */
        dlattb(imat, uplo, trans, &diag, n, kd, ws->AB, ldab,
               ws->XACT, ws->WORK, &info, rng_state);
        char diag_str[2] = {diag, '\0'};

        char ctx[128];

        /*
         * TEST 7: Solve the system op(A)*x = b with NORMIN='N'.
         */
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%s trans=%s diag=%c imat=%d TEST 7",
                 n, kd, uplo, trans, diag, imat);
        set_test_context(ctx);

        cblas_dcopy(n, ws->XACT, 1, ws->B, 1);
        dlatbs(uplo, trans, diag_str, "N", n, kd, ws->AB, ldab,
               ws->B, &scale, ws->RWORK, &info);

        dtbt03(uplo, trans, diag_str, n, kd, 1, ws->AB, ldab,
               scale, ws->RWORK, ONE, ws->B, lda, ws->XACT, lda,
               ws->WORK, &result[6]);
        assert_residual_below(result[6], THRESH);

        /*
         * TEST 8: Solve op(A)*x = b again with NORMIN='Y'.
         */
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%s trans=%s diag=%c imat=%d TEST 8",
                 n, kd, uplo, trans, diag, imat);
        set_test_context(ctx);

        cblas_dcopy(n, ws->XACT, 1, ws->B, 1);
        dlatbs(uplo, trans, diag_str, "Y", n, kd, ws->AB, ldab,
               ws->B, &scale, ws->RWORK, &info);

        dtbt03(uplo, trans, diag_str, n, kd, 1, ws->AB, ldab,
               scale, ws->RWORK, ONE, ws->B, lda, ws->XACT, lda,
               ws->WORK, &result[7]);
        assert_residual_below(result[7], THRESH);
    }

    clear_test_context();
}

/**
 * Build test array with all parameter combinations.
 */
static void build_test_array(void)
{
    int standard_idx = 0;
    int latbs_idx = 0;

    for (int in = 0; in < (int)NN; in++) {
        int n = NVAL[in];

        int nimat = NTYPE1;
        int nimat2 = NTYPES;
        if (n <= 0) {
            nimat = 1;
            nimat2 = NTYPE1 + 1;
        }

        int nk = (n + 1 < 4) ? n + 1 : 4;
        for (int ik = 0; ik < nk; ik++) {
            int kd;
            if (ik == 0)       kd = 0;
            else if (ik == 1)  kd = (n > 0) ? n : 0;
            else if (ik == 2)  kd = (3 * n - 1) / 4;
            else               kd = (n + 1) / 4;

            /* Standard tests: IMAT 1-NTYPE1 */
            for (int imat = 1; imat <= nimat; imat++) {
                for (int iuplo = 0; iuplo < 2; iuplo++) {
                    standard_params_t* p = &g_standard_params[standard_idx];
                    p->n = n;
                    p->kd = kd;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    snprintf(p->name, sizeof(p->name), "n%d_kd%d_type%d_%s",
                             n, kd, imat, UPLOS[iuplo]);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_standard;
                    g_tests[g_num_tests].initial_state = p;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_num_tests++;
                    standard_idx++;
                }
            }

            /* Pathological tests (DLATBS): IMAT NTYPE1+1 to NTYPES */
            for (int imat = NTYPE1 + 1; imat <= nimat2; imat++) {
                for (int iuplo = 0; iuplo < 2; iuplo++) {
                    latbs_params_t* p = &g_latbs_params[latbs_idx];
                    p->n = n;
                    p->kd = kd;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    snprintf(p->name, sizeof(p->name), "latbs_n%d_kd%d_type%d_%s",
                             n, kd, imat, UPLOS[iuplo]);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_latbs;
                    g_tests[g_num_tests].initial_state = p;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_num_tests++;
                    latbs_idx++;
                }
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

    return _cmocka_run_group_tests("dchktb", g_tests, (size_t)g_num_tests,
                                   group_setup, group_teardown);
}
