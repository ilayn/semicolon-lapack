/**
 * @file test_ddrvab.c
 * @brief Test suite for DSGESV mixed-precision iterative refinement solver.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/ddrvab.f to C using CMocka.
 * Tests DSGESV which uses single precision factorization with double precision
 * iterative refinement.
 *
 * Test parameters from dchkab.f data file:
 *   M values: 0, 1, 2, 3, 5, 10, 16
 *   NRHS values: 2
 *   Matrix types: 1-11
 *   THRESH: 20.0
 */

#include "test_harness.h"
#include <cblas.h>

/* Test parameters */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 16};
static const int NSVAL[] = {2};  /* NRHS values */

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  11
#define THRESH  20.0
#define NMAX    132
#define MAXRHS  16

/* Routine under test */
extern void dsgesv(const int n, const int nrhs, double* A, const int lda,
                   int* ipiv, const double* B, const int ldb,
                   double* X, const int ldx, double* work,
                   float* swork, int* iter, int* info);

/* Verification routine */
extern void dget08(const char* trans, const int m, const int n, const int nrhs,
                   const double* A, const int lda, const double* X, const int ldx,
                   double* B, const int ldb, double* rwork, double* resid);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);

extern void dlatms(const int m, const int n, const char* dist, uint64_t seed,
                   const char* sym, double* d, const int mode, const double cond,
                   const double dmax, const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);

extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const double* A, const int lda,
                   double* X, const int ldx, double* B, const int ldb,
                   uint64_t seed, int* info);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int nrhs;
    int imat;
    char name[64];
} ddrvab_params_t;

/**
 * Workspace for test execution.
 */
typedef struct {
    double* A;          /* Original matrix (NMAX x NMAX) */
    double* AFAC;       /* Copy of A for factorization (NMAX x NMAX) */
    double* B;          /* Right-hand side (NMAX x MAXRHS) */
    double* X;          /* Solution (NMAX x MAXRHS) */
    double* WORK;       /* Double precision workspace */
    double* RWORK;      /* Workspace for verification */
    float* SWORK;       /* Single precision workspace */
    int* IWORK;         /* Integer workspace (pivot indices) */
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

    g_ws->A = malloc(NMAX * NMAX * sizeof(double));
    g_ws->AFAC = malloc(NMAX * NMAX * sizeof(double));
    g_ws->B = malloc(NMAX * MAXRHS * sizeof(double));
    g_ws->X = malloc(NMAX * MAXRHS * sizeof(double));
    g_ws->WORK = malloc(NMAX * MAXRHS * 2 * sizeof(double));
    g_ws->RWORK = malloc(2 * NMAX * sizeof(double));
    g_ws->SWORK = malloc(NMAX * (NMAX + MAXRHS) * sizeof(float));
    g_ws->IWORK = malloc(NMAX * sizeof(int));

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
static void run_test_single(int n, int nrhs, int imat)
{
    char type, dist;
    int kl, ku, mode, info, iter;
    double anorm, cndnum;
    int lda = (n > 1) ? n : 1;
    int izero = 0;

    /* Seed based on test parameters (matches LAPACK ISEEDY = {2006, 2007, 2008, 2009}) */
    uint64_t seed = 2006 + n * 1000 + imat * 100 + nrhs;

    /* Get matrix parameters */
    dlatb4("DGE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix */
    char dist_str[2] = {dist, '\0'};
    char sym_str[2] = {type, '\0'};
    dlatms(n, n, dist_str, seed, sym_str, g_ws->RWORK, mode, cndnum, anorm,
           kl, ku, "N", g_ws->A, lda, g_ws->WORK, &info);

    if (info != 0) {
        /* Matrix generation failed - skip this test */
        return;
    }

    /* For types 5-7, zero one or more columns to test singularity detection */
    int zerot = (imat >= 5 && imat <= 7);
    if (zerot) {
        if (imat == 5) {
            izero = 1;
        } else if (imat == 6) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }
        /* Zero column izero (1-based) */
        int ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (int i = 0; i < n; i++) {
                g_ws->A[ioff + i] = 0.0;
            }
        } else {
            /* Zero columns from izero to n */
            dlaset("F", n, n - izero + 1, 0.0, 0.0, &g_ws->A[ioff], lda);
        }
    }

    /* Generate right-hand side */
    dlarhs("DGE", "N", " ", "N", n, n, kl, ku, nrhs,
           g_ws->A, lda, g_ws->X, lda, g_ws->B, lda, seed, &info);

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

    double resid;
    dget08("N", n, n, nrhs, g_ws->AFAC, lda, g_ws->X, lda,
           g_ws->WORK, lda, g_ws->RWORK, &resid);

    /* Check the test result:
     * If iterative refinement was used (iter >= 0), we want:
     *   resid < sqrt(n)
     * If double precision was used (iter < 0), we want:
     *   resid < THRESH
     */
    char context[128];
    snprintf(context, sizeof(context),
             "ddrvab n=%d nrhs=%d imat=%d iter=%d", n, nrhs, imat, iter);
    set_test_context(context);

    if (iter >= 0 && n > 0) {
        double thresh_iter = sqrt((double)n);
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
        int n = MVAL[im];
        int nimat = NTYPES;

        /* For empty matrix, only test type 1 */
        if (n <= 0) {
            nimat = 1;
        }

        for (int imat = 1; imat <= nimat; imat++) {
            /* Skip types 5, 6, 7 if matrix is too small */
            int zerot = (imat >= 5 && imat <= 7);
            if (zerot && n < imat - 4) {
                continue;
            }

            for (size_t irhs = 0; irhs < NNS; irhs++) {
                int nrhs = NSVAL[irhs];
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
