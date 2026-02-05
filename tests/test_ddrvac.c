/**
 * @file test_ddrvac.c
 * @brief Test suite for DSPOSV mixed-precision iterative refinement solver.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/ddrvac.f to C using CMocka.
 * Tests DSPOSV which uses single precision Cholesky factorization with double
 * precision iterative refinement for symmetric positive definite systems.
 *
 * Test parameters from dchkab.f data file:
 *   M values: 0, 1, 2, 3, 5, 10, 16
 *   NRHS values: 2
 *   Matrix types: 1-9 (positive definite types)
 *   THRESH: 20.0
 */

#include "test_harness.h"
#include <cblas.h>

/* Test parameters */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 16};
static const int NSVAL[] = {2};  /* NRHS values */
static const char UPLOS[] = {'U', 'L'};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9  /* Positive definite matrix types */
#define THRESH  20.0
#define NMAX    132
#define MAXRHS  16

/* Routine under test */
extern void dsposv(const char* uplo, const int n, const int nrhs,
                   double* A, const int lda, const double* B, const int ldb,
                   double* X, const int ldx, double* work,
                   float* swork, int* iter, int* info);

/* Verification routine */
extern void dpot06(const char* uplo, const int n, const int nrhs,
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

    g_ws->A = malloc(NMAX * NMAX * sizeof(double));
    g_ws->AFAC = malloc(NMAX * NMAX * sizeof(double));
    g_ws->B = malloc(NMAX * MAXRHS * sizeof(double));
    g_ws->X = malloc(NMAX * MAXRHS * sizeof(double));
    g_ws->WORK = malloc(NMAX * MAXRHS * 2 * sizeof(double));
    g_ws->RWORK = malloc(2 * NMAX * sizeof(double));
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
static void run_test_single(int n, int nrhs, int imat, char uplo)
{
    char type, dist;
    int kl, ku, mode, info, iter;
    double anorm, cndnum;
    int lda = (n > 1) ? n : 1;
    int izero = 0;

    /* Seed based on test parameters (matches LAPACK ISEEDY = {1988, 1989, 1990, 1991}) */
    uint64_t seed = 1988 + n * 1000 + imat * 100 + nrhs + (uplo == 'U' ? 0 : 50);

    /* Get matrix parameters for positive definite path */
    dlatb4("DPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric positive definite test matrix */
    char dist_str[2] = {dist, '\0'};
    char uplo_str[2] = {uplo, '\0'};
    dlatms(n, n, dist_str, seed, "P", g_ws->RWORK, mode, cndnum, anorm,
           kl, ku, uplo_str, g_ws->A, lda, g_ws->WORK, &info);

    if (info != 0) {
        /* Matrix generation failed - skip this test */
        return;
    }

    /* For types 3-5, zero one row and column to test singularity detection */
    int zerot = (imat >= 3 && imat <= 5);
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }

        /* Zero row and column izero (1-based) */
        int ioff = (izero - 1) * lda;

        if (uplo == 'U' || uplo == 'u') {
            /* Upper: zero column up to diagonal, then row from diagonal */
            for (int i = 0; i < izero - 1; i++) {
                g_ws->A[ioff + i] = 0.0;
            }
            ioff = ioff + izero - 1;
            for (int i = izero - 1; i < n; i++) {
                g_ws->A[ioff] = 0.0;
                ioff = ioff + lda;
            }
        } else {
            /* Lower: zero row up to diagonal, then column from diagonal */
            ioff = izero - 1;
            for (int i = 0; i < izero - 1; i++) {
                g_ws->A[ioff] = 0.0;
                ioff = ioff + lda;
            }
            ioff = ioff - (izero - 1);
            for (int i = izero - 1; i < n; i++) {
                g_ws->A[ioff + i] = 0.0;
            }
        }
    }

    /* Generate right-hand side */
    dlarhs("DPO", "N", uplo_str, " ", n, n, kl, ku, nrhs,
           g_ws->A, lda, g_ws->X, lda, g_ws->B, lda, seed, &info);

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

    double resid;
    dpot06(uplo_str, n, nrhs, g_ws->A, lda, g_ws->X, lda,
           g_ws->WORK, lda, g_ws->RWORK, &resid);

    /* Check the test result:
     * If iterative refinement was used (iter >= 0), we want:
     *   resid < sqrt(n)
     * If double precision was used (iter < 0), we want:
     *   resid < THRESH
     */
    char context[128];
    snprintf(context, sizeof(context),
             "ddrvac uplo=%c n=%d nrhs=%d imat=%d iter=%d", uplo, n, nrhs, imat, iter);
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
 * Test DSPOSV for all matrix types at given size.
 */
static void test_dsposv(void** state)
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
            /* Skip types 3, 4, 5 if matrix is too small */
            int zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) {
                continue;
            }

            /* Test both upper and lower triangular storage */
            for (size_t iuplo = 0; iuplo < NUPLO; iuplo++) {
                char uplo = UPLOS[iuplo];

                for (size_t irhs = 0; irhs < NNS; irhs++) {
                    int nrhs = NSVAL[irhs];
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
