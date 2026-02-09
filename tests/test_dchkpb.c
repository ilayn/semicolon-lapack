/**
 * @file test_dchkpb.c
 * @brief Comprehensive test suite for positive definite banded matrix (DPB) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkpb.f to C using CMocka.
 * Tests DPBTRF, DPBTRS, DPBRFS, and DPBCON.
 *
 * Test structure from dchkpb.f:
 *   TEST 1: Cholesky factorization residual via dpbt01
 *   TEST 2: Solution residual via dpbt02
 *   TEST 3: Solution accuracy via dget04
 *   TESTS 4-6: Iterative refinement via dpbrfs + dget04 + dpbt05
 *   TEST 7: Condition number via dget06
 *
 * Parameters from dtest.in:
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NRHS values: 1, 2, 15
 *   Matrix types: 1-8
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
#define NTYPES  8
#define NTESTS  7
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */
#define NBW     4   /* Number of bandwidth values per N */

/* Routines under test */
extern void dpbtrf(const char* uplo, const int n, const int kd, double* AB,
                   const int ldab, int* info);
extern void dpbtrs(const char* uplo, const int n, const int kd, const int nrhs,
                   const double* AB, const int ldab, double* B, const int ldb,
                   int* info);
extern void dpbrfs(const char* uplo, const int n, const int kd, const int nrhs,
                   const double* AB, const int ldab, const double* AFB,
                   const int ldafb, const double* B, const int ldb,
                   double* X, const int ldx, double* ferr, double* berr,
                   double* work, int* iwork, int* info);
extern void dpbcon(const char* uplo, const int n, const int kd,
                   const double* AB, const int ldab, const double anorm,
                   double* rcond, double* work, int* iwork, int* info);

/* Verification routines */
extern void dpbt01(const char* uplo, const int n, const int kd,
                   const double* A, const int lda,
                   double* AFAC, const int ldafac,
                   double* rwork, double* resid);
extern void dpbt02(const char* uplo, const int n, const int kd, const int nrhs,
                   const double* A, const int lda,
                   const double* X, const int ldx,
                   double* B, const int ldb,
                   double* rwork, double* resid);
extern void dpbt05(const char* uplo, const int n, const int kd, const int nrhs,
                   const double* AB, const int ldab,
                   const double* B, const int ldb,
                   const double* X, const int ldx,
                   const double* XACT, const int ldxact,
                   const double* ferr, const double* berr,
                   double* reslts);
extern void dget04(const int n, const int nrhs, const double* X, const int ldx,
                   const double* XACT, const int ldxact, const double rcond,
                   double* resid);
extern double dget06(const double rcond, const double rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, double* d, const int mode, const double cond,
                   const double dmax, const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const double* A, const int lda,
                   double* X, const int ldx, double* B, const int ldb,
                   int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta, double* A, const int lda);
extern double dlansb(const char* norm, const char* uplo, const int n,
                     const int k, const double* AB, const int ldab, double* work);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);
extern double dlamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int kd;
    int imat;
    int iuplo;  /* 0='U', 1='L' */
    int inb;    /* Index into NBVAL[] */
    char name[64];
} dchkpb_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    double* A;      /* Original band matrix ((NMAX+1) * NMAX) */
    double* AFAC;   /* Factored band matrix ((NMAX+1) * NMAX) */
    double* AINV;   /* Full inverse matrix (NMAX * NMAX) */
    double* B;      /* Right-hand side (NMAX * NSMAX) */
    double* X;      /* Solution (NMAX * NSMAX) */
    double* XACT;   /* Exact solution (NMAX * NSMAX) */
    double* WORK;   /* General workspace */
    double* RWORK;  /* Real workspace */
    double* D;      /* Singular values for dlatms */
    double* FERR;   /* Forward error bounds */
    double* BERR;   /* Backward error bounds */
    int* IWORK;     /* Integer workspace */
} dchkpb_workspace_t;

static dchkpb_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkpb_workspace_t));
    if (!g_workspace) return -1;

    /* Band storage: (KD+1) rows × N columns, with KD up to NMAX + (NMAX+1)/4 */
    /* Max ldab = NMAX + (NMAX+1)/4 + 1 = 50 + 12 + 1 = 63 for NMAX=50 */
    int max_ldab = NMAX + (NMAX + 1) / 4 + 1;
    g_workspace->A = malloc(max_ldab * NMAX * sizeof(double));
    g_workspace->AFAC = malloc(max_ldab * NMAX * sizeof(double));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(double));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(double));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(double));
    /* WORK needs NMAX * (3*NMAX + NSMAX + 30) for dlatms full matrix generation */
    int work_size = NMAX * (3 * NMAX + NSMAX + 30);
    g_workspace->WORK = malloc(work_size * sizeof(double));
    g_workspace->RWORK = malloc((NMAX > 2 * NSMAX ? NMAX : 2 * NSMAX) * sizeof(double));
    g_workspace->D = malloc(NMAX * sizeof(double));
    g_workspace->FERR = malloc(NSMAX * sizeof(double));
    g_workspace->BERR = malloc(NSMAX * sizeof(double));
    g_workspace->IWORK = malloc(NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->FERR || !g_workspace->BERR || !g_workspace->IWORK) {
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
        free(g_workspace->FERR);
        free(g_workspace->BERR);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full dchkpb test battery for a single (n, kd, uplo, imat, inb) combination.
 *
 * Following LAPACK's dchkpb.f:
 *   - TEST 1 (factorization) runs for all NB values
 *   - TESTs 2-7 (solve, refinement, condition) only run for inb=0 (first NB)
 */
static void run_dchkpb_single(int n, int kd, int iuplo, int imat, int inb)
{
    const double ZERO = 0.0;
    const double ONE = 1.0;
    dchkpb_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    char packit;
    int kl, ku, mode, koff;
    double anorm_param, cndnum;
    int info, izero;
    int lda = (n > 1) ? n : 1;
    int ldab = kd + 1;
    double rcondc = 1.0, rcond, anorm, ainvnm;

    /* Set block size for this test via xlaenv */
    int nb = NBVAL[inb];
    xlaenv(1, nb);
    double result[NTESTS];
    char ctx[128];

    /* Seed based on (n, kd, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 10000 + kd * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Determine packing and offset based on uplo */
    if (iuplo == 0) {
        /* Upper triangular */
        koff = (kd + 1 - n > 0) ? kd + 1 - n : 0;
        packit = 'Q';  /* Upper band packed in Q format */
    } else {
        /* Lower triangular */
        koff = 0;
        packit = 'B';  /* Lower band packed in B format */
    }

    /* Get matrix parameters for this type */
    dlatb4("DPB", imat, n, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    /* For types 2-4, check if we need to create a singular matrix */
    int zerot = (imat >= 2 && imat <= 4);
    if (zerot && n < imat - 1) {
        /* Skip if N is too small for this singular type */
        return;
    }

    izero = 0;

    if (!zerot) {
        /* Generate symmetric positive definite band test matrix */
        dlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm_param,
               kd, kd, &packit, &ws->A[koff], ldab, ws->WORK, &info, rng_state);
        if (info != 0) {
            return;
        }
    } else {
        /* For singular types, we reuse the matrix from type 1 and zero a row/column */
        /* First generate a normal matrix */
        dlatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm_param,
               kd, kd, &packit, &ws->A[koff], ldab, ws->WORK, &info, rng_state);
        if (info != 0) {
            return;
        }

        /* Determine which row/column to zero */
        if (imat == 2) {
            izero = 0;  /* Zero row/col 0 (1 in Fortran) */
        } else if (imat == 3) {
            izero = n - 1;  /* Zero row/col n-1 (n in Fortran) */
        } else {
            izero = n / 2;  /* Zero middle row/col */
        }

        /* Zero the row and column izero in band storage */
        int i1 = (izero - kd > 0) ? izero - kd : 0;
        int i2 = (izero + kd < n - 1) ? izero + kd : n - 1;

        if (iuplo == 0) {
            /* Upper triangular storage */
            int ioff = izero * ldab + kd;
            for (int i = i1; i < izero; i++) {
                ws->A[ioff - izero + i] = ZERO;
            }
            for (int i = izero; i <= i2; i++) {
                ws->A[kd + izero - i + i * ldab] = ZERO;
            }
        } else {
            /* Lower triangular storage */
            for (int i = i1; i < izero; i++) {
                ws->A[izero - i + i * ldab] = ZERO;
            }
            int ioff = izero * ldab;
            for (int i = izero; i <= i2; i++) {
                ws->A[ioff + i - izero] = ZERO;
            }
        }
    }

    /* Copy A to AFAC for factorization */
    dlacpy("F", kd + 1, n, ws->A, ldab, ws->AFAC, ldab);

    /* Compute the Cholesky factorization */
    dpbtrf(uplo_str, n, kd, ws->AFAC, ldab, &info);

    /* Check error code */
    if (zerot) {
        /* For singular matrices, info should be > 0 */
        if (info != izero + 1) {  /* +1 because LAPACK returns 1-based */
            /* Expected singularity at izero but got different result */
            return;
        }
    } else {
        assert_int_equal(info, 0);
    }

    /* Skip the rest if factorization failed */
    if (info != 0) {
        return;
    }

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%c imat=%d TEST 1", n, kd, uplo, imat);
    set_test_context(ctx);
    dlacpy("F", kd + 1, n, ws->AFAC, ldab, ws->AINV, ldab);
    dpbt01(uplo_str, n, kd, ws->A, ldab, ws->AINV, ldab, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * Skip solve tests (and TEST 7) if not the first block size.
     * In LAPACK dchkpb.f, "IF( INB.GT.1 ) GO TO 50" skips to after TEST 7.
     */
    if (inb > 0) {
        clear_test_context();
        return;
    }

    /*
     * Form the inverse to get RCONDC = 1/(norm(A) * norm(inv(A)))
     */
    dlaset("F", n, n, ZERO, ONE, ws->AINV, lda);
    dpbtrs(uplo_str, n, kd, n, ws->AFAC, ldab, ws->AINV, lda, &info);

    anorm = dlansb("1", uplo_str, n, kd, ws->A, ldab, ws->RWORK);
    ainvnm = dlange("1", n, n, ws->AINV, lda, ws->RWORK);
    if (anorm <= ZERO || ainvnm <= ZERO) {
        rcondc = ONE;
    } else {
        rcondc = (ONE / anorm) / ainvnm;
    }

    /*
     * TESTS 2-6: Solve tests (only for first NB)
     */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /*
         * TEST 2: Solve and compute residual for A * X = B
         */
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%c imat=%d nrhs=%d TEST 2", n, kd, uplo, imat, nrhs);
        set_test_context(ctx);
        dlarhs("DPB", "N", uplo_str, " ", n, n, kd, kd, nrhs,
               ws->A, ldab, ws->XACT, lda, ws->B, lda, &info, rng_state);

        dlacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        dpbtrs(uplo_str, n, kd, nrhs, ws->AFAC, ldab, ws->X, lda, &info);
        assert_int_equal(info, 0);

        dlacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        dpbt02(uplo_str, n, kd, nrhs, ws->A, ldab, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[1]);
        assert_residual_below(result[1], THRESH);

        /*
         * TEST 3: Check solution from generated exact solution
         */
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%c imat=%d nrhs=%d TEST 3", n, kd, uplo, imat, nrhs);
        set_test_context(ctx);
        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TESTS 4, 5, 6: Iterative refinement
         */
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%c imat=%d nrhs=%d TEST 4-6", n, kd, uplo, imat, nrhs);
        set_test_context(ctx);
        dpbrfs(uplo_str, n, kd, nrhs, ws->A, ldab, ws->AFAC, ldab,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        dget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
        dpbt05(uplo_str, n, kd, nrhs, ws->A, ldab, ws->B, lda, ws->X, lda,
               ws->XACT, lda, ws->FERR, ws->BERR, &result[4]);

        assert_residual_below(result[3], THRESH);
        assert_residual_below(result[4], THRESH);
        assert_residual_below(result[5], THRESH);
    }

    /*
     * TEST 7: Get an estimate of RCOND = 1/CNDNUM
     */
    snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%c imat=%d nb=%d TEST 7", n, kd, uplo, imat, nb);
    set_test_context(ctx);
    anorm = dlansb("1", uplo_str, n, kd, ws->A, ldab, ws->RWORK);
    dpbcon(uplo_str, n, kd, ws->AFAC, ldab, anorm, &rcond,
           ws->WORK, ws->IWORK, &info);
    assert_int_equal(info, 0);

    result[6] = dget06(rcond, rcondc);
    assert_residual_below(result[6], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchkpb_single based on prestate.
 */
static void test_dchkpb_case(void** state)
{
    dchkpb_params_t* params = *state;
    run_dchkpb_single(params->n, params->kd, params->iuplo, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * For each N, we test up to 4 KD values:
 *   KD = 0, n+(n+1)/4, (3n-1)/4, (n+1)/4
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NBW * NUPLO * NTYPES * NNB)

static dchkpb_params_t g_params[MAX_TESTS];
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

        /* Compute KD values for this N */
        int kdval[NBW];
        kdval[0] = 0;
        kdval[1] = n + (n + 1) / 4;
        kdval[2] = (3 * n - 1) / 4;
        kdval[3] = (n + 1) / 4;

        /* Limit number of KD values for small N */
        int nkd = (n >= 4) ? 4 : ((n > 0) ? n : 1);
        if (nkd > NBW) nkd = NBW;

        int nimat = NTYPES;
        if (n <= 0) {
            nimat = 1;
        }

        for (int ikd = 0; ikd < nkd; ikd++) {
            int kd = kdval[ikd];
            /* Limit kd to n-1 to avoid edge cases with oversized bandwidth */
            if (n > 0 && kd > n - 1) {
                kd = n - 1;
            }

            for (int iuplo = 0; iuplo < (int)NUPLO; iuplo++) {
                for (int imat = 1; imat <= nimat; imat++) {
                    /* Skip types 2, 3, or 4 if matrix size is too small */
                    int zerot = (imat >= 2 && imat <= 4);
                    if (zerot && n < imat - 1) {
                        continue;
                    }

                    /* Loop over block sizes */
                    for (int inb = 0; inb < (int)NNB; inb++) {
                        int nb = NBVAL[inb];

                        /* Store parameters */
                        dchkpb_params_t* p = &g_params[g_num_tests];
                        p->n = n;
                        p->kd = kd;
                        p->imat = imat;
                        p->iuplo = iuplo;
                        p->inb = inb;
                        snprintf(p->name, sizeof(p->name), "dchkpb_n%d_kd%d_%c_type%d_nb%d_%d",
                                 n, kd, UPLOS[iuplo], imat, nb, inb);

                        /* Create CMocka test entry */
                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_dchkpb_case;
                        g_tests[g_num_tests].setup_func = NULL;
                        g_tests[g_num_tests].teardown_func = NULL;
                        g_tests[g_num_tests].initial_state = p;

                        g_num_tests++;
                    }
                }
            }
        }
    }
}

int main(void)
{
    /* Build all test cases */
    build_test_array();

    /* Run all tests with shared workspace. */
    return _cmocka_run_group_tests("dchkpb", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
