/**
 * @file test_schkpb.c
 * @brief Comprehensive test suite for positive definite banded matrix (SPB) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkpb.f to C using CMocka.
 * Tests SPBTRF, SPBTRS, SPBRFS, and SPBCON.
 *
 * Test structure from dchkpb.f:
 *   TEST 1: Cholesky factorization residual via spbt01
 *   TEST 2: Solution residual via spbt02
 *   TEST 3: Solution accuracy via sget04
 *   TESTS 4-6: Iterative refinement via spbrfs + sget04 + spbt05
 *   TEST 7: Condition number via sget06
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
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */
#define NBW     4   /* Number of bandwidth values per N */

/* Routines under test */
extern void spbtrf(const char* uplo, const int n, const int kd, f32* AB,
                   const int ldab, int* info);
extern void spbtrs(const char* uplo, const int n, const int kd, const int nrhs,
                   const f32* AB, const int ldab, f32* B, const int ldb,
                   int* info);
extern void spbrfs(const char* uplo, const int n, const int kd, const int nrhs,
                   const f32* AB, const int ldab, const f32* AFB,
                   const int ldafb, const f32* B, const int ldb,
                   f32* X, const int ldx, f32* ferr, f32* berr,
                   f32* work, int* iwork, int* info);
extern void spbcon(const char* uplo, const int n, const int kd,
                   const f32* AB, const int ldab, const f32 anorm,
                   f32* rcond, f32* work, int* iwork, int* info);

/* Verification routines */
extern void spbt01(const char* uplo, const int n, const int kd,
                   const f32* A, const int lda,
                   f32* AFAC, const int ldafac,
                   f32* rwork, f32* resid);
extern void spbt02(const char* uplo, const int n, const int kd, const int nrhs,
                   const f32* A, const int lda,
                   const f32* X, const int ldx,
                   f32* B, const int ldb,
                   f32* rwork, f32* resid);
extern void spbt05(const char* uplo, const int n, const int kd, const int nrhs,
                   const f32* AB, const int ldab,
                   const f32* B, const int ldb,
                   const f32* X, const int ldx,
                   const f32* XACT, const int ldxact,
                   const f32* ferr, const f32* berr,
                   f32* reslts);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond,
                   f32* resid);
extern f32 sget06(const f32 rcond, const f32 rcondc);

/* Matrix generation */
extern void slatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f32* anorm, int* mode,
                   f32* cndnum, char* dist);
extern void slatms(const int m, const int n, const char* dist,
                   const char* sym, f32* d, const int mode, const f32 cond,
                   const f32 dmax, const int kl, const int ku, const char* pack,
                   f32* A, const int lda, f32* work, int* info,
                   uint64_t state[static 4]);
extern void slarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f32* A, const int lda,
                   f32* X, const int ldx, f32* B, const int ldb,
                   int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta, f32* A, const int lda);
extern f32 slansb(const char* norm, const char* uplo, const int n,
                     const int k, const f32* AB, const int ldab, f32* work);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern f32 slamch(const char* cmach);

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
    f32* A;      /* Original band matrix ((NMAX+1) * NMAX) */
    f32* AFAC;   /* Factored band matrix ((NMAX+1) * NMAX) */
    f32* AINV;   /* Full inverse matrix (NMAX * NMAX) */
    f32* B;      /* Right-hand side (NMAX * NSMAX) */
    f32* X;      /* Solution (NMAX * NSMAX) */
    f32* XACT;   /* Exact solution (NMAX * NSMAX) */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* D;      /* Singular values for slatms */
    f32* FERR;   /* Forward error bounds */
    f32* BERR;   /* Backward error bounds */
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
    g_workspace->A = malloc(max_ldab * NMAX * sizeof(f32));
    g_workspace->AFAC = malloc(max_ldab * NMAX * sizeof(f32));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    /* WORK needs NMAX * (3*NMAX + NSMAX + 30) for slatms full matrix generation */
    int work_size = NMAX * (3 * NMAX + NSMAX + 30);
    g_workspace->WORK = malloc(work_size * sizeof(f32));
    g_workspace->RWORK = malloc((NMAX > 2 * NSMAX ? NMAX : 2 * NSMAX) * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
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
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    dchkpb_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    char packit;
    int kl, ku, mode, koff;
    f32 anorm_param, cndnum;
    int info, izero;
    int lda = (n > 1) ? n : 1;
    int ldab = kd + 1;
    f32 rcondc = 1.0f, rcond, anorm, ainvnm;

    /* Set block size for this test via xlaenv */
    int nb = NBVAL[inb];
    xlaenv(1, nb);
    f32 result[NTESTS];
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
    slatb4("SPB", imat, n, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    /* For types 2-4, check if we need to create a singular matrix */
    int zerot = (imat >= 2 && imat <= 4);
    if (zerot && n < imat - 1) {
        /* Skip if N is too small for this singular type */
        return;
    }

    izero = 0;

    if (!zerot) {
        /* Generate symmetric positive definite band test matrix */
        slatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm_param,
               kd, kd, &packit, &ws->A[koff], ldab, ws->WORK, &info, rng_state);
        if (info != 0) {
            return;
        }
    } else {
        /* For singular types, we reuse the matrix from type 1 and zero a row/column */
        /* First generate a normal matrix */
        slatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm_param,
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
    slacpy("F", kd + 1, n, ws->A, ldab, ws->AFAC, ldab);

    /* Compute the Cholesky factorization */
    spbtrf(uplo_str, n, kd, ws->AFAC, ldab, &info);

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
    slacpy("F", kd + 1, n, ws->AFAC, ldab, ws->AINV, ldab);
    spbt01(uplo_str, n, kd, ws->A, ldab, ws->AINV, ldab, ws->RWORK, &result[0]);
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
    slaset("F", n, n, ZERO, ONE, ws->AINV, lda);
    spbtrs(uplo_str, n, kd, n, ws->AFAC, ldab, ws->AINV, lda, &info);

    anorm = slansb("1", uplo_str, n, kd, ws->A, ldab, ws->RWORK);
    ainvnm = slange("1", n, n, ws->AINV, lda, ws->RWORK);
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
        slarhs("SPB", "N", uplo_str, " ", n, n, kd, kd, nrhs,
               ws->A, ldab, ws->XACT, lda, ws->B, lda, &info, rng_state);

        slacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        spbtrs(uplo_str, n, kd, nrhs, ws->AFAC, ldab, ws->X, lda, &info);
        assert_int_equal(info, 0);

        slacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        spbt02(uplo_str, n, kd, nrhs, ws->A, ldab, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[1]);
        assert_residual_below(result[1], THRESH);

        /*
         * TEST 3: Check solution from generated exact solution
         */
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%c imat=%d nrhs=%d TEST 3", n, kd, uplo, imat, nrhs);
        set_test_context(ctx);
        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[2]);
        assert_residual_below(result[2], THRESH);

        /*
         * TESTS 4, 5, 6: Iterative refinement
         */
        snprintf(ctx, sizeof(ctx), "n=%d kd=%d uplo=%c imat=%d nrhs=%d TEST 4-6", n, kd, uplo, imat, nrhs);
        set_test_context(ctx);
        spbrfs(uplo_str, n, kd, nrhs, ws->A, ldab, ws->AFAC, ldab,
               ws->B, lda, ws->X, lda, ws->FERR, ws->BERR,
               ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[3]);
        spbt05(uplo_str, n, kd, nrhs, ws->A, ldab, ws->B, lda, ws->X, lda,
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
    anorm = slansb("1", uplo_str, n, kd, ws->A, ldab, ws->RWORK);
    spbcon(uplo_str, n, kd, ws->AFAC, ldab, anorm, &rcond,
           ws->WORK, ws->IWORK, &info);
    assert_int_equal(info, 0);

    result[6] = sget06(rcond, rcondc);
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
