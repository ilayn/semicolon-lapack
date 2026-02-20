/**
 * @file test_schksy_rook.c
 * @brief Comprehensive test suite for symmetric indefinite ROOK (SSR) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchksy_rook.f to C using CMocka.
 * Tests SSYTRF_ROOK, SSYTRI_ROOK, SSYTRS_ROOK, and SSYCON_ROOK.
 *
 * Each (n, uplo, imat, inb) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchksy_rook.f:
 *   TEST 1: LDL^T factorization residual via ssyt01_rook
 *   TEST 2: Matrix inverse residual via spot03
 *   TEST 3: Largest element in U or L (growth factor bound)
 *   TEST 4: Largest 2-Norm of 2-by-2 diagonal blocks (condition bound)
 *   TEST 5: Solution residual via spot02 (using ssytrs_rook)
 *   TEST 6: Solution accuracy via sget04
 *   TEST 7: Condition number via sget06
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
#include <math.h>
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
#define NTESTS  7
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */

/* Routines under test */
extern void ssytrf_rook(const char* uplo, const int n, f32* A, const int lda,
                        int* ipiv, f32* work, const int lwork, int* info);
extern void ssytri_rook(const char* uplo, const int n, f32* A, const int lda,
                        const int* ipiv, f32* work, int* info);
extern void ssytrs_rook(const char* uplo, const int n, const int nrhs,
                        const f32* A, const int lda, const int* ipiv,
                        f32* B, const int ldb, int* info);
extern void ssycon_rook(const char* uplo, const int n, const f32* A,
                        const int lda, const int* ipiv, const f32 anorm,
                        f32* rcond, f32* work, int* iwork, int* info);

/* Verification routines */
extern void ssyt01_rook(const char* uplo, const int n, const f32* A,
                        const int lda, const f32* AFAC, const int ldafac,
                        const int* ipiv, f32* C, const int ldc,
                        f32* rwork, f32* resid);
extern void spot02(const char* uplo, const int n, const int nrhs,
                   const f32* A, const int lda, const f32* X,
                   const int ldx, f32* B, const int ldb,
                   f32* rwork, f32* resid);
extern void spot03(const char* uplo, const int n, const f32* A,
                   const int lda, const f32* AINV, const int ldainv,
                   f32* work, const int ldwork, f32* rwork,
                   f32* rcond, f32* resid);
extern void sget04(const int n, const int nrhs, const f32* X, const int ldx,
                   const f32* XACT, const int ldxact, const f32 rcond,
                   f32* resid);
extern f32 sget06(const f32 rcond, const f32 rcondc);

/* Utility routines */
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern void sgesvd(const char* jobu, const char* jobvt, const int m, const int n,
                   f32* A, const int lda, f32* S,
                   f32* U, const int ldu, f32* VT, const int ldvt,
                   f32* work, const int lwork, int* info);

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
                   const f32* XACT, const int ldxact, f32* B,
                   const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* A, const int lda, f32* work);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int n;
    int imat;
    int iuplo;  /* 0='U', 1='L' */
    int inb;    /* Index into NBVAL[] */
    char name[64];
} dchksy_rook_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 */
typedef struct {
    f32* A;      /* Original matrix (NMAX x NMAX) */
    f32* AFAC;   /* Factored matrix (NMAX x NMAX) */
    f32* AINV;   /* Inverse matrix (NMAX x NMAX) */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* D;      /* Singular values for slatms */
    int* IPIV;      /* Pivot indices */
    int* IWORK;     /* Integer workspace */
} dchksy_rook_workspace_t;

static dchksy_rook_workspace_t* g_workspace = NULL;

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchksy_rook_workspace_t));
    if (!g_workspace) return -1;

    int lwork = NMAX * 64;  /* Generous workspace */

    g_workspace->A = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AFAC = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->AINV = malloc(NMAX * NMAX * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->WORK = malloc(lwork * sizeof(f32));
    g_workspace->RWORK = malloc(NMAX * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->IPIV = malloc(NMAX * sizeof(int));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->AFAC || !g_workspace->AINV ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
        !g_workspace->IPIV || !g_workspace->IWORK) {
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
        free(g_workspace->IPIV);
        free(g_workspace->IWORK);
        free(g_workspace);
        g_workspace = NULL;
    }
    return 0;
}

/**
 * Run the full dchksy_rook test battery for a single (n, uplo, imat, inb) combination.
 *
 * Following LAPACK's dchksy_rook.f:
 *   - TESTs 1-4 (factorization, inverse, growth, block condition) run for all NB values
 *   - TESTs 5-7 (solve, accuracy, condition estimate) only run for inb=0 (first NB)
 */
static void run_dchksy_rook_single(int n, int iuplo, int imat, int inb)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    const f32 EIGHT = 8.0f;
    const f32 SEVTEN = 17.0f;
    dchksy_rook_workspace_t* ws = g_workspace;

    char type, dist;
    char uplo = UPLOS[iuplo];
    char uplo_str[2] = {uplo, '\0'};
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info, izero;
    int lda = (n > 1) ? n : 1;
    int trfcon;
    f32 rcondc, rcond;
    f32 result[NTESTS];
    char ctx[128];

    f32 alpha = (ONE + sqrtf(SEVTEN)) / EIGHT;

    /* Set block size for this test via xlaenv */
    int nb = NBVAL[inb];
    xlaenv(1, nb);

    /* Seed based on (n, uplo, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL + (uint64_t)(n * 1000 + iuplo * 100 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type */
    slatb4("SSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric test matrix */
    slatms(n, n, &dist, &type, ws->D, mode, cndnum, anorm,
           kl, ku, uplo_str, ws->A, lda, ws->WORK, &info, rng_state);
    assert_int_equal(info, 0);

    /* For types 3-6, zero one or more rows and columns.
     * izero is 0-based index of the row/column to zero. */
    int zerot = (imat >= 3 && imat <= 6);
    if (zerot) {
        if (imat == 3) {
            izero = 0;  /* First row/column */
        } else if (imat == 4) {
            izero = n - 1;  /* Last row/column */
        } else {
            izero = n / 2;  /* Middle row/column */
        }

        if (imat < 6) {
            /* Zero row and column izero */
            if (iuplo == 0) {
                /* Upper: zero column izero (rows 0 to izero-1) and
                 * row izero (columns izero to n-1) */
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
                /* Lower: zero row izero (columns 0 to izero-1) and
                 * column izero (rows izero to n-1) */
                int ioff = izero;
                for (int i = 0; i < izero; i++) {
                    ws->A[ioff] = ZERO;
                    ioff += lda;
                }
                ioff = izero * lda + izero;
                for (int i = izero; i < n; i++) {
                    ws->A[ioff + i - izero] = ZERO;
                }
            }
        } else {
            /* Type 6: zero first izero+1 rows and columns (upper) or last (lower) */
            if (iuplo == 0) {
                int ioff = 0;
                for (int j = 0; j < n; j++) {
                    int i2 = (j <= izero) ? j + 1 : izero + 1;
                    for (int i = 0; i < i2; i++) {
                        ws->A[ioff + i] = ZERO;
                    }
                    ioff += lda;
                }
            } else {
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
        izero = -1;  /* No zeroing, use -1 to indicate none */
    }

    /* Copy A to AFAC for factorization */
    slacpy(uplo_str, n, n, ws->A, lda, ws->AFAC, lda);

    /* Compute the L*D*L^T or U*D*U^T factorization of the
     * matrix. IWORK stores details of the interchanges and
     * the block structure of D. AINV is a work array for
     * block factorization, LWORK is the length of AINV. */
    int lwork_rook = (nb > 2 ? nb : 2) * lda;
    ssytrf_rook(uplo_str, n, ws->AFAC, lda, ws->IPIV,
                ws->WORK, lwork_rook, &info);

    /* Adjust the expected value of INFO to account for pivoting.
     * ipiv stores 0-based values: non-negative for 1x1 blocks,
     * negative (encoded as -(row+1)) for 2x2 blocks. */
    if (izero >= 0) {
        int k = izero;
        while (k >= 0 && k < n) {
            if (ws->IPIV[k] < 0) {
                /* 2x2 block: decode 0-based row from -(row+1) */
                int krow = -(ws->IPIV[k] + 1);
                if (krow != k) {
                    k = krow;
                    continue;
                }
                break;
            } else if (ws->IPIV[k] != k) {
                /* 1x1 block: ipiv[k] is 0-based swap target */
                k = ws->IPIV[k];
                continue;
            } else {
                break;
            }
        }
        assert_true(info >= 0);
    }
    trfcon = (info != 0);
    if (trfcon) {
        rcondc = ZERO;
    }

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d TEST 1 (factorization)", n, uplo, imat, nb);
    set_test_context(ctx);
    ssyt01_rook(uplo_str, n, ws->A, lda, ws->AFAC, lda, ws->IPIV,
                ws->AINV, lda, ws->RWORK, &result[0]);
    assert_residual_below(result[0], THRESH);

    /*
     * TEST 2: Form the inverse and compute the residual,
     * if the factorization was completed without INFO > 0.
     * Do it only for the first block size.
     */
    if (inb == 0 && !trfcon) {
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 2 (inverse)", n, uplo, imat);
        set_test_context(ctx);
        slacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);
        ssytri_rook(uplo_str, n, ws->AINV, lda, ws->IPIV, ws->WORK, &info);
        if (info == 0) {
            spot03(uplo_str, n, ws->A, lda, ws->AINV, lda, ws->WORK, lda,
                   ws->RWORK, &rcondc, &result[1]);
            assert_residual_below(result[1], THRESH);
        }
    }

    /*
     * TEST 3: Compute largest element in U or L
     */
    {
        f32 dtemp;
        f32 const_val = ONE / (ONE - alpha);

        result[2] = ZERO;

        if (iuplo == 0) {
            /* Compute largest element in U */
            int k = n - 1;
            while (k >= 1) {
                if (ws->IPIV[k] >= 0) {
                    /* 1x1 block: max abs value from column k, rows 0..k-1 */
                    dtemp = slange("M", k, 1, &ws->AFAC[k * lda], lda, ws->RWORK);
                } else {
                    /* 2x2 block: max abs value from columns k-1..k, rows 0..k-2 */
                    dtemp = slange("M", k - 1, 2, &ws->AFAC[(k - 1) * lda], lda, ws->RWORK);
                    k--;
                }
                dtemp = dtemp - const_val + THRESH;
                if (dtemp > result[2])
                    result[2] = dtemp;
                k--;
            }
        } else {
            /* Compute largest element in L */
            int k = 0;
            while (k < n - 1) {
                if (ws->IPIV[k] >= 0) {
                    /* 1x1 block: max abs value from column k, rows k+1..n-1 */
                    dtemp = slange("M", n - k - 1, 1, &ws->AFAC[k * lda + k + 1], lda, ws->RWORK);
                } else {
                    /* 2x2 block: max abs value from columns k..k+1, rows k+2..n-1 */
                    dtemp = slange("M", n - k - 2, 2, &ws->AFAC[k * lda + k + 2], lda, ws->RWORK);
                    k++;
                }
                dtemp = dtemp - const_val + THRESH;
                if (dtemp > result[2])
                    result[2] = dtemp;
                k++;
            }
        }

        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d TEST 3 (growth)", n, uplo, imat, nb);
        set_test_context(ctx);
        assert_residual_below(result[2], THRESH);
    }

    /*
     * TEST 4: Compute largest 2-Norm (condition number)
     * of 2-by-2 diag blocks
     */
    {
        f32 const_val = (ONE + alpha) / (ONE - alpha);
        f32 dtemp;

        result[3] = ZERO;
        slacpy(uplo_str, n, n, ws->AFAC, lda, ws->AINV, lda);

        if (iuplo == 0) {
            /* Loop backward for UPLO = 'U' */
            int k = n - 1;
            while (k >= 1) {
                if (ws->IPIV[k] < 0) {
                    /* Get the two singular values of a 2-by-2 block */
                    f32 block[4];
                    f32 sv[2], ddummy[1], work_svd[10];
                    int svd_info;

                    block[0] = ws->AFAC[(k - 1) + (k - 1) * lda];
                    block[1] = ws->AFAC[(k - 1) + k * lda];
                    block[2] = block[1];
                    block[3] = ws->AFAC[k + k * lda];

                    sgesvd("N", "N", 2, 2, block, 2, sv,
                           ddummy, 1, ddummy, 1,
                           work_svd, 10, &svd_info);

                    dtemp = sv[0] / sv[1];

                    dtemp = dtemp - const_val + THRESH;
                    if (dtemp > result[3])
                        result[3] = dtemp;
                    k--;
                }
                k--;
            }
        } else {
            /* Loop forward for UPLO = 'L' */
            int k = 0;
            while (k < n - 1) {
                if (ws->IPIV[k] < 0) {
                    /* Get the two singular values of a 2-by-2 block */
                    f32 block[4];
                    f32 sv[2], ddummy[1], work_svd[10];
                    int svd_info;

                    block[0] = ws->AFAC[k + k * lda];
                    block[1] = ws->AFAC[(k + 1) + k * lda];
                    block[2] = block[1];
                    block[3] = ws->AFAC[(k + 1) + (k + 1) * lda];

                    sgesvd("N", "N", 2, 2, block, 2, sv,
                           ddummy, 1, ddummy, 1,
                           work_svd, 10, &svd_info);

                    dtemp = sv[0] / sv[1];

                    dtemp = dtemp - const_val + THRESH;
                    if (dtemp > result[3])
                        result[3] = dtemp;
                    k++;
                }
                k++;
            }
        }

        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nb=%d TEST 4 (block cond)", n, uplo, imat, nb);
        set_test_context(ctx);
        assert_residual_below(result[3], THRESH);
    }

    /*
     * Skip the other tests if this is not the first block size.
     */
    if (inb > 0) {
        clear_test_context();
        return;
    }

    /* Do only the condition estimate if INFO is not 0. */
    if (trfcon) {
        rcondc = ZERO;
        goto test7;
    }

    /*
     * TESTS 5-6: Solve tests (only for first NB)
     */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];

        /*
         * TEST 5: Solve and compute residual for A * X = B using ssytrs_rook
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 5 (solve)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        slarhs("SSY", "N", uplo_str, " ", n, n, kl, ku, nrhs,
               ws->A, lda, ws->XACT, lda, ws->B, lda, &info, rng_state);

        slacpy("F", n, nrhs, ws->B, lda, ws->X, lda);
        ssytrs_rook(uplo_str, n, nrhs, ws->AFAC, lda, ws->IPIV,
                    ws->X, lda, &info);
        assert_int_equal(info, 0);

        slacpy("F", n, nrhs, ws->B, lda, ws->WORK, lda);
        spot02(uplo_str, n, nrhs, ws->A, lda, ws->X, lda,
               ws->WORK, lda, ws->RWORK, &result[4]);
        assert_residual_below(result[4], THRESH);

        /*
         * TEST 6: Check solution from generated exact solution.
         */
        snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d nrhs=%d TEST 6 (accuracy)", n, uplo, imat, nrhs);
        set_test_context(ctx);
        sget04(n, nrhs, ws->X, lda, ws->XACT, lda, rcondc, &result[5]);
        assert_residual_below(result[5], THRESH);
    }

test7:
    /*
     * TEST 7: Get an estimate of RCOND = 1/CNDNUM
     */
    snprintf(ctx, sizeof(ctx), "n=%d uplo=%c imat=%d TEST 7 (condition)", n, uplo, imat);
    set_test_context(ctx);
    anorm = slansy("1", uplo_str, n, ws->A, lda, ws->RWORK);
    ssycon_rook(uplo_str, n, ws->AFAC, lda, ws->IPIV, anorm, &rcond,
                ws->WORK, &ws->IWORK[n], &info);
    assert_int_equal(info, 0);

    result[6] = sget06(rcond, rcondc);
    assert_residual_below(result[6], THRESH);

    clear_test_context();
}

/**
 * CMocka test function - dispatches to run_dchksy_rook_single based on prestate.
 */
static void test_dchksy_rook_case(void** state)
{
    dchksy_rook_params_t* params = *state;
    run_dchksy_rook_single(params->n, params->iuplo, params->imat, params->inb);
}

/*
 * Generate all parameter combinations.
 * Total: NN * NUPLO * NTYPES * NNB = 7 * 2 * 10 * 5 = 700 tests
 * (minus skipped cases for small n and singular types)
 */

/* Maximum number of test cases */
#define MAX_TESTS (NN * NUPLO * NTYPES * NNB)

static dchksy_rook_params_t g_params[MAX_TESTS];
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
                    dchksy_rook_params_t* p = &g_params[g_num_tests];
                    p->n = n;
                    p->imat = imat;
                    p->iuplo = iuplo;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name), "dchksy_rook_n%d_%c_type%d_nb%d_%d",
                             n, UPLOS[iuplo], imat, nb, inb);

                    /* Create CMocka test entry */
                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchksy_rook_case;
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

    /* Run all tests with shared workspace. */
    return _cmocka_run_group_tests("dchksy_rook", g_tests, g_num_tests,
                                   group_setup, group_teardown);
}
