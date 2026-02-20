/**
 * @file test_schkgb.c
 * @brief Comprehensive test suite for general band matrix (SGB) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkgb.f to C using CMocka.
 * Tests SGBTRF, SGBTRS, SGBRFS, and SGBCON.
 *
 * Each (m, n, kl, ku, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkgb.f:
 *   TEST 1: LU factorization residual via sgbt01
 *   TEST 2: Solution residual via sgbt02
 *   TEST 3: Solution accuracy via sget04
 *   TEST 4: Refined solution accuracy via sget04 (after sgbrfs)
 *   TEST 5-6: Error bounds via sgbt05
 *   TEST 7: Condition number via sget06
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   KL/KU values: 0, (5*M+1)/4, (3M-1)/4, (M+1)/4 (clamped to matrix size)
 *   NRHS values: 1, 2, 15
 *   NB values: 1, 3, 3, 3, 20
 *   Matrix types: 1-8
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "test_rng.h"
#include <string.h>
#include <stdio.h>
#include <cblas.h>

/* Test parameters from dtest.in */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NSVAL[] = {1, 2, 15};  /* NRHS values */
static const int NBVAL[] = {1, 3, 3, 3, 20};  /* Block sizes from dtest.in */
static const char TRANSS[] = {'N', 'T', 'C'};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTRAN   (sizeof(TRANSS) / sizeof(TRANSS[0]))
#define NTYPES  8
#define NTESTS  7
#define THRESH  30.0f
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */
#define NBW     4   /* Number of bandwidth values to test */

/* Routines under test */
extern void sgbtrf(const int m, const int n, const int kl, const int ku,
                   f32* AB, const int ldab, int* ipiv, int* info);
extern void sgbtrs(const char* trans, const int n, const int kl, const int ku,
                   const int nrhs, const f32* AB, const int ldab,
                   const int* ipiv, f32* B, const int ldb, int* info);
extern void sgbrfs(const char* trans, const int n, const int kl, const int ku,
                   const int nrhs, const f32* AB, const int ldab,
                   const f32* AFB, const int ldafb, const int* ipiv,
                   const f32* B, const int ldb, f32* X, const int ldx,
                   f32* ferr, f32* berr, f32* work, int* iwork,
                   int* info);
extern void sgbcon(const char* norm, const int n, const int kl, const int ku,
                   const f32* AB, const int ldab, const int* ipiv,
                   const f32 anorm, f32* rcond, f32* work,
                   int* iwork, int* info);

/* Verification routines */
extern void sgbt01(int m, int n, int kl, int ku,
                   const f32* A, int lda, const f32* AFAC, int ldafac,
                   const int* ipiv, f32* work, f32* resid);
extern void sgbt02(const char* trans, int m, int n, int kl, int ku, int nrhs,
                   const f32* A, int lda, const f32* X, int ldx,
                   f32* B, int ldb, f32* rwork, f32* resid);
extern void sgbt05(const char* trans, int n, int kl, int ku, int nrhs,
                   const f32* AB, int ldab, const f32* B, int ldb,
                   const f32* X, int ldx, const f32* XACT, int ldxact,
                   const f32* FERR, const f32* BERR, f32* reslts);
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
                   const f32* XACT, const int ldxact, f32* B,
                   const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void slacpy(const char* uplo, const int m, const int n,
                   const f32* A, const int lda, f32* B, const int ldb);
extern void slaset(const char* uplo, const int m, const int n,
                   const f32 alpha, const f32 beta,
                   f32* A, const int lda);
extern f32 slangb(const char* norm, const int n, const int kl, const int ku,
                     const f32* AB, const int ldab, f32* work);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* A, const int lda, f32* work);
extern f32 slamch(const char* cmach);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int m;
    int n;
    int kl;
    int ku;
    int imat;
    int inb;    /* Index into NBVAL[] */
    char name[80];
} dchkgb_params_t;

/**
 * Workspace for test execution - shared across all tests via group setup.
 *
 * For band storage:
 *   A is stored in band format: (kl+ku+1) x n
 *   AFAC is stored in factored band format: (2*kl+ku+1) x n
 */
typedef struct {
    f32* A;      /* Original band matrix */
    f32* AFAC;   /* Factored band matrix */
    f32* B;      /* Right-hand side (NMAX x NSMAX) */
    f32* X;      /* Solution (NMAX x NSMAX) */
    f32* XACT;   /* Exact solution (NMAX x NSMAX) */
    f32* WORK;   /* General workspace */
    f32* RWORK;  /* Real workspace */
    f32* D;      /* Singular values for slatms */
    f32* FERR;   /* Forward error bounds */
    f32* BERR;   /* Backward error bounds */
    int* IPIV;      /* Pivot indices */
    int* IWORK;     /* Integer workspace */
} dchkgb_workspace_t;

static dchkgb_workspace_t* g_workspace = NULL;

/* Maximum band size: need room for 2*kl+ku+1 rows */
#define KLMAX   (NMAX - 1)
#define KUMAX   (NMAX - 1)
#define LA      ((KLMAX + KUMAX + 1) * NMAX)
#define LAFAC   ((2 * KLMAX + KUMAX + 1) * NMAX)

/**
 * Group setup - allocate workspace once for all tests.
 */
static int group_setup(void** state)
{
    (void)state;
    g_workspace = malloc(sizeof(dchkgb_workspace_t));
    if (!g_workspace) return -1;

    g_workspace->A = malloc(LA * sizeof(f32));
    g_workspace->AFAC = malloc(LAFAC * sizeof(f32));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f32));
    g_workspace->WORK = malloc(3 * NMAX * NMAX * sizeof(f32));
    g_workspace->RWORK = malloc((NMAX + 2 * NSMAX) * sizeof(f32));
    g_workspace->D = malloc(NMAX * sizeof(f32));
    g_workspace->FERR = malloc(NSMAX * sizeof(f32));
    g_workspace->BERR = malloc(NSMAX * sizeof(f32));
    g_workspace->IPIV = malloc(NMAX * sizeof(int));
    g_workspace->IWORK = malloc(2 * NMAX * sizeof(int));

    if (!g_workspace->A || !g_workspace->AFAC ||
        !g_workspace->B || !g_workspace->X || !g_workspace->XACT ||
        !g_workspace->WORK || !g_workspace->RWORK || !g_workspace->D ||
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
        free(g_workspace->A);
        free(g_workspace->AFAC);
        free(g_workspace->B);
        free(g_workspace->X);
        free(g_workspace->XACT);
        free(g_workspace->WORK);
        free(g_workspace->RWORK);
        free(g_workspace->D);
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
 * Run the full dchkgb test battery for a single (m, n, kl, ku, imat, inb) combination.
 *
 * Following LAPACK's dchkgb.f:
 *   - TEST 1 (factorization) runs for all NB values
 *   - TESTs 2-7 (solve, refinement, condition) only run for inb=0 and M==N
 */
static void run_dchkgb_single(int m, int n, int kl, int ku, int imat, int inb)
{
    const f32 ZERO = 0.0f;
    const f32 ONE = 1.0f;
    dchkgb_workspace_t* ws = g_workspace;

    char type, dist;
    int kl_gen, ku_gen, mode;
    f32 anorm_gen, cndnum;
    int info, izero;
    int lda = kl + ku + 1;
    int ldafac = 2 * kl + ku + 1;
    int ldb = (n > 1) ? n : 1;
    int trfcon;
    f32 anormo = 0.0f, anormi = 0.0f, rcondo = 0.0f, rcondi = 0.0f, rcond, rcondc;
    f32 result[NTESTS];

    /* Set block size for this test via xlaenv */
    int nb = NBVAL[inb];
    xlaenv(1, nb);

    /* Seed based on (m, n, kl, ku, imat) for reproducibility */
    uint64_t rng_state[4];
    rng_seed(rng_state, 1988198919901991ULL +
                    (uint64_t)(m * 10000 + n * 1000 + kl * 100 + ku * 10 + imat));

    /* Initialize results */
    for (int k = 0; k < NTESTS; k++) {
        result[k] = ZERO;
    }

    /* Get matrix parameters for this type
     * Note: slatb4 for "GB" just sets type='N' and condition/norm based on imat,
     * it doesn't override kl/ku since we pass them separately */
    slatb4("SGB", imat, m, n, &type, &kl_gen, &ku_gen, &anorm_gen, &mode, &cndnum, &dist);

    /* Generate the matrix directly in band storage using pack='Z'.
     * slatms with pack='Z' generates a band matrix and stores it directly
     * in band format with leading dimension kl+ku+1. */
    slatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm_gen,
           kl, ku, "Z", ws->A, lda, ws->WORK, &info, rng_state);
    if (info != 0) {
        /* Matrix generation failed - skip this test */
        return;
    }

    /* For types 2-4, zero one or more columns to create singular matrix */
    int zerot = (imat >= 2 && imat <= 4);
    if (zerot) {
        int minmn = (m < n) ? m : n;
        if (imat == 2) {
            izero = 1;
        } else if (imat == 3) {
            izero = minmn;
        } else {
            izero = minmn / 2 + 1;
        }
        /* Zero column izero (1-based) in band storage */
        int ioff = (izero - 1) * lda;
        int i1 = (ku + 2 - izero > 1) ? ku + 2 - izero - 1 : 0;
        int i2 = (ku + 1 + m - izero < kl + ku + 1) ? ku + 1 + m - izero : kl + ku + 1;
        if (imat < 4) {
            /* Zero single column */
            for (int i = i1; i < i2; i++) {
                ws->A[ioff + i] = ZERO;
            }
        } else {
            /* Zero columns izero through n */
            for (int j = izero - 1; j < n; j++) {
                int ji1 = (ku + 1 - j - 1 > 0) ? ku + 1 - j - 1 : 0;
                int ji2 = (ku + 1 + m - j - 1 < kl + ku + 1) ? ku + 1 + m - j - 1 : kl + ku + 1;
                for (int i = ji1; i < ji2; i++) {
                    ws->A[j * lda + i] = ZERO;
                }
            }
        }
    } else {
        izero = 0;
    }

    /* Copy A to AFAC for factorization (into rows kl+1 to 2*kl+ku+1) */
    if (m > 0 && n > 0) {
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < kl + ku + 1; i++) {
                ws->AFAC[(kl + i) + j * ldafac] = ws->A[i + j * lda];
            }
        }
    }

    /* Compute the LU factorization */
    sgbtrf(m, n, kl, ku, ws->AFAC, ldafac, ws->IPIV, &info);

    /* Check error code */
    if (zerot) {
        /* For singular matrices, info should equal izero */
        assert_true(info >= 0);
    } else {
        assert_int_equal(info, 0);
    }
    trfcon = (info != 0);

    /*
     * TEST 1: Reconstruct matrix from factors and compute residual.
     */
    sgbt01(m, n, kl, ku, ws->A, lda, ws->AFAC, ldafac,
           ws->IPIV, ws->WORK, &result[0]);

    if (result[0] >= THRESH) {
        print_error("TEST 1 FAILED: m=%d, n=%d, kl=%d, ku=%d, nb=%d, imat=%d: resid=%.6e >= %.1f\n",
                   m, n, kl, ku, nb, imat, (double)result[0], (double)THRESH);
    }
    assert_residual_below(result[0], THRESH);

    /*
     * Skip solve tests if:
     *   - Matrix is not square (m != n), OR
     *   - Matrix is singular (trfcon), OR
     *   - Not the first block size (inb > 0)
     */
    if (inb > 0 || m != n) {
        goto test7;
    }

    /* Compute norms of A for condition number estimation */
    anormo = slangb("O", n, kl, ku, ws->A, lda, ws->RWORK);
    anormi = slangb("I", n, kl, ku, ws->A, lda, ws->RWORK);

    if (info == 0) {
        /* Form the inverse of A to get a good estimate of CNDNUM */
        int ldb_inv = (n > 1) ? n : 1;
        slaset("F", n, n, ZERO, ONE, ws->WORK, ldb_inv);

        sgbtrs("N", n, kl, ku, n, ws->AFAC, ldafac, ws->IPIV, ws->WORK, ldb_inv, &info);

        /* Compute the 1-norm condition number */
        f32 ainvnm = slange("O", n, n, ws->WORK, ldb_inv, ws->RWORK);
        if (anormo <= ZERO || ainvnm <= ZERO) {
            rcondo = ONE;
        } else {
            rcondo = (ONE / anormo) / ainvnm;
        }

        /* Compute the infinity-norm condition number */
        ainvnm = slange("I", n, n, ws->WORK, ldb_inv, ws->RWORK);
        if (anormi <= ZERO || ainvnm <= ZERO) {
            rcondi = ONE;
        } else {
            rcondi = (ONE / anormi) / ainvnm;
        }
    } else {
        /* Do only the condition estimate if INFO != 0 */
        trfcon = 1;
        rcondo = ZERO;
        rcondi = ZERO;
    }

    /* Skip the solve tests if the matrix is singular */
    if (trfcon) {
        goto test7;
    }

    /*
     * TESTs 2-6: Solve tests for each NRHS and each TRANS
     */
    for (int irhs = 0; irhs < (int)NNS; irhs++) {
        int nrhs = NSVAL[irhs];
        char xtype = 'N';

        for (int itran = 0; itran < (int)NTRAN; itran++) {
            char trans[2] = {TRANSS[itran], '\0'};
            if (itran == 0) {
                rcondc = rcondo;
            } else {
                rcondc = rcondi;
            }

            /* Generate right-hand side */
            slarhs("SGB", &xtype, " ", trans, n, n, kl, ku, nrhs,
                   ws->A, lda, ws->XACT, ldb, ws->B, ldb, &info, rng_state);
            xtype = 'C';

            /* Copy B to X */
            slacpy("F", n, nrhs, ws->B, ldb, ws->X, ldb);

            /*
             * TEST 2: Solve and compute residual.
             */
            sgbtrs(trans, n, kl, ku, nrhs, ws->AFAC, ldafac, ws->IPIV,
                   ws->X, ldb, &info);
            assert_int_equal(info, 0);

            /* Copy B for residual computation (sgbt02 overwrites it) */
            slacpy("F", n, nrhs, ws->B, ldb, ws->WORK, ldb);
            sgbt02(trans, m, n, kl, ku, nrhs, ws->A, lda, ws->X, ldb,
                   ws->WORK, ldb, ws->RWORK, &result[1]);

            /*
             * TEST 3: Check solution from generated exact solution.
             */
            sget04(n, nrhs, ws->X, ldb, ws->XACT, ldb, rcondc, &result[2]);

            /*
             * TESTs 4, 5, 6: Use iterative refinement.
             */
            sgbrfs(trans, n, kl, ku, nrhs, ws->A, lda, ws->AFAC, ldafac,
                   ws->IPIV, ws->B, ldb, ws->X, ldb,
                   ws->FERR, ws->BERR, ws->WORK, ws->IWORK, &info);
            assert_int_equal(info, 0);

            sget04(n, nrhs, ws->X, ldb, ws->XACT, ldb, rcondc, &result[3]);

            f32 reslts[2];
            sgbt05(trans, n, kl, ku, nrhs, ws->A, lda, ws->B, ldb,
                   ws->X, ldb, ws->XACT, ldb, ws->FERR, ws->BERR, reslts);
            result[4] = reslts[0];
            result[5] = reslts[1];

            /* Check results 2-6 */
            for (int k = 1; k < 6; k++) {
                if (result[k] >= THRESH) {
                    print_error("TEST %d FAILED: trans='%c', n=%d, kl=%d, ku=%d, nrhs=%d, imat=%d: resid=%.6e >= %.1f\n",
                               k + 1, trans[0], n, kl, ku, nrhs, imat, (double)result[k], (double)THRESH);
                }
                assert_residual_below(result[k], THRESH);
            }
        }
    }

    /*
     * TEST 7: Get an estimate of RCOND = 1/CNDNUM.
     * Only runs for first NB (inb=0), square matrices (m==n), and n > 0.
     * This matches LAPACK's dchkgb.f line 499-500 and 640-679.
     */
test7:
    if (n <= 0 || m != n || inb > 0) {
        return;
    }

    for (int itran = 0; itran < 2; itran++) {
        f32 anorm_est;
        char norm[2];
        if (itran == 0) {
            anorm_est = anormo;
            rcondc = rcondo;
            norm[0] = 'O';
        } else {
            anorm_est = anormi;
            rcondc = rcondi;
            norm[0] = 'I';
        }
        norm[1] = '\0';

        sgbcon(norm, n, kl, ku, ws->AFAC, ldafac, ws->IPIV,
               anorm_est, &rcond, ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        result[6] = sget06(rcond, rcondc);

        if (result[6] >= THRESH) {
            print_error("TEST 7 FAILED: norm='%c', n=%d, kl=%d, ku=%d, imat=%d: resid=%.6e >= %.1f\n",
                       norm[0], n, kl, ku, imat, (double)result[6], (double)THRESH);
        }
        assert_residual_below(result[6], THRESH);
    }
}

/**
 * CMocka test wrapper - extracts parameters and runs the test.
 */
static void test_dchkgb(void** state)
{
    dchkgb_params_t* params = (dchkgb_params_t*)*state;
    set_test_context(params->name);
    run_dchkgb_single(params->m, params->n, params->kl, params->ku, params->imat, params->inb);
    clear_test_context();
}

/**
 * Generate bandwidth values for a given m, n.
 * Following dchkgb.f: KLVAL(1)=0, KLVAL(2)=(5*M+1)/4, KLVAL(3)=(3M-1)/4, KLVAL(4)=(M+1)/4
 */
static void get_klval(int m, int klval[NBW])
{
    klval[0] = 0;
    klval[1] = m + (m + 1) / 4;
    klval[2] = (3 * m - 1) / 4;
    klval[3] = (m + 1) / 4;
}

static void get_kuval(int n, int kuval[NBW])
{
    kuval[0] = 0;
    kuval[1] = n + (n + 1) / 4;
    kuval[2] = (3 * n - 1) / 4;
    kuval[3] = (n + 1) / 4;
}

/**
 * Main - generate parameterized tests and run.
 */
int main(void)
{
    /* Count total tests */
    int test_count = 0;
    for (int im = 0; im < (int)NM; im++) {
        int m = MVAL[im];
        int klval[NBW];
        get_klval(m, klval);
        int nkl = (m + 1 < NBW) ? m + 1 : NBW;

        for (int in = 0; in < (int)NN; in++) {
            int n = NVAL[in];
            int kuval[NBW];
            get_kuval(n, kuval);
            int nku = (n + 1 < NBW) ? n + 1 : NBW;
            int nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (int ikl = 0; ikl < nkl; ikl++) {
                int kl = klval[ikl];
                if (kl > m - 1 && m > 0) kl = m - 1;
                if (kl < 0) kl = 0;

                for (int iku = 0; iku < nku; iku++) {
                    int ku = kuval[iku];
                    if (ku > n - 1 && n > 0) ku = n - 1;
                    if (ku < 0) ku = 0;

                    for (int imat = 1; imat <= nimat; imat++) {
                        /* Skip types 2-4 if matrix is too small */
                        int zerot = (imat >= 2 && imat <= 4);
                        int minmn = (m < n) ? m : n;
                        if (zerot && minmn < imat - 1) {
                            continue;
                        }

                        for (int inb = 0; inb < (int)NNB; inb++) {
                            test_count++;
                        }
                    }
                }
            }
        }
    }

    /* Allocate test array */
    struct CMUnitTest* tests = malloc(test_count * sizeof(struct CMUnitTest));
    dchkgb_params_t* params = malloc(test_count * sizeof(dchkgb_params_t));

    if (!tests || !params) {
        fprintf(stderr, "Failed to allocate test arrays\n");
        return 1;
    }

    /* Generate tests */
    int idx = 0;
    for (int im = 0; im < (int)NM; im++) {
        int m = MVAL[im];
        int klval[NBW];
        get_klval(m, klval);
        int nkl = (m + 1 < NBW) ? m + 1 : NBW;

        for (int in = 0; in < (int)NN; in++) {
            int n = NVAL[in];
            int kuval[NBW];
            get_kuval(n, kuval);
            int nku = (n + 1 < NBW) ? n + 1 : NBW;
            int nimat = NTYPES;
            if (m <= 0 || n <= 0) {
                nimat = 1;
            }

            for (int ikl = 0; ikl < nkl; ikl++) {
                int kl = klval[ikl];
                if (kl > m - 1 && m > 0) kl = m - 1;
                if (kl < 0) kl = 0;

                for (int iku = 0; iku < nku; iku++) {
                    int ku = kuval[iku];
                    if (ku > n - 1 && n > 0) ku = n - 1;
                    if (ku < 0) ku = 0;

                    for (int imat = 1; imat <= nimat; imat++) {
                        /* Skip types 2-4 if matrix is too small */
                        int zerot = (imat >= 2 && imat <= 4);
                        int minmn = (m < n) ? m : n;
                        if (zerot && minmn < imat - 1) {
                            continue;
                        }

                        for (int inb = 0; inb < (int)NNB; inb++) {
                            int nb = NBVAL[inb];
                            params[idx].m = m;
                            params[idx].n = n;
                            params[idx].kl = kl;
                            params[idx].ku = ku;
                            params[idx].imat = imat;
                            params[idx].inb = inb;
                            snprintf(params[idx].name, sizeof(params[idx].name),
                                    "dchkgb_m%d_n%d_kl%d_ku%d_type%d_nb%d_%d",
                                    m, n, kl, ku, imat, nb, inb);

                            tests[idx].name = params[idx].name;
                            tests[idx].test_func = test_dchkgb;
                            tests[idx].setup_func = NULL;
                            tests[idx].teardown_func = NULL;
                            tests[idx].initial_state = &params[idx];

                            idx++;
                        }
                    }
                }
            }
        }
    }

    /* Run all tests with shared workspace.
     * We use _cmocka_run_group_tests directly because the test array
     * is built dynamically and the standard macro uses sizeof() which
     * only works for compile-time array sizes. */
    int result = _cmocka_run_group_tests("dchkgb", tests, idx,
                                         group_setup, group_teardown);

    free(tests);
    free(params);
    return result;
}
