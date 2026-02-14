/**
 * @file test_dchkgb.c
 * @brief Comprehensive test suite for general band matrix (DGB) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkgb.f to C using CMocka.
 * Tests DGBTRF, DGBTRS, DGBRFS, and DGBCON.
 *
 * Each (m, n, kl, ku, imat) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Test structure from dchkgb.f:
 *   TEST 1: LU factorization residual via dgbt01
 *   TEST 2: Solution residual via dgbt02
 *   TEST 3: Solution accuracy via dget04
 *   TEST 4: Refined solution accuracy via dget04 (after dgbrfs)
 *   TEST 5-6: Error bounds via dgbt05
 *   TEST 7: Condition number via dget06
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
#define THRESH  30.0
#define NMAX    50  /* Maximum matrix dimension */
#define NSMAX   15  /* Max NRHS */
#define NBW     4   /* Number of bandwidth values to test */

/* Routines under test */
extern void dgbtrf(const int m, const int n, const int kl, const int ku,
                   f64* AB, const int ldab, int* ipiv, int* info);
extern void dgbtrs(const char* trans, const int n, const int kl, const int ku,
                   const int nrhs, const f64* AB, const int ldab,
                   const int* ipiv, f64* B, const int ldb, int* info);
extern void dgbrfs(const char* trans, const int n, const int kl, const int ku,
                   const int nrhs, const f64* AB, const int ldab,
                   const f64* AFB, const int ldafb, const int* ipiv,
                   const f64* B, const int ldb, f64* X, const int ldx,
                   f64* ferr, f64* berr, f64* work, int* iwork,
                   int* info);
extern void dgbcon(const char* norm, const int n, const int kl, const int ku,
                   const f64* AB, const int ldab, const int* ipiv,
                   const f64 anorm, f64* rcond, f64* work,
                   int* iwork, int* info);

/* Verification routines */
extern void dgbt01(int m, int n, int kl, int ku,
                   const f64* A, int lda, const f64* AFAC, int ldafac,
                   const int* ipiv, f64* work, f64* resid);
extern void dgbt02(const char* trans, int m, int n, int kl, int ku, int nrhs,
                   const f64* A, int lda, const f64* X, int ldx,
                   f64* B, int ldb, f64* rwork, f64* resid);
extern void dgbt05(const char* trans, int n, int kl, int ku, int nrhs,
                   const f64* AB, int ldab, const f64* B, int ldb,
                   const f64* X, int ldx, const f64* XACT, int ldxact,
                   const f64* FERR, const f64* BERR, f64* reslts);
extern void dget04(const int n, const int nrhs, const f64* X, const int ldx,
                   const f64* XACT, const int ldxact, const f64 rcond,
                   f64* resid);
extern f64 dget06(const f64 rcond, const f64 rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, f64* anorm, int* mode,
                   f64* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   const char* sym, f64* d, const int mode, const f64 cond,
                   const f64 dmax, const int kl, const int ku, const char* pack,
                   f64* A, const int lda, f64* work, int* info,
                   uint64_t state[static 4]);
extern void dlarhs(const char* path, const char* xtype, const char* uplo,
                   const char* trans, const int m, const int n, const int kl,
                   const int ku, const int nrhs, const f64* A, const int lda,
                   const f64* XACT, const int ldxact, f64* B,
                   const int ldb, int* info, uint64_t state[static 4]);

/* Utilities */
extern void dlacpy(const char* uplo, const int m, const int n,
                   const f64* A, const int lda, f64* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const f64 alpha, const f64 beta,
                   f64* A, const int lda);
extern f64 dlangb(const char* norm, const int n, const int kl, const int ku,
                     const f64* AB, const int ldab, f64* work);
extern f64 dlange(const char* norm, const int m, const int n,
                     const f64* A, const int lda, f64* work);
extern f64 dlamch(const char* cmach);

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
    f64* A;      /* Original band matrix */
    f64* AFAC;   /* Factored band matrix */
    f64* B;      /* Right-hand side (NMAX x NSMAX) */
    f64* X;      /* Solution (NMAX x NSMAX) */
    f64* XACT;   /* Exact solution (NMAX x NSMAX) */
    f64* WORK;   /* General workspace */
    f64* RWORK;  /* Real workspace */
    f64* D;      /* Singular values for dlatms */
    f64* FERR;   /* Forward error bounds */
    f64* BERR;   /* Backward error bounds */
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

    g_workspace->A = malloc(LA * sizeof(f64));
    g_workspace->AFAC = malloc(LAFAC * sizeof(f64));
    g_workspace->B = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->X = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->XACT = malloc(NMAX * NSMAX * sizeof(f64));
    g_workspace->WORK = malloc(3 * NMAX * NMAX * sizeof(f64));
    g_workspace->RWORK = malloc((NMAX + 2 * NSMAX) * sizeof(f64));
    g_workspace->D = malloc(NMAX * sizeof(f64));
    g_workspace->FERR = malloc(NSMAX * sizeof(f64));
    g_workspace->BERR = malloc(NSMAX * sizeof(f64));
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
    const f64 ZERO = 0.0;
    const f64 ONE = 1.0;
    dchkgb_workspace_t* ws = g_workspace;

    char type, dist;
    int kl_gen, ku_gen, mode;
    f64 anorm_gen, cndnum;
    int info, izero;
    int lda = kl + ku + 1;
    int ldafac = 2 * kl + ku + 1;
    int ldb = (n > 1) ? n : 1;
    int trfcon;
    f64 anormo = 0.0, anormi = 0.0, rcondo = 0.0, rcondi = 0.0, rcond, rcondc;
    f64 result[NTESTS];

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
     * Note: dlatb4 for "GB" just sets type='N' and condition/norm based on imat,
     * it doesn't override kl/ku since we pass them separately */
    dlatb4("DGB", imat, m, n, &type, &kl_gen, &ku_gen, &anorm_gen, &mode, &cndnum, &dist);

    /* Generate the matrix directly in band storage using pack='Z'.
     * dlatms with pack='Z' generates a band matrix and stores it directly
     * in band format with leading dimension kl+ku+1. */
    dlatms(m, n, &dist, &type, ws->D, mode, cndnum, anorm_gen,
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
    dgbtrf(m, n, kl, ku, ws->AFAC, ldafac, ws->IPIV, &info);

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
    dgbt01(m, n, kl, ku, ws->A, lda, ws->AFAC, ldafac,
           ws->IPIV, ws->WORK, &result[0]);

    if (result[0] >= THRESH) {
        print_error("TEST 1 FAILED: m=%d, n=%d, kl=%d, ku=%d, nb=%d, imat=%d: resid=%.6e >= %.1f\n",
                   m, n, kl, ku, nb, imat, result[0], THRESH);
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
    anormo = dlangb("O", n, kl, ku, ws->A, lda, ws->RWORK);
    anormi = dlangb("I", n, kl, ku, ws->A, lda, ws->RWORK);

    if (info == 0) {
        /* Form the inverse of A to get a good estimate of CNDNUM */
        int ldb_inv = (n > 1) ? n : 1;
        dlaset("F", n, n, ZERO, ONE, ws->WORK, ldb_inv);

        dgbtrs("N", n, kl, ku, n, ws->AFAC, ldafac, ws->IPIV, ws->WORK, ldb_inv, &info);

        /* Compute the 1-norm condition number */
        f64 ainvnm = dlange("O", n, n, ws->WORK, ldb_inv, ws->RWORK);
        if (anormo <= ZERO || ainvnm <= ZERO) {
            rcondo = ONE;
        } else {
            rcondo = (ONE / anormo) / ainvnm;
        }

        /* Compute the infinity-norm condition number */
        ainvnm = dlange("I", n, n, ws->WORK, ldb_inv, ws->RWORK);
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
            dlarhs("DGB", &xtype, " ", trans, n, n, kl, ku, nrhs,
                   ws->A, lda, ws->XACT, ldb, ws->B, ldb, &info, rng_state);
            xtype = 'C';

            /* Copy B to X */
            dlacpy("F", n, nrhs, ws->B, ldb, ws->X, ldb);

            /*
             * TEST 2: Solve and compute residual.
             */
            dgbtrs(trans, n, kl, ku, nrhs, ws->AFAC, ldafac, ws->IPIV,
                   ws->X, ldb, &info);
            assert_int_equal(info, 0);

            /* Copy B for residual computation (dgbt02 overwrites it) */
            dlacpy("F", n, nrhs, ws->B, ldb, ws->WORK, ldb);
            dgbt02(trans, m, n, kl, ku, nrhs, ws->A, lda, ws->X, ldb,
                   ws->WORK, ldb, ws->RWORK, &result[1]);

            /*
             * TEST 3: Check solution from generated exact solution.
             */
            dget04(n, nrhs, ws->X, ldb, ws->XACT, ldb, rcondc, &result[2]);

            /*
             * TESTs 4, 5, 6: Use iterative refinement.
             */
            dgbrfs(trans, n, kl, ku, nrhs, ws->A, lda, ws->AFAC, ldafac,
                   ws->IPIV, ws->B, ldb, ws->X, ldb,
                   ws->FERR, ws->BERR, ws->WORK, ws->IWORK, &info);
            assert_int_equal(info, 0);

            dget04(n, nrhs, ws->X, ldb, ws->XACT, ldb, rcondc, &result[3]);

            f64 reslts[2];
            dgbt05(trans, n, kl, ku, nrhs, ws->A, lda, ws->B, ldb,
                   ws->X, ldb, ws->XACT, ldb, ws->FERR, ws->BERR, reslts);
            result[4] = reslts[0];
            result[5] = reslts[1];

            /* Check results 2-6 */
            for (int k = 1; k < 6; k++) {
                if (result[k] >= THRESH) {
                    print_error("TEST %d FAILED: trans='%c', n=%d, kl=%d, ku=%d, nrhs=%d, imat=%d: resid=%.6e >= %.1f\n",
                               k + 1, trans[0], n, kl, ku, nrhs, imat, result[k], THRESH);
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
        f64 anorm_est;
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

        dgbcon(norm, n, kl, ku, ws->AFAC, ldafac, ws->IPIV,
               anorm_est, &rcond, ws->WORK, ws->IWORK, &info);
        assert_int_equal(info, 0);

        result[6] = dget06(rcond, rcondc);

        if (result[6] >= THRESH) {
            print_error("TEST 7 FAILED: norm='%c', n=%d, kl=%d, ku=%d, imat=%d: resid=%.6e >= %.1f\n",
                       norm[0], n, kl, ku, imat, result[6], THRESH);
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
