/**
 * @file test_zdrvac.c
 * @brief Test suite for ZCPOSV mixed-precision iterative refinement solver.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zdrvac.f to C using CMocka.
 * Tests ZCPOSV which uses single precision complex Cholesky factorization with
 * c128 precision iterative refinement for Hermitian positive definite systems.
 *
 * Test parameters from zchkab.f data file:
 *   M values: 0, 1, 2, 3, 5, 10, 16
 *   NRHS values: 2
 *   Matrix types: 1-9 (positive definite types)
 *   THRESH: 20.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
/* Test parameters */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 16};
static const INT NSVAL[] = {2};  /* NRHS values */
static const char UPLOS[] = {'U', 'L'};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NUPLO   (sizeof(UPLOS) / sizeof(UPLOS[0]))
#define NTYPES  9  /* Positive definite matrix types */
#define THRESH  20.0
#define NMAX    132
#define MAXRHS  16

/**
 * Workspace for test execution.
 */
typedef struct {
    c128* A;          /* Original matrix (NMAX x NMAX) */
    c128* AFAC;       /* Copy of A for factorization (NMAX x NMAX) */
    c128* B;          /* Right-hand side (NMAX x MAXRHS) */
    c128* X;          /* Solution (NMAX x MAXRHS) */
    c128* WORK;       /* Double complex workspace */
    f64* RWORK;       /* Real workspace */
    c64* SWORK;       /* Single complex workspace */
} zdrvac_workspace_t;

static zdrvac_workspace_t* g_ws = NULL;

/**
 * Group setup - allocate workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrvac_workspace_t));
    if (!g_ws) return -1;

    g_ws->A = malloc(NMAX * NMAX * sizeof(c128));
    g_ws->AFAC = malloc(NMAX * NMAX * sizeof(c128));
    g_ws->B = malloc(NMAX * MAXRHS * sizeof(c128));
    g_ws->X = malloc(NMAX * MAXRHS * sizeof(c128));
    g_ws->WORK = malloc(NMAX * MAXRHS * 2 * sizeof(c128));
    g_ws->RWORK = malloc(2 * NMAX * sizeof(f64));
    g_ws->SWORK = malloc(NMAX * (NMAX + MAXRHS) * sizeof(c64));

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
static void run_test_single(INT n, INT nrhs, INT imat, char uplo)
{
    char type, dist;
    INT kl, ku, mode, info, iter;
    f64 anorm, cndnum;
    INT lda = (n > 1) ? n : 1;
    INT izero = 0;
    const f64 ZERO = 0.0;

    uint64_t rng_state[4];
    rng_seed(rng_state, 1988 + n * 1000 + imat * 100 + nrhs + (uplo == 'U' ? 0 : 50));

    char uplo_str[2] = {uplo, '\0'};

    zlatb4("ZPO", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char dist_str[2] = {dist, '\0'};
    zlatms(n, n, dist_str, "P", g_ws->RWORK, mode, cndnum, anorm,
           kl, ku, uplo_str, g_ws->A, lda, g_ws->WORK, &info, rng_state);

    if (info != 0) {
        return;
    }

    INT zerot = (imat >= 3 && imat <= 5);
    if (zerot) {
        if (imat == 3) {
            izero = 1;
        } else if (imat == 4) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }

        INT ioff = (izero - 1) * lda;

        if (uplo == 'U' || uplo == 'u') {
            for (INT i = 0; i < izero - 1; i++) {
                g_ws->A[ioff + i] = ZERO;
            }
            ioff = ioff + izero - 1;
            for (INT i = izero - 1; i < n; i++) {
                g_ws->A[ioff] = ZERO;
                ioff = ioff + lda;
            }
        } else {
            ioff = izero - 1;
            for (INT i = 0; i < izero - 1; i++) {
                g_ws->A[ioff] = ZERO;
                ioff = ioff + lda;
            }
            ioff = ioff - (izero - 1);
            for (INT i = izero - 1; i < n; i++) {
                g_ws->A[ioff + i] = ZERO;
            }
        }
    }

    zlaipd(n, g_ws->A, lda + 1, 0);

    zlarhs("ZPO", "N", uplo_str, " ", n, n, kl, ku, nrhs,
           g_ws->A, lda, g_ws->X, lda, g_ws->B, lda, &info, rng_state);

    zlacpy("A", n, n, g_ws->A, lda, g_ws->AFAC, lda);

    zcposv(uplo_str, n, nrhs, g_ws->AFAC, lda, g_ws->B, lda,
           g_ws->X, lda, g_ws->WORK, g_ws->SWORK, g_ws->RWORK,
           &iter, &info);

    if (iter < 0) {
        zlacpy("A", n, n, g_ws->A, lda, g_ws->AFAC, lda);
    }

    if (info != izero && izero != 0) {
        char context[128];
        snprintf(context, sizeof(context),
                 "zdrvac uplo=%c n=%d nrhs=%d imat=%d: INFO=%d, expected=%d",
                 uplo, n, nrhs, imat, info, izero);
        set_test_context(context);
        assert_int_equal(info, izero);
        clear_test_context();
        return;
    }

    if (info != 0) {
        return;
    }

    zlacpy("A", n, nrhs, g_ws->B, lda, g_ws->WORK, lda);

    f64 resid;
    zpot06(uplo_str, n, nrhs, g_ws->A, lda, g_ws->X, lda,
           g_ws->WORK, lda, g_ws->RWORK, &resid);

    char context[128];
    snprintf(context, sizeof(context),
             "zdrvac uplo=%c n=%d nrhs=%d imat=%d iter=%d",
             uplo, n, nrhs, imat, iter);
    set_test_context(context);

    if (iter >= 0 && n > 0) {
        f64 thresh_iter = sqrt((f64)n);
        assert_residual_below(resid, thresh_iter);
    } else {
        assert_residual_ok(resid);
    }

    clear_test_context();
}

/**
 * Test ZCPOSV for all matrix types at given size.
 */
static void test_zcposv(void** state)
{
    (void)state;

    for (size_t im = 0; im < NM; im++) {
        INT n = MVAL[im];
        INT nimat = NTYPES;

        if (n <= 0) {
            nimat = 1;
        }

        for (INT imat = 1; imat <= nimat; imat++) {
            INT zerot = (imat >= 3 && imat <= 5);
            if (zerot && n < imat - 2) {
                continue;
            }

            for (size_t iuplo = 0; iuplo < NUPLO; iuplo++) {
                char uplo = UPLOS[iuplo];

                for (size_t irhs = 0; irhs < NNS; irhs++) {
                    INT nrhs = NSVAL[irhs];
                    run_test_single(n, nrhs, imat, uplo);
                }
            }
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zcposv),
    };

    return cmocka_run_group_tests_name("zdrvac", tests, group_setup, group_teardown);
}
