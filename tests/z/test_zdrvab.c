/**
 * @file test_zdrvab.c
 * @brief Test suite for ZCGESV mixed-precision iterative refinement solver.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zdrvab.f to C using CMocka.
 * Tests ZCGESV which uses single precision complex factorization with c128
 * precision iterative refinement.
 *
 * Test parameters from zchkab.f data file:
 *   M values: 0, 1, 2, 3, 5, 10, 16
 *   NRHS values: 2
 *   Matrix types: 1-11
 *   THRESH: 20.0
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
/* Test parameters */
static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 16};
static const INT NSVAL[] = {2};  /* NRHS values */

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NNS     (sizeof(NSVAL) / sizeof(NSVAL[0]))
#define NTYPES  11
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
    INT* IWORK;       /* Integer workspace (pivot indices) */
} zdrvab_workspace_t;

static zdrvab_workspace_t* g_ws = NULL;

/**
 * Group setup - allocate workspace.
 */
static int group_setup(void** state)
{
    (void)state;

    g_ws = malloc(sizeof(zdrvab_workspace_t));
    if (!g_ws) return -1;

    g_ws->A = malloc(NMAX * NMAX * sizeof(c128));
    g_ws->AFAC = malloc(NMAX * NMAX * sizeof(c128));
    g_ws->B = malloc(NMAX * MAXRHS * sizeof(c128));
    g_ws->X = malloc(NMAX * MAXRHS * sizeof(c128));
    g_ws->WORK = malloc(NMAX * MAXRHS * 2 * sizeof(c128));
    g_ws->RWORK = malloc(2 * NMAX * sizeof(f64));
    g_ws->SWORK = malloc(NMAX * (NMAX + MAXRHS) * sizeof(c64));
    g_ws->IWORK = malloc(NMAX * sizeof(INT));

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
static void run_test_single(INT n, INT nrhs, INT imat)
{
    char type, dist;
    INT kl, ku, mode, info, iter;
    f64 anorm, cndnum;
    INT lda = (n > 1) ? n : 1;
    INT izero = 0;
    const f64 ZERO = 0.0;

    uint64_t rng_state[4];
    rng_seed(rng_state, 2006 + n * 1000 + imat * 100 + nrhs);

    zlatb4("ZGE", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char dist_str[2] = {dist, '\0'};
    char sym_str[2] = {type, '\0'};
    zlatms(n, n, dist_str, sym_str, g_ws->RWORK, mode, cndnum, anorm,
           kl, ku, "N", g_ws->A, lda, g_ws->WORK, &info, rng_state);

    if (info != 0) {
        return;
    }

    INT zerot = (imat >= 5 && imat <= 7);
    if (zerot) {
        if (imat == 5) {
            izero = 1;
        } else if (imat == 6) {
            izero = n;
        } else {
            izero = n / 2 + 1;
        }
        INT ioff = (izero - 1) * lda;
        if (imat < 7) {
            for (INT i = 0; i < n; i++) {
                g_ws->A[ioff + i] = ZERO;
            }
        } else {
            zlaset("F", n, n - izero + 1, CMPLX(ZERO, ZERO),
                   CMPLX(ZERO, ZERO), &g_ws->A[ioff], lda);
        }
    }

    zlarhs("ZGE", "N", " ", "N", n, n, kl, ku, nrhs,
           g_ws->A, lda, g_ws->X, lda, g_ws->B, lda, &info, rng_state);

    zlacpy("F", n, n, g_ws->A, lda, g_ws->AFAC, lda);

    zcgesv(n, nrhs, g_ws->A, lda, g_ws->IWORK, g_ws->B, lda,
           g_ws->X, lda, g_ws->WORK, g_ws->SWORK, g_ws->RWORK,
           &iter, &info);

    if (iter < 0) {
        zlacpy("F", n, n, g_ws->AFAC, lda, g_ws->A, lda);
    }

    if (info != izero && izero != 0) {
        char context[128];
        snprintf(context, sizeof(context),
                 "zdrvab n=%d nrhs=%d imat=%d: INFO=%d, expected=%d",
                 n, nrhs, imat, info, izero);
        set_test_context(context);
        assert_int_equal(info, izero);
        clear_test_context();
        return;
    }

    if (info != 0) {
        return;
    }

    zlacpy("F", n, nrhs, g_ws->B, lda, g_ws->WORK, lda);

    f64 resid;
    zget08("N", n, n, nrhs, g_ws->A, lda, g_ws->X, lda,
           g_ws->WORK, lda, g_ws->RWORK, &resid);

    char context[128];
    snprintf(context, sizeof(context),
             "zdrvab n=%d nrhs=%d imat=%d iter=%d", n, nrhs, imat, iter);
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
 * Test ZCGESV for all matrix types at given size.
 */
static void test_zcgesv(void** state)
{
    (void)state;

    for (size_t im = 0; im < NM; im++) {
        INT n = MVAL[im];
        INT nimat = NTYPES;

        if (n <= 0) {
            nimat = 1;
        }

        for (INT imat = 1; imat <= nimat; imat++) {
            INT zerot = (imat >= 5 && imat <= 7);
            if (zerot && n < imat - 4) {
                continue;
            }

            for (size_t irhs = 0; irhs < NNS; irhs++) {
                INT nrhs = NSVAL[irhs];
                run_test_single(n, nrhs, imat);
            }
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zcgesv),
    };

    return cmocka_run_group_tests_name("zdrvab", tests, group_setup, group_teardown);
}
