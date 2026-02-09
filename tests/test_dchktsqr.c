/**
 * @file test_dchktsqr.c
 * @brief Comprehensive test suite for tall-skinny QR and short-wide LQ routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchktsqr.f to C using CMocka.
 * Tests DGEQR, DGEMQR (via tall-skinny QR) and DGELQ, DGEMLQ (via short-wide LQ).
 *
 * Each (mode, m, n, mb, nb) combination is registered as a separate CMocka test.
 * The actual test logic is in dtsqr01(), which generates random matrices, factors
 * them, and checks 6 residuals.
 *
 * Test structure from dchktsqr.f:
 *   For 'TS' (tall-skinny) mode:
 *     RESULT(1): | R - Q'*A | / (|A| * eps * max(1,M))
 *     RESULT(2): | I - Q'*Q | / (eps * max(1,M))
 *     RESULT(3): | Q*C - Q*C | / (|C| * eps * max(1,M))
 *     RESULT(4): | Q'*C - Q'*C | / (|C| * eps * max(1,M))
 *     RESULT(5): | C*Q - C*Q | / (|C| * eps * max(1,M))
 *     RESULT(6): | C*Q' - C*Q' | / (|C| * eps * max(1,M))
 *   For 'SW' (short-wide) mode:
 *     Same structure but for LQ factorization
 *
 * Parameters from dtest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   NB values: 1, 3, 3, 3, 20
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include <stdio.h>

/* Test parameters from dtest.in */
static const int MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const int NBVAL[] = {1, 3, 3, 3, 20};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTESTS  6
#define THRESH  30.0

/* Verification routine */
extern void dtsqr01(const char* tssw, const int m, const int n, const int mb,
                    const int nb, double* result);

/**
 * Test parameters for a single test case.
 */
typedef struct {
    int mode;   /* 0='TS', 1='SW' */
    int m;
    int n;
    int imb;    /* Index into NBVAL[] for MB */
    int inb;    /* Index into NBVAL[] for NB */
    char name[80];
} dchktsqr_params_t;

/* Maximum number of test cases: 2 modes * NM * NN * NNB * NNB
 * but only counted when min(m,n) != 0 */
#define MAX_TESTS (2 * NM * NN * NNB * NNB)

static dchktsqr_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

/**
 * CMocka test function - calls dtsqr01 for the given parameter combination.
 */
static void test_dchktsqr_case(void** state)
{
    dchktsqr_params_t* p = *state;
    double result[NTESTS];

    int mb = NBVAL[p->imb];
    int nb = NBVAL[p->inb];

    xlaenv(1, mb);
    xlaenv(2, nb);

    const char* tssw = (p->mode == 0) ? "TS" : "SW";
    dtsqr01(tssw, p->m, p->n, mb, nb, result);

    char ctx[128];
    for (int t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "%s m=%d n=%d mb=%d nb=%d test(%d)",
                 tssw, p->m, p->n, mb, nb, t + 1);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }
    clear_test_context();
}

/**
 * Build the test array with all parameter combinations.
 */
static void build_test_array(void)
{
    g_num_tests = 0;

    for (int mode = 0; mode < 2; mode++) {
        const char* tag = (mode == 0) ? "TS" : "SW";

        for (int im = 0; im < (int)NM; im++) {
            int m = MVAL[im];

            for (int jn = 0; jn < (int)NN; jn++) {
                int n = NVAL[jn];

                if (m == 0 || n == 0) {
                    continue;
                }

                for (int inb = 0; inb < (int)NNB; inb++) {
                    int mb = NBVAL[inb];

                    for (int imb = 0; imb < (int)NNB; imb++) {
                        int nb = NBVAL[imb];

                        dchktsqr_params_t* p = &g_params[g_num_tests];
                        p->mode = mode;
                        p->m = m;
                        p->n = n;
                        p->imb = inb;
                        p->inb = imb;
                        snprintf(p->name, sizeof(p->name),
                                 "%s_m%d_n%d_mb%d_nb%d",
                                 tag, m, n, mb, nb);

                        g_tests[g_num_tests].name = p->name;
                        g_tests[g_num_tests].test_func = test_dchktsqr_case;
                        g_tests[g_num_tests].initial_state = p;
                        g_tests[g_num_tests].setup_func = NULL;
                        g_tests[g_num_tests].teardown_func = NULL;

                        g_num_tests++;
                    }
                }
            }
        }
    }
}

int main(void)
{
    build_test_array();

    return _cmocka_run_group_tests("dchktsqr", g_tests, (size_t)g_num_tests,
                                   NULL, NULL);
}
