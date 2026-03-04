/**
 * @file test_zchkqrtp.c
 * @brief Test suite for ZTPQRT and ZTPMQRT routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchkqrtp.f to C using CMocka.
 * Tests ZTPQRT and ZTPMQRT.
 *
 * Each (m, n, l, nb) combination is registered as a separate CMocka test,
 * providing pytest-like parameterized test behavior with clear failure isolation.
 *
 * Parameters from ztest.in:
 *   M values: 0, 1, 2, 3, 5, 10, 50
 *   N values: 0, 1, 2, 3, 5, 10, 50
 *   L values: 0 and MIN(M,N)
 *   NB values: 1, 3, 3, 3, 20
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include <stdio.h>

static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 50};
static const INT NBVAL[] = {1, 3, 3, 3, 20};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTESTS  6
#define THRESH  30.0

typedef struct {
    INT m;
    INT n;
    INT l;
    INT inb;
    char name[80];
} zchkqrtp_params_t;

static void run_zchkqrtp_single(INT m, INT n, INT l, INT nb)
{
    f64 result[NTESTS];
    char ctx[128];

    if (nb > n || nb <= 0) return;

    zqrt05(m, n, l, nb, result);

    for (INT t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "m=%d n=%d l=%d nb=%d test(%d)", m, n, l, nb, t + 1);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }
    clear_test_context();
}

static void test_zchkqrtp_case(void** state)
{
    zchkqrtp_params_t* params = *state;
    run_zchkqrtp_single(params->m, params->n, params->l, NBVAL[params->inb]);
}

/* L takes values 0 and MIN(M,N), so at most 2 L-values per (M,N) pair */
#define MAX_TESTS (NM * NN * 2 * NNB)

static zchkqrtp_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT im = 0; im < (INT)NM; im++) {
        INT m = MVAL[im];
        for (INT in = 0; in < (INT)NN; in++) {
            INT n = NVAL[in];
            INT minmn = (m < n) ? m : n;

            /* L loop: DO L = 0, MINMN, MAX(MINMN, 1)
             * This gives L = 0 always, and L = MINMN when MINMN > 0 */
            INT lstep = (minmn > 1) ? minmn : 1;
            for (INT l = 0; l <= minmn; l += lstep) {
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    zchkqrtp_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->l = l;
                    p->inb = inb;
                    snprintf(p->name, sizeof(p->name),
                             "zchkqrtp_m%d_n%d_l%d_nb%d_%d",
                             m, n, l, nb, inb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchkqrtp_case;
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
    build_test_array();
    return _cmocka_run_group_tests("zchkqrtp", g_tests, g_num_tests,
                                   NULL, NULL);
}
