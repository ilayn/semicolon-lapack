/**
 * @file test_dchklqt.c
 * @brief Comprehensive test suite for blocked LQ factorization (LQT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchklqt.f to C using CMocka.
 * Tests DGELQT and DGEMLQT.
 *
 * Test structure from dchklqt.f:
 *   TEST 1: |A - L*Q| / (eps * max(1,M) * |A|)
 *   TEST 2: |I - Q*Q'| / (eps * max(1,N))
 *   TEST 3: |Q*C - Q*C| / (eps * max(1,M) * |C|)
 *   TEST 4: |Q'*C - Q'*C| / (eps * max(1,M) * |C|)
 *   TEST 5: |C*Q - C*Q| / (eps * max(1,M) * |C|)
 *   TEST 6: |C*Q' - C*Q'| / (eps * max(1,M) * |C|)
 *
 * Parameters:
 *   M values: 0, 1, 2, 3, 5, 10, 16, 50
 *   N values: 0, 1, 2, 3, 5, 10, 16, 50
 *   NB values: 1, 2, 3, 5, 10, 16
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include <string.h>
#include <stdio.h>

static const int MVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const int NBVAL[] = {1, 2, 3, 5, 10, 16};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTESTS  6
#define THRESH  30.0

extern void dlqt04(const int m, const int n, const int nb, double* restrict result);

typedef struct {
    int m;
    int n;
    int nb;
    char name[64];
} dchklqt_params_t;

static void run_dchklqt_single(int m, int n, int nb)
{
    double result[NTESTS];
    char ctx[128];
    int minmn = (m < n) ? m : n;

    if (nb > minmn || nb <= 0 || minmn == 0) {
        return;
    }

    dlqt04(m, n, nb, result);

    for (int t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "m=%d n=%d nb=%d TEST %d", m, n, nb, t + 1);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }

    clear_test_context();
}

static void test_dchklqt_case(void** state)
{
    dchklqt_params_t* params = *state;
    run_dchklqt_single(params->m, params->n, params->nb);
}

#define MAX_TESTS (NM * NN * NNB)

static dchklqt_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (int im = 0; im < (int)NM; im++) {
        int m = MVAL[im];

        for (int in = 0; in < (int)NN; in++) {
            int n = NVAL[in];
            int minmn = (m < n) ? m : n;

            for (int inb = 0; inb < (int)NNB; inb++) {
                int nb = NBVAL[inb];

                if (nb > minmn || nb <= 0 || minmn == 0) {
                    continue;
                }

                dchklqt_params_t* p = &g_params[g_num_tests];
                p->m = m;
                p->n = n;
                p->nb = nb;
                snprintf(p->name, sizeof(p->name), "dchklqt_m%d_n%d_nb%d", m, n, nb);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dchklqt_case;
                g_tests[g_num_tests].setup_func = NULL;
                g_tests[g_num_tests].teardown_func = NULL;
                g_tests[g_num_tests].initial_state = p;

                g_num_tests++;
            }
        }
    }
}

int main(void)
{
    build_test_array();

    if (g_num_tests == 0) {
        printf("No valid test cases generated\n");
        return 0;
    }

    return _cmocka_run_group_tests("dchklqt", g_tests, g_num_tests, NULL, NULL);
}
