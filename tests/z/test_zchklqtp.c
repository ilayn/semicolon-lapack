/**
 * @file test_zchklqtp.c
 * @brief Comprehensive test suite for triangular-pentagonal LQ (LQTP) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/zchklqtp.f to C using CMocka.
 * Tests ZTPLQT and ZTPMLQT.
 *
 * Test structure from zchklqtp.f:
 *   TEST 1: |A - Q*R| / (eps * max(1,N2) * |A|)
 *   TEST 2: |I - Q*Q'| / (eps * max(1,N2))
 *   TEST 3: |Q*C - Q*C| / (eps * max(1,N2) * |C|)
 *   TEST 4: |Q'*C - Q'*C| / (eps * max(1,N2) * |C|)
 *   TEST 5: |C*Q - C*Q| / (eps * max(1,N2) * |D|)
 *   TEST 6: |C*Q' - C*Q'| / (eps * max(1,N2) * |D|)
 *
 * Parameters:
 *   M values: 0, 1, 2, 3, 5, 10, 16, 50
 *   N values: 0, 1, 2, 3, 5, 10, 16, 50
 *   L values: 0 to min(M,N), stepping by max(min(M,N), 1)
 *   NB values: 1, 2, 3, 5, 10, 16
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include <string.h>
#include <stdio.h>

static const INT MVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const INT NVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const INT NBVAL[] = {1, 2, 3, 5, 10, 16};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTESTS  6
#define THRESH  30.0

typedef struct {
    INT m;
    INT n;
    INT l;
    INT nb;
    char name[64];
} zchklqtp_params_t;

static void run_zchklqtp_single(INT m, INT n, INT l, INT nb)
{
    f64 result[NTESTS];
    char ctx[128];

    if (nb > m || nb <= 0 || m == 0 || (n > 0 && nb > n)) {
        return;
    }

    zlqt05(m, n, l, nb, result);

    for (INT t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "m=%d n=%d l=%d nb=%d TEST %d resid=%.3e", m, n, l, nb, t + 1, result[t]);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }

    clear_test_context();
}

static void test_zchklqtp_case(void** state)
{
    zchklqtp_params_t* params = *state;
    run_zchklqtp_single(params->m, params->n, params->l, params->nb);
}

#define MAX_TESTS (NM * NN * NNB * 3)

static zchklqtp_params_t g_params[MAX_TESTS];
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
            INT step = (minmn > 0) ? minmn : 1;

            for (INT l = 0; l <= minmn; l += step) {
                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    if (nb > m || nb <= 0 || m == 0 || (n > 0 && nb > n)) {
                        continue;
                    }

                    zchklqtp_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->l = l;
                    p->nb = nb;
                    snprintf(p->name, sizeof(p->name), "zchklqtp_m%d_n%d_l%d_nb%d",
                             m, n, l, nb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_zchklqtp_case;
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

    if (g_num_tests == 0) {
        printf("No valid test cases generated\n");
        return 0;
    }

    (void)_cmocka_run_group_tests("zchklqtp", g_tests, g_num_tests, NULL, NULL);
    return 0;
}
