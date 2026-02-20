/**
 * @file test_schklqtp.c
 * @brief Comprehensive test suite for triangular-pentagonal LQ (LQTP) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchklqtp.f to C using CMocka.
 * Tests STPLQT and STPMLQT.
 *
 * Test structure from dchklqtp.f:
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
#include <string.h>
#include <stdio.h>

static const int MVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const int NVAL[] = {0, 1, 2, 3, 5, 10, 16, 50};
static const int NBVAL[] = {1, 2, 3, 5, 10, 16};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTESTS  6
#define THRESH  30.0f

extern void slqt05(const int m, const int n, const int l, const int nb,
                   f32* restrict result);

typedef struct {
    int m;
    int n;
    int l;
    int nb;
    char name[64];
} dchklqtp_params_t;

static void run_dchklqtp_single(int m, int n, int l, int nb)
{
    f32 result[NTESTS];
    char ctx[128];

    // Skip invalid parameter combinations:
    // - nb > m: stplqt requires mb <= m
    // - nb <= 0: invalid block size
    // - m == 0: trivial case
    // - n > 0 && nb > n: slqt05.f documents "NB <= N" constraint (line 53)
    if (nb > m || nb <= 0 || m == 0 || (n > 0 && nb > n)) {
        return;
    }

    slqt05(m, n, l, nb, result);

    for (int t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "m=%d n=%d l=%d nb=%d TEST %d resid=%.3e", m, n, l, nb, t + 1, result[t]);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }

    clear_test_context();
}

static void test_dchklqtp_case(void** state)
{
    dchklqtp_params_t* params = *state;
    run_dchklqtp_single(params->m, params->n, params->l, params->nb);
}

#define MAX_TESTS (NM * NN * NNB * 3)

static dchklqtp_params_t g_params[MAX_TESTS];
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
            int step = (minmn > 0) ? minmn : 1;

            for (int l = 0; l <= minmn; l += step) {
                for (int inb = 0; inb < (int)NNB; inb++) {
                    int nb = NBVAL[inb];

                    // Skip invalid parameter combinations
                    if (nb > m || nb <= 0 || m == 0 || (n > 0 && nb > n)) {
                        continue;
                    }

                    dchklqtp_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->l = l;
                    p->nb = nb;
                    snprintf(p->name, sizeof(p->name), "dchklqtp_m%d_n%d_l%d_nb%d",
                             m, n, l, nb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchklqtp_case;
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

    return _cmocka_run_group_tests("dchklqtp", g_tests, g_num_tests, NULL, NULL);
}
