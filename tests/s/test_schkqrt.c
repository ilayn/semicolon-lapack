/**
 * @file test_schkqrt.c
 * @brief Comprehensive test suite for blocked QR factorization (QRT) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkqrt.f to C using CMocka.
 * Tests SGEQRT and SGEMQRT.
 *
 * Test structure from dchkqrt.f:
 *   TEST 1: |A - Q*R| / (eps * max(1,M) * |A|)
 *   TEST 2: |I - Q'*Q| / (eps * max(1,M))
 *   TEST 3: |Q*C - Q*C| / (eps * max(1,M) * |C|)
 *   TEST 4: |Q'*C - Q'*C| / (eps * max(1,M) * |C|)
 *   TEST 5: |C*Q - C*Q| / (eps * max(1,M) * |D|)
 *   TEST 6: |C*Q' - C*Q'| / (eps * max(1,M) * |D|)
 *
 * Parameters:
 *   M values: 0, 1, 2, 3, 5, 10, 16, 50
 *   N values: 0, 1, 2, 3, 5, 10, 16, 50
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
#define THRESH  30.0f

typedef struct {
    INT m;
    INT n;
    INT nb;
    char name[64];
} dchkqrt_params_t;

static void run_dchkqrt_single(INT m, INT n, INT nb)
{
    f32 result[NTESTS];
    char ctx[128];
    INT minmn = (m < n) ? m : n;

    if (nb > minmn || nb <= 0 || minmn == 0) {
        return;
    }

    sqrt04(m, n, nb, result);

    for (INT t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "m=%d n=%d nb=%d TEST %d", m, n, nb, t + 1);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }

    clear_test_context();
}

static void test_dchkqrt_case(void** state)
{
    dchkqrt_params_t* params = *state;
    run_dchkqrt_single(params->m, params->n, params->nb);
}

#define MAX_TESTS (NM * NN * NNB)

static dchkqrt_params_t g_params[MAX_TESTS];
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

            for (INT inb = 0; inb < (INT)NNB; inb++) {
                INT nb = NBVAL[inb];

                if (nb > minmn || nb <= 0 || minmn == 0) {
                    continue;
                }

                dchkqrt_params_t* p = &g_params[g_num_tests];
                p->m = m;
                p->n = n;
                p->nb = nb;
                snprintf(p->name, sizeof(p->name), "dchkqrt_m%d_n%d_nb%d", m, n, nb);

                g_tests[g_num_tests].name = p->name;
                g_tests[g_num_tests].test_func = test_dchkqrt_case;
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

    return _cmocka_run_group_tests("dchkqrt", g_tests, g_num_tests, NULL, NULL);
}
