/**
 * @file test_schkqrtp.c
 * @brief Comprehensive test suite for triangular-pentagonal QR (QRTP) routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkqrtp.f to C using CMocka.
 * Tests STPQRT and STPMQRT.
 *
 * Test structure from dchkqrtp.f:
 *   TEST 1: |A - Q*R| / (eps * max(1,M2) * |A|) where M2 = M+N
 *   TEST 2: |I - Q'*Q| / (eps * max(1,M2))
 *   TEST 3: |Q*C - Q*C| / (eps * max(1,M2) * |C|)
 *   TEST 4: |Q'*C - Q'*C| / (eps * max(1,M2) * |C|)
 *   TEST 5: |C*Q - C*Q| / (eps * max(1,M2) * |D|)
 *   TEST 6: |C*Q' - C*Q'| / (eps * max(1,M2) * |D|)
 *
 * Parameters:
 *   M values: 0, 1, 2, 3, 5, 10, 16, 50
 *   N values: 0, 1, 2, 3, 5, 10, 16, 50
 *   NB values: 1, 2, 3, 5, 10, 16
 *   L values: 0, min(M,N) (and intermediate if min(M,N) > 1)
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
    INT l;
    INT nb;
    char name[64];
} dchkqrtp_params_t;

static void run_dchkqrtp_single(INT m, INT n, INT l, INT nb)
{
    f32 result[NTESTS];
    char ctx[128];

    if (nb > n || nb <= 0) {
        return;
    }

    sqrt05(m, n, l, nb, result);

    for (INT t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "m=%d n=%d l=%d nb=%d TEST %d", m, n, l, nb, t + 1);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }

    clear_test_context();
}

static void test_dchkqrtp_case(void** state)
{
    dchkqrtp_params_t* params = *state;
    run_dchkqrtp_single(params->m, params->n, params->l, params->nb);
}

#define MAX_TESTS (NM * NN * NNB * 3)

static dchkqrtp_params_t g_params[MAX_TESTS];
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

            for (INT l = 0; l <= minmn; l += (minmn > 0 ? minmn : 1)) {

                for (INT inb = 0; inb < (INT)NNB; inb++) {
                    INT nb = NBVAL[inb];

                    if (nb > n || nb <= 0) {
                        continue;
                    }

                    dchkqrtp_params_t* p = &g_params[g_num_tests];
                    p->m = m;
                    p->n = n;
                    p->l = l;
                    p->nb = nb;
                    snprintf(p->name, sizeof(p->name), "dchkqrtp_m%d_n%d_l%d_nb%d",
                             m, n, l, nb);

                    g_tests[g_num_tests].name = p->name;
                    g_tests[g_num_tests].test_func = test_dchkqrtp_case;
                    g_tests[g_num_tests].setup_func = NULL;
                    g_tests[g_num_tests].teardown_func = NULL;
                    g_tests[g_num_tests].initial_state = p;

                    g_num_tests++;

                    if (g_num_tests >= (INT)MAX_TESTS) {
                        return;
                    }
                }

                if (minmn == 0) break;
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

    return _cmocka_run_group_tests("dchkqrtp", g_tests, g_num_tests, NULL, NULL);
}
