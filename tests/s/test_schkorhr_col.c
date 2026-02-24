/**
 * @file test_schkorhr_col.c
 * @brief Comprehensive test suite for Householder reconstruction routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkorhr_col.f to C using CMocka.
 * Tests:
 *   1) SORGTSQR and SORHR_COL using SLATSQR, SGEMQRT (via sorhr_col01)
 *   2) SORGTSQR_ROW and SORHR_COL inside SGETSQRHRT (via sorhr_col02)
 *
 * Test structure from dchkorhr_col.f:
 *   TEST 1: |R - Q'*A| / (eps * M * |A|)
 *   TEST 2: |I - Q'*Q| / (eps * M)
 *   TEST 3: |Q*C - Q*C| / (eps * M * |C|)
 *   TEST 4: |Q'*C - Q'*C| / (eps * M * |C|)
 *   TEST 5: |D*Q - D*Q| / (eps * M * |D|)
 *   TEST 6: |D*Q' - D*Q'| / (eps * M * |D|)
 *
 * Parameters:
 *   M values: 5, 10, 16, 50
 *   N values: 2, 5, 10, 16
 *   MB1 values (must be > N): 16, 32, 64
 *   NB1 values: 1, 2, 5, 10
 *   NB2 values: 1, 2, 5, 10
 *   THRESH: 30.0
 */

#include "test_harness.h"
#include "verify.h"
#include <string.h>
#include <stdio.h>

static const INT MVAL[] = {5, 10, 16, 50};
static const INT NVAL[] = {2, 5, 10, 16};
static const INT NBVAL[] = {1, 2, 5, 10, 16, 32, 64};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTESTS  6
#define THRESH  30.0f

typedef struct {
    INT m;
    INT n;
    INT mb1;
    INT nb1;
    INT nb2;
    INT variant;  /* 1 = sorhr_col01, 2 = sorhr_col02 */
    char name[80];
} dchkorhr_col_params_t;

static void run_dchkorhr_col_single(INT m, INT n, INT mb1, INT nb1, INT nb2, INT variant)
{
    f32 result[NTESTS];
    char ctx[128];
    INT minmn = (m < n) ? m : n;

    if (minmn == 0 || m < n || mb1 <= n || nb1 <= 0 || nb2 <= 0) {
        return;
    }

    if (variant == 1) {
        sorhr_col01(m, n, mb1, nb1, nb2, result);
    } else {
        sorhr_col02(m, n, mb1, nb1, nb2, result);
    }

    const char* vname = (variant == 1) ? "SORGTSQR+SORHR_COL" : "SORGTSQR_ROW+SORHR_COL";
    for (INT t = 0; t < NTESTS; t++) {
        snprintf(ctx, sizeof(ctx), "%s m=%d n=%d mb1=%d nb1=%d nb2=%d TEST %d",
                 vname, m, n, mb1, nb1, nb2, t + 1);
        set_test_context(ctx);
        assert_residual_below(result[t], THRESH);
    }

    clear_test_context();
}

static void test_dchkorhr_col_case(void** state)
{
    dchkorhr_col_params_t* params = *state;
    run_dchkorhr_col_single(params->m, params->n, params->mb1, params->nb1,
                            params->nb2, params->variant);
}

#define MAX_TESTS (NM * NN * NNB * NNB * NNB * 2)

static dchkorhr_col_params_t g_params[MAX_TESTS];
static struct CMUnitTest g_tests[MAX_TESTS];
static INT g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (INT variant = 1; variant <= 2; variant++) {
        for (INT im = 0; im < (INT)NM; im++) {
            INT m = MVAL[im];

            for (INT in = 0; in < (INT)NN; in++) {
                INT n = NVAL[in];
                INT minmn = (m < n) ? m : n;

                if (minmn == 0 || m < n) {
                    continue;
                }

                for (INT imb1 = 0; imb1 < (INT)NNB; imb1++) {
                    INT mb1 = NBVAL[imb1];

                    if (mb1 <= n) {
                        continue;
                    }

                    for (INT inb1 = 0; inb1 < (INT)NNB; inb1++) {
                        INT nb1 = NBVAL[inb1];

                        if (nb1 <= 0) {
                            continue;
                        }

                        for (INT inb2 = 0; inb2 < (INT)NNB; inb2++) {
                            INT nb2 = NBVAL[inb2];

                            if (nb2 <= 0) {
                                continue;
                            }

                            dchkorhr_col_params_t* p = &g_params[g_num_tests];
                            p->m = m;
                            p->n = n;
                            p->mb1 = mb1;
                            p->nb1 = nb1;
                            p->nb2 = nb2;
                            p->variant = variant;
                            snprintf(p->name, sizeof(p->name),
                                     "dchkorhr_col%d_m%d_n%d_mb1_%d_nb1_%d_nb2_%d",
                                     variant, m, n, mb1, nb1, nb2);

                            g_tests[g_num_tests].name = p->name;
                            g_tests[g_num_tests].test_func = test_dchkorhr_col_case;
                            g_tests[g_num_tests].setup_func = NULL;
                            g_tests[g_num_tests].teardown_func = NULL;
                            g_tests[g_num_tests].initial_state = p;

                            g_num_tests++;
                        }
                    }
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

    return _cmocka_run_group_tests("dchkorhr_col", g_tests, g_num_tests, NULL, NULL);
}
