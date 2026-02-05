/**
 * @file test_dchkorhr_col.c
 * @brief Comprehensive test suite for Householder reconstruction routines.
 *
 * This is a faithful port of LAPACK's TESTING/LIN/dchkorhr_col.f to C using CMocka.
 * Tests:
 *   1) DORGTSQR and DORHR_COL using DLATSQR, DGEMQRT (via dorhr_col01)
 *   2) DORGTSQR_ROW and DORHR_COL inside DGETSQRHRT (via dorhr_col02)
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
#include <string.h>
#include <stdio.h>

static const int MVAL[] = {5, 10, 16, 50};
static const int NVAL[] = {2, 5, 10, 16};
static const int NBVAL[] = {1, 2, 5, 10, 16, 32, 64};

#define NM      (sizeof(MVAL) / sizeof(MVAL[0]))
#define NN      (sizeof(NVAL) / sizeof(NVAL[0]))
#define NNB     (sizeof(NBVAL) / sizeof(NBVAL[0]))
#define NTESTS  6
#define THRESH  30.0

extern void dorhr_col01(const int m, const int n, const int mb1, const int nb1,
                        const int nb2, double* restrict result);
extern void dorhr_col02(const int m, const int n, const int mb1, const int nb1,
                        const int nb2, double* restrict result);

typedef struct {
    int m;
    int n;
    int mb1;
    int nb1;
    int nb2;
    int variant;  /* 1 = dorhr_col01, 2 = dorhr_col02 */
    char name[80];
} dchkorhr_col_params_t;

static void run_dchkorhr_col_single(int m, int n, int mb1, int nb1, int nb2, int variant)
{
    double result[NTESTS];
    char ctx[128];
    int minmn = (m < n) ? m : n;

    if (minmn == 0 || m < n || mb1 <= n || nb1 <= 0 || nb2 <= 0) {
        return;
    }

    if (variant == 1) {
        dorhr_col01(m, n, mb1, nb1, nb2, result);
    } else {
        dorhr_col02(m, n, mb1, nb1, nb2, result);
    }

    const char* vname = (variant == 1) ? "DORGTSQR+DORHR_COL" : "DORGTSQR_ROW+DORHR_COL";
    for (int t = 0; t < NTESTS; t++) {
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
static int g_num_tests = 0;

static void build_test_array(void)
{
    g_num_tests = 0;

    for (int variant = 1; variant <= 2; variant++) {
        for (int im = 0; im < (int)NM; im++) {
            int m = MVAL[im];

            for (int in = 0; in < (int)NN; in++) {
                int n = NVAL[in];
                int minmn = (m < n) ? m : n;

                if (minmn == 0 || m < n) {
                    continue;
                }

                for (int imb1 = 0; imb1 < (int)NNB; imb1++) {
                    int mb1 = NBVAL[imb1];

                    if (mb1 <= n) {
                        continue;
                    }

                    for (int inb1 = 0; inb1 < (int)NNB; inb1++) {
                        int nb1 = NBVAL[inb1];

                        if (nb1 <= 0) {
                            continue;
                        }

                        for (int inb2 = 0; inb2 < (int)NNB; inb2++) {
                            int nb2 = NBVAL[inb2];

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
