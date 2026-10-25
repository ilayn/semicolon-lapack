/**
 * @file test_slarf_incv.c
 * @brief SLARF, SLARF1F and SLARF1L regression tests.
 *
 * Each routine is applied from both sides with INCV = -2, -1, 1, 2, with
 * v either full or zero at its leading end, its trailing end or everywhere,
 * and compared with a dense application of H = I - tau * u * u**T, where u
 * is v with the implicit unit entry of SLARF1F (first) or SLARF1L (last).
 * The gaps between the stored entries of v hold a guard value.
 * test_empty checks that SLARF1F and SLARF1L leave an empty C untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0f
#define MM 5
#define NN 4
#define MAXINC 2

typedef void (*larf_fn)(const char*, const INT, const INT, const f32*,
                        const INT, const f32, f32*, const INT, f32*);

/* unit: -1 none (SLARF), 0 first entry (SLARF1F), 1 last entry (SLARF1L) */
static void check_larf(larf_fn fn, INT unit)
{
    const char* sides[2] = {"L", "R"};
    const INT incvs[4] = {-2, -1, 1, 2};
    const f32 tau = 0.7f;
    const f32 guard = 1.0e6f;
    const f32 eps = slamch("P");
    f32 v[1 + (MM - 1) * MAXINC], u[MM], C[MM * NN], R[MM * NN], work[MM];

    for (INT is = 0; is < 2; is++) {
        const INT len = (is == 0) ? MM : NN;
        for (INT ii = 0; ii < 4; ii++) {
            const INT incv = incvs[ii];
            const INT ainc = (incv > 0) ? incv : -incv;
            for (INT pat = 0; pat < 4; pat++) {
                for (INT k = 0; k < 1 + (len - 1) * ainc; k++) {
                    v[k] = guard;
                }
                for (INT k = 0; k < len; k++) {
                    f32 val = 0.3f + 0.1f * k;
                    if ((pat == 1 && k < len / 2) ||
                        (pat == 2 && k >= len / 2) || pat == 3) {
                        val = 0.0f;
                    }
                    v[(incv > 0) ? k * incv : (len - 1 - k) * ainc] = val;
                    u[k] = val;
                }
                if (unit == 0) u[0] = 1.0f;
                if (unit == 1) u[len - 1] = 1.0f;

                f32 cmax = 0.0f;
                for (INT j = 0; j < NN; j++) {
                    for (INT i = 0; i < MM; i++) {
                        C[i + j * MM] = sinf(1.0f + i + 3.0f * j);
                        cmax = fmaxf(cmax, fabsf(C[i + j * MM]));
                    }
                }
                for (INT j = 0; j < NN; j++) {
                    for (INT i = 0; i < MM; i++) {
                        f32 s = 0.0f;
                        if (is == 0) {
                            for (INT p = 0; p < MM; p++) s += u[i] * u[p] * C[p + j * MM];
                        } else {
                            for (INT p = 0; p < NN; p++) s += C[i + p * MM] * u[p] * u[j];
                        }
                        R[i + j * MM] = C[i + j * MM] - tau * s;
                    }
                }

                fn(sides[is], MM, NN, v, incv, tau, C, MM, work);

                f32 err = 0.0f;
                for (INT k = 0; k < MM * NN; k++) {
                    f32 d = fabsf(C[k] - R[k]);
                    if (!(d <= err)) err = d;
                }
                f32 resid = err / (len * eps * cmax);
                assert_residual_below(resid, THRESH);
            }
        }
    }
}

static void test_larf(void** state)
{
    (void)state;
    check_larf(slarf, -1);
}

static void test_larf1f(void** state)
{
    (void)state;
    check_larf(slarf1f, 0);
}

static void test_larf1l(void** state)
{
    (void)state;
    check_larf(slarf1l, 1);
}

static void test_empty(void** state)
{
    (void)state;
    const f32 guard = -7.0f;
    f32 buf[8], v[4] = {1.0f, 1.0f, 1.0f, 1.0f}, work[4];
    for (INT k = 0; k < 8; k++) buf[k] = guard;

    slarf1f("L", 0, 2, v, 1, 0.5f, &buf[2], 2, work);
    slarf1f("R", 2, 0, v, 1, 0.5f, &buf[2], 2, work);
    slarf1l("L", 0, 2, v, 1, 0.5f, &buf[2], 2, work);
    slarf1l("R", 2, 0, v, 1, 0.5f, &buf[2], 2, work);

    for (INT k = 0; k < 8; k++) {
        assert_true(buf[k] == guard);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_larf),
        cmocka_unit_test(test_larf1f),
        cmocka_unit_test(test_larf1l),
        cmocka_unit_test(test_empty),
    };
    (void)cmocka_run_group_tests_name("dlarf_incv", tests, NULL, NULL);
    return 0;
}
