/**
 * @file test_zlarf_incv.c
 * @brief ZLARF, ZLARF1F and ZLARF1L regression tests.
 *
 * Each routine is applied from both sides with INCV = -2, -1, 1, 2, with
 * v either full or zero at its leading end, its trailing end or everywhere,
 * and compared with a dense application of H = I - tau * u * u**H, where u
 * is v with the implicit unit entry of ZLARF1F (first) or ZLARF1L (last).
 * The gaps between the stored entries of v hold a guard value.
 * test_empty checks that ZLARF1F and ZLARF1L leave an empty C untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <stdlib.h>

#define THRESH 30.0
#define MM 5
#define NN 4
#define MAXINC 2

typedef void (*larf_fn)(const char*, const INT, const INT, const c128*,
                        const INT, const c128, c128*, const INT, c128*);

/* unit: -1 none (ZLARF), 0 first entry (ZLARF1F), 1 last entry (ZLARF1L) */
static void check_larf(larf_fn fn, INT unit)
{
    const char* sides[2] = {"L", "R"};
    const INT incvs[4] = {-2, -1, 1, 2};
    const c128 tau = CMPLX(0.7, 0.2);
    const c128 guard = CMPLX(1.0e6, 1.0e6);
    const f64 eps = dlamch("P");
    c128 v[1 + (MM - 1) * MAXINC], u[MM], C[MM * NN], R[MM * NN], work[MM];

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
                    c128 val = CMPLX(0.3 + 0.1 * k, 0.05 * k);
                    if ((pat == 1 && k < len / 2) ||
                        (pat == 2 && k >= len / 2) || pat == 3) {
                        val = CMPLX(0.0, 0.0);
                    }
                    v[(incv > 0) ? k * incv : (len - 1 - k) * ainc] = val;
                    u[k] = val;
                }
                if (unit == 0) u[0] = CMPLX(1.0, 0.0);
                if (unit == 1) u[len - 1] = CMPLX(1.0, 0.0);

                f64 cmax = 0.0;
                for (INT j = 0; j < NN; j++) {
                    for (INT i = 0; i < MM; i++) {
                        C[i + j * MM] = CMPLX(sin(1.0 + i + 3.0 * j), cos(2.0 * i - j));
                        cmax = fmax(cmax, cabs(C[i + j * MM]));
                    }
                }
                for (INT j = 0; j < NN; j++) {
                    for (INT i = 0; i < MM; i++) {
                        c128 s = CMPLX(0.0, 0.0);
                        if (is == 0) {
                            for (INT p = 0; p < MM; p++) s += u[i] * conj(u[p]) * C[p + j * MM];
                        } else {
                            for (INT p = 0; p < NN; p++) s += C[i + p * MM] * u[p] * conj(u[j]);
                        }
                        R[i + j * MM] = C[i + j * MM] - tau * s;
                    }
                }

                fn(sides[is], MM, NN, v, incv, tau, C, MM, work);

                f64 err = 0.0;
                for (INT k = 0; k < MM * NN; k++) {
                    f64 d = cabs(C[k] - R[k]);
                    if (!(d <= err)) err = d;
                }
                f64 resid = err / (len * eps * cmax);
                assert_residual_below(resid, THRESH);
            }
        }
    }
}

static void test_larf(void** state)
{
    (void)state;
    check_larf(zlarf, -1);
}

static void test_larf1f(void** state)
{
    (void)state;
    check_larf(zlarf1f, 0);
}

static void test_larf1l(void** state)
{
    (void)state;
    check_larf(zlarf1l, 1);
}

static void test_empty(void** state)
{
    (void)state;
    const c128 guard = CMPLX(-7.0, 3.0);
    const c128 tau = CMPLX(0.5, 0.0);
    c128 buf[8], v[4], work[4];
    for (INT k = 0; k < 8; k++) buf[k] = guard;
    for (INT k = 0; k < 4; k++) v[k] = CMPLX(1.0, 0.0);

    zlarf1f("L", 0, 2, v, 1, tau, &buf[2], 2, work);
    zlarf1f("R", 2, 0, v, 1, tau, &buf[2], 2, work);
    zlarf1l("L", 0, 2, v, 1, tau, &buf[2], 2, work);
    zlarf1l("R", 2, 0, v, 1, tau, &buf[2], 2, work);

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
    (void)cmocka_run_group_tests_name("zlarf_incv", tests, NULL, NULL);
    return 0;
}
