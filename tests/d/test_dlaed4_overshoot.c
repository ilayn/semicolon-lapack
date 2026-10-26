/**
 * @file test_dlaed4_overshoot.c
 * @brief DLAED4 regression test for the overshoot fallback step.
 *
 * The secular equation of Reference-LAPACK issue #1166 (N = 29, I = 23)
 * has its root very close to the left pole. Its rational interpolation
 * step overshoots the bracket several times, and halving towards the
 * bracket end used up the 30 iterations without convergence (INFO = 1).
 * The root must be found and satisfy the secular equation.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>

#define THRESH 30.0
#define NN 29

static void test_issue1166(void** state)
{
    (void)state;
    const INT n = NN;
    const INT i = 22;
    const f64 rho = 6.2964808955495258e-05;
    const f64 d[NN] = {
        -8.9907411945275305e-17, 3.1527281176230556e-06, 7.8104209108493992e-06,
        1.4528378412105169e-05, 1.6571840586254615e-05, 1.6791959834500457e-05,
        1.6951003434697501e-05, 1.6961635049587819e-05, 1.6969218460422806e-05,
        1.7028037406879576e-05, 1.9135232081591496e-05, 2.1876238530929982e-05,
        2.1920146416725336e-05, 3.9357203097119847e-05, 1.0557423781645928e-04,
        1.3686127039388258e-04, 1.3686300570614894e-04, 1.3686313427158908e-04,
        1.3686314099616321e-04, 1.3686314127052412e-04, 1.3686314210337733e-04,
        1.3686314231692531e-04, 1.3686314257213152e-04, 5.3532577970471340e-02,
        9.9996866934433004e-01, 1.0000000849390769e+00, 1.0000000901678945e+00,
        1.0000000918404301e+00, 1.0000001095998177e+00
    };
    const f64 z[NN] = {
        1.6700411548440888e-07, 2.2704845744451106e-06, -6.8596999842130412e-06,
        -1.9813464524908256e-04, 3.4676472662621390e-03, 5.8131816598176476e-02,
        -2.4389840480820292e-03, 2.2571085689296944e-03, -2.0150012240231657e-03,
        -9.7432047705138750e-03, 7.3184880793391433e-04, -2.9319792346511997e-02,
        8.3364827297377855e-07, -8.9722126354645391e-09, 7.0412738449665979e-01,
        -1.4042299148200350e-06, 3.8731670039916887e-06, -5.4330381400333468e-07,
        4.4026694557354993e-07, 2.2528424563293299e-07, 3.6064772311683760e-06,
        2.8472995783701232e-07, 8.3086036055888118e-07, 2.8464273601162125e-02,
        -7.0642255076744132e-01, -5.2009238883566685e-08, 8.6348330718758654e-08,
        -2.9664496252475970e-08, 1.4932776629316714e-09
    };
    f64 delta[NN];
    f64 dlam;
    INT info;

    dlaed4(n, i, d, z, delta, rho, &dlam, &info);
    assert_info_success(info);
    assert_true(d[i] < dlam && dlam < d[i + 1]);

    /* 1/rho + sum z(j)**2 / (d(j) - dlam) must vanish to working
     * precision relative to the size of its terms. */
    const f64 eps = dlamch("E");
    f64 w = 1.0 / rho;
    f64 wabs = 1.0 / rho;
    for (INT j = 0; j < n; j++) {
        f64 t = z[j] * (z[j] / delta[j]);
        w += t;
        wabs += fabs(t);
    }
    f64 resid = fabs(w) / (n * eps * wabs);
    assert_residual_below(resid, THRESH);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_issue1166),
    };
    (void)cmocka_run_group_tests_name("dlaed4_overshoot", tests, NULL, NULL);
    return 0;
}
