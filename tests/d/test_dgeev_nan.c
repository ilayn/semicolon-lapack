/**
 * @file test_dgeev_nan.c
 * @brief DGEEV regression test for a matrix that contains a NaN.
 *
 * DGEEV must reject the matrix with INFO = -4 through XERBLA instead of
 * passing it on to DGEBAL.
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define NMAX 3
#define LW (4 * NMAX)

static void test_nan(void** state)
{
    (void)state;
    f64 A[NMAX * NMAX], wr[NMAX], wi[NMAX], VL[NMAX * NMAX], VR[NMAX * NMAX];
    f64 work[LW];
    INT info, lerr, ok;

    for (INT j = 0; j < NMAX; j++) {
        for (INT i = 0; i < NMAX; i++) {
            A[i + j * NMAX] = (i == j) ? 2.0 : 1.0;
        }
    }
    A[1 + 1 * NMAX] = (f64)NAN;

    strcpy(xerbla_srnamt, "DGEEV");
    xerbla_infot = 4;
    xerbla_lerr = 0;
    xerbla_ok = 1;
    dgeev("V", "V", NMAX, A, NMAX, wr, wi, VL, NMAX, VR, NMAX, work, LW, &info);
    lerr = xerbla_lerr;
    ok = xerbla_ok;
    xerbla_srnamt[0] = '\0';
    xerbla_lerr = 0;
    xerbla_ok = 1;

    assert_int_equal(info, -4);
    assert_true(lerr);
    assert_true(ok);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_nan),
    };
    (void)cmocka_run_group_tests_name("dgeev_nan", tests, NULL, NULL);
    return 0;
}
