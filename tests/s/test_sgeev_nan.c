/**
 * @file test_sgeev_nan.c
 * @brief SGEEV regression test for a matrix that contains a NaN.
 *
 * SGEEV must reject the matrix with INFO = -4 through XERBLA instead of
 * passing it on to SGEBAL.
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
    f32 A[NMAX * NMAX], wr[NMAX], wi[NMAX], VL[NMAX * NMAX], VR[NMAX * NMAX];
    f32 work[LW];
    INT info, lerr, ok;

    for (INT j = 0; j < NMAX; j++) {
        for (INT i = 0; i < NMAX; i++) {
            A[i + j * NMAX] = (i == j) ? 2.0f : 1.0f;
        }
    }
    A[1 + 1 * NMAX] = NAN;

    strcpy(xerbla_srnamt, "SGEEV");
    xerbla_infot = 4;
    xerbla_lerr = 0;
    xerbla_ok = 1;
    sgeev("V", "V", NMAX, A, NMAX, wr, wi, VL, NMAX, VR, NMAX, work, LW, &info);
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
    (void)cmocka_run_group_tests_name("sgeev_nan", tests, NULL, NULL);
    return 0;
}
