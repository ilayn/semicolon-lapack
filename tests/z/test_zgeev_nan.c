/**
 * @file test_zgeev_nan.c
 * @brief ZGEEV regression test for a matrix that contains a NaN.
 *
 * ZGEEV must reject the matrix with INFO = -4 through XERBLA instead of
 * passing it on to ZGEBAL.
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
    c128 A[NMAX * NMAX], W[NMAX], VL[NMAX * NMAX], VR[NMAX * NMAX];
    c128 work[LW];
    f64 rwork[2 * NMAX];
    INT info, lerr, ok;

    for (INT j = 0; j < NMAX; j++) {
        for (INT i = 0; i < NMAX; i++) {
            A[i + j * NMAX] = (i == j) ? CMPLX(2.0, 1.0) : CMPLX(1.0, 0.0);
        }
    }
    A[1 + 1 * NMAX] = CMPLX(NAN, 0.0);

    strcpy(xerbla_srnamt, "ZGEEV");
    xerbla_infot = 4;
    xerbla_lerr = 0;
    xerbla_ok = 1;
    zgeev("V", "V", NMAX, A, NMAX, W, VL, NMAX, VR, NMAX, work, LW, rwork, &info);
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
    (void)cmocka_run_group_tests_name("zgeev_nan", tests, NULL, NULL);
    return 0;
}
