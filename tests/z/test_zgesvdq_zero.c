/**
 * @file test_zgesvdq_zero.c
 * @brief ZGESVDQ regression test for the pivots returned for the zero matrix.
 *
 * With JOBP = 'P' and A = 0, ZGESVDQ returns early with NUMRANK = 0, zero
 * singular values and the identity column and row permutations in IWORK.
 */

#include "test_harness.h"
#include "verify.h"
#include <string.h>

static void test_zero_matrix(void** state)
{
    (void)state;
    const INT m = 6, n = 4;
    const INT liwork = n + m - 1;
    const INT lrwork = (m > 5 * n) ? m : 5 * n;
    c128 A[6 * 4], U[6 * 4], V[4 * 4], cwork[64];
    f64 S[4], rwork[20];
    INT iwork[9], numrank, info;

    for (INT i = 0; i < m * n; i++) A[i] = CMPLX(0.0, 0.0);
    for (INT i = 0; i < 9; i++) iwork[i] = -7;
    zgesvdq("H", "P", "N", "U", "V", m, n, A, m, S, U, m, V, n, &numrank,
            iwork, liwork, cwork, 64, rwork, lrwork, &info);
    assert_info_success(info);
    assert_int_equal(numrank, 0);
    for (INT p = 0; p < n; p++) {
        assert_int_equal(iwork[p], p);
        assert_true(S[p] == 0.0);
    }
    for (INT p = n; p < n + m - 1; p++) {
        assert_int_equal(iwork[p], p - n);
    }
    assert_true(rwork[1] == -1.0);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_zero_matrix),
    };
    (void)cmocka_run_group_tests_name("zgesvdq_zero", tests, NULL, NULL);
    return 0;
}
