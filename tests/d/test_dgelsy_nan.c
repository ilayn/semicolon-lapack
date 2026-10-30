/**
 * @file test_dgelsy_nan.c
 * @brief DGELSY regression test for the rank loop with a NaN in A.
 *
 * DGELSY increases RANK while SMAXPR*RCOND <= SMINPR. A NaN in A reaches
 * the triangular factor and the incremental condition estimates, the test
 * is then false and the loop stops, so RANK < MIN(M,N). Each case puts one
 * NaN in a different entry of A and checks INFO = 0 and RANK < MIN(M,N).
 */

#include "test_harness.h"
#include "verify.h"
#include <math.h>
#include <string.h>

#define M 6
#define N 5
#define LWORK 1000

static void test_nan(void** state)
{
    (void)state;
    const f64 rcond = 1.0e-8;
    f64 A[M * N], B[M], work[LWORK];
    INT jpvt[N], rank, info;

    for (INT pos = 0; pos < M * N; pos += 7) {
        for (INT i = 0; i < M * N; i++) {
            A[i] = sin(1.0 + (f64)i) + ((i % 7 == 0) ? 2.0 : 0.0);
        }
        A[pos] = (f64)NAN;
        for (INT i = 0; i < M; i++) B[i] = 1.0 + (f64)i;
        memset(jpvt, 0, sizeof(jpvt));
        dgelsy(M, N, 1, A, M, B, M, jpvt, rcond, &rank, work, LWORK, &info);
        assert_info_success(info);
        if (!(rank < N)) {
            fail_msg("NaN in A(%lld): RANK = %lld, expected < %d",
                     (long long)pos, (long long)rank, N);
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_nan),
    };
    (void)cmocka_run_group_tests_name("dgelsy_nan", tests, NULL, NULL);
    return 0;
}
