/**
 * @file test_cgelsy_nan.c
 * @brief CGELSY regression test for the rank loop with a NaN in A.
 *
 * CGELSY increases RANK while SMAXPR*RCOND <= SMINPR. A NaN in A reaches
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
    const f32 rcond = 1.0e-8f;
    c64 A[M * N], B[M], work[LWORK];
    f32 rwork[2 * N];
    INT jpvt[N], rank, info;

    for (INT pos = 0; pos < M * N; pos += 7) {
        for (INT i = 0; i < M * N; i++) {
            A[i] = CMPLXF(sinf(1.0f + (f32)i), cosf(2.0f * (f32)i))
                   + ((i % 7 == 0) ? 2.0f : 0.0f);
        }
        A[pos] = CMPLXF((f32)NAN, 0.0f);
        for (INT i = 0; i < M; i++) B[i] = CMPLXF(1.0f + (f32)i, 0.0f);
        memset(jpvt, 0, sizeof(jpvt));
        cgelsy(M, N, 1, A, M, B, M, jpvt, rcond, &rank, work, LWORK, rwork,
               &info);
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
    (void)cmocka_run_group_tests_name("cgelsy_nan", tests, NULL, NULL);
    return 0;
}
