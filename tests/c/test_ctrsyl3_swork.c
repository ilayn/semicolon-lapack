/**
 * @file test_ctrsyl3_swork.c
 * @brief CTRSYL3 regression test for the SWORK dimensions it reports.
 *
 * CTRSYL3 returns the dimensions ROWS and COLS of SWORK both from a
 * workspace query and on exit. The C port needs MAX(m, n) more rows than
 * the reference, so both must report the larger ROWS. The case queries,
 * solves with exactly the reported LDSWORK, and checks that the values on
 * exit equal the queried ones.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

static void test_reported_dims(void** state)
{
    (void)state;
    const INT m = 100, n = 100;
    INT info;
    f32 scale, swq[2];
    uint64_t rng_state[4];

    c64* A = calloc(m * m, sizeof(c64));
    c64* B = calloc(n * n, sizeof(c64));
    c64* C = malloc(m * n * sizeof(c64));
    assert_non_null(A); assert_non_null(B); assert_non_null(C);

    rng_seed(rng_state, 0x7E5F3ULL);
    for (INT j = 0; j < m; j++) {
        for (INT i = 0; i < j; i++) {
            f32 re = rng_uniform_symmetric_f32(rng_state);
            f32 im = rng_uniform_symmetric_f32(rng_state);
            A[i + j * m] = CMPLXF(re, im);
        }
        A[j + j * m] = CMPLXF(2.0f + (f32)j, 0.5f);
    }
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < j; i++) {
            f32 re = rng_uniform_symmetric_f32(rng_state);
            f32 im = rng_uniform_symmetric_f32(rng_state);
            B[i + j * n] = CMPLXF(re, im);
        }
        B[j + j * n] = CMPLXF(1.0f + (f32)j, -0.5f);
    }
    for (INT i = 0; i < m * n; i++) {
        f32 re = rng_uniform_symmetric_f32(rng_state);
        f32 im = rng_uniform_symmetric_f32(rng_state);
        C[i] = CMPLXF(re, im);
    }

    ctrsyl3("N", "N", 1, m, n, A, m, B, n, C, m, &scale, swq, -1, &info);
    assert_info_success(info);
    INT ldswork = (INT)swq[0];
    INT cols = (INT)swq[1];
    f32* swork = malloc(ldswork * cols * sizeof(f32));
    assert_non_null(swork);

    ctrsyl3("N", "N", 1, m, n, A, m, B, n, C, m, &scale, swork, ldswork, &info);
    assert_info_success(info);
    assert_int_equal((INT)swork[0], ldswork);
    assert_int_equal((INT)swork[1], cols);

    free(A); free(B); free(C); free(swork);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_reported_dims),
    };
    (void)cmocka_run_group_tests_name("ctrsyl3_swork", tests, NULL, NULL);
    return 0;
}
