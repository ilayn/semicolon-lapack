/**
 * @file test_dtrsyl3_swork.c
 * @brief DTRSYL3 regression test for the SWORK dimensions it reports.
 *
 * DTRSYL3 returns the dimensions ROWS and COLS of SWORK both from a
 * workspace query and on exit. The C port needs MAX(m, n) more rows than
 * the reference, so both must report the larger ROWS. The case queries,
 * solves with exactly the reported LDSWORK and LIWORK, and checks that the
 * values on exit equal the queried ones.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

static void test_reported_dims(void** state)
{
    (void)state;
    const INT m = 100, n = 100;
    INT info, iwq;
    f64 scale, swq[2];
    uint64_t rng_state[4];

    f64* A = calloc(m * m, sizeof(f64));
    f64* B = calloc(n * n, sizeof(f64));
    f64* C = malloc(m * n * sizeof(f64));
    assert_non_null(A); assert_non_null(B); assert_non_null(C);

    rng_seed(rng_state, 0x7E5F3ULL);
    for (INT j = 0; j < m; j++) {
        for (INT i = 0; i < j; i++) {
            A[i + j * m] = rng_uniform_symmetric(rng_state);
        }
        A[j + j * m] = 2.0 + (f64)j;
    }
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < j; i++) {
            B[i + j * n] = rng_uniform_symmetric(rng_state);
        }
        B[j + j * n] = 1.0 + (f64)j;
    }
    for (INT i = 0; i < m * n; i++) {
        C[i] = rng_uniform_symmetric(rng_state);
    }

    dtrsyl3("N", "N", 1, m, n, A, m, B, n, C, m, &scale, &iwq, -1, swq, -1, &info);
    assert_info_success(info);
    INT ldswork = (INT)swq[0];
    INT cols = (INT)swq[1];
    INT liwork = iwq;
    f64* swork = malloc(ldswork * cols * sizeof(f64));
    INT* iwork = malloc(liwork * sizeof(INT));
    assert_non_null(swork); assert_non_null(iwork);

    dtrsyl3("N", "N", 1, m, n, A, m, B, n, C, m, &scale, iwork, liwork,
            swork, ldswork, &info);
    assert_info_success(info);
    assert_int_equal((INT)swork[0], ldswork);
    assert_int_equal((INT)swork[1], cols);
    assert_int_equal(iwork[0], liwork);

    free(A); free(B); free(C); free(swork); free(iwork);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_reported_dims),
    };
    (void)cmocka_run_group_tests_name("dtrsyl3_swork", tests, NULL, NULL);
    return 0;
}
