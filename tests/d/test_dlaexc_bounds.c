/**
 * @file test_dlaexc_bounds.c
 * @brief DLAEXC regression test for a first block that ends at row N.
 *
 * DLAEXC returns at once when J1+N1 > N (1-based J1), that is when there
 * is no second block after T11. Each case calls DLAEXC with the 0-based
 * j1 = n - n1 for N1 = 1 and 2, and checks that INFO = 0 and that T and Q
 * are left untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>
#include <string.h>

static void run_last_block(const INT n1)
{
    const INT n = 6;
    const INT n2 = 1;
    const INT j1 = n - n1;
    INT info = -1;
    uint64_t rng_state[4];

    f64* T = malloc(n * n * sizeof(f64));
    f64* Q = malloc(n * n * sizeof(f64));
    f64* T0 = malloc(n * n * sizeof(f64));
    f64* Q0 = malloc(n * n * sizeof(f64));
    f64* work = malloc(n * sizeof(f64));
    assert_non_null(T); assert_non_null(Q); assert_non_null(T0);
    assert_non_null(Q0); assert_non_null(work);

    rng_seed(rng_state, 0xE7C1ULL);
    for (INT i = 0; i < n * n; i++) {
        T[i] = rng_uniform_symmetric(rng_state);
        Q[i] = rng_uniform_symmetric(rng_state);
    }
    memcpy(T0, T, n * n * sizeof(f64));
    memcpy(Q0, Q, n * n * sizeof(f64));

    dlaexc(1, n, T, n, Q, n, j1, n1, n2, work, &info);
    assert_info_success(info);
    assert_memory_equal(T, T0, n * n * sizeof(f64));
    assert_memory_equal(Q, Q0, n * n * sizeof(f64));

    free(T); free(Q); free(T0); free(Q0); free(work);
}

static void test_n1_1(void** state)
{
    (void)state;
    run_last_block(1);
}

static void test_n1_2(void** state)
{
    (void)state;
    run_last_block(2);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_n1_1),
        cmocka_unit_test(test_n1_2),
    };
    (void)cmocka_run_group_tests_name("dlaexc_bounds", tests, NULL, NULL);
    return 0;
}
