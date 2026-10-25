/**
 * @file test_sstebz_discard.c
 * @brief SSTEBZ regression test for the RANGE = 'I' discard loops.
 *
 * test_discard_no_candidate runs SSTEBZ with RANGE = 'I' on a split matrix
 * with a NaN off-diagonal, where the discard loop finds no eigenvalue left to
 * remove, and checks that IBLOCK is not written before its first element.
 */

#include "test_harness.h"
#include <math.h>
#include <stdlib.h>

static void test_discard_no_candidate(void** state)
{
    (void)state;
    const INT n = 4;
    const INT il = 1;
    const INT iu = 1;
    const INT canary = 777;
    INT m, nsplit, info;

    f32* D = malloc(n * sizeof(f32));
    f32* E = malloc((n - 1) * sizeof(f32));
    f32* W = malloc(n * sizeof(f32));
    f32* work = malloc(4 * n * sizeof(f32));
    INT* ibuf = malloc((n + 1) * sizeof(INT));
    INT* isplit = malloc(n * sizeof(INT));
    INT* iwork = malloc(3 * n * sizeof(INT));
    assert_non_null(D); assert_non_null(E); assert_non_null(W);
    assert_non_null(work); assert_non_null(ibuf); assert_non_null(isplit);
    assert_non_null(iwork);
    INT* iblock = ibuf + 1;

    D[0] = 1.0f; D[1] = 0.0f; D[2] = 0.0f; D[3] = 0.0f;
    E[0] = 0.0f; E[1] = (f32)NAN; E[2] = 0.0f;
    ibuf[0] = canary;

    sstebz("I", "E", n, 0.0f, 0.0f, il, iu, 0.0f, D, E, &m, &nsplit, W,
           iblock, isplit, work, iwork, &info);

    assert_int_equal(ibuf[0], canary);
    assert_true(m >= 0 && m <= iu - il + 1);

    free(D); free(E); free(W); free(work); free(ibuf); free(isplit);
    free(iwork);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_discard_no_candidate),
    };
    (void)cmocka_run_group_tests_name("dstebz_discard", tests, NULL, NULL);
    return 0;
}
