/**
 * @file test_cstein_unconverged.c
 * @brief CSTEIN regression test for the negative IBLOCK entries with which
 *        SSTEBZ flags eigenvalues that did not converge.
 *
 * SSTEBZ stores -(b+2) in iblock for an eigenvalue of block b that did not
 * converge. CSTEIN must treat such an entry as block b, so the computed
 * eigenvectors must equal those computed from the unflagged iblock.
 */

#include "test_harness.h"
#include "verify.h"

#define N 6

static void test_flagged_blocks(void** state)
{
    (void)state;
    const f32 D[N] = {1.0f, 2.0f, 3.5f, 4.0f, 5.0f, -1.0f};
    const f32 E[N - 1] = {0.5f, 0.0f, 0.7f, 0.3f, 0.0f};
    const INT blocks[N] = {0, 0, 1, 1, 1, 2};
    const INT flags[4][N] = {
        {0, 1, 0, 0, 0, 0},
        {1, 1, 0, 0, 0, 0},
        {0, 0, 1, 0, 1, 0},
        {0, 0, 0, 0, 0, 1},
    };
    INT m, nsplit, info;
    f32 W[N], work[5 * N];
    c64 Zref[N * N], Z[N * N];
    INT iblock[N], iblkf[N], isplit[N], iwork[3 * N], ifail[N];

    sstebz("A", "B", N, 0.0f, 0.0f, 0, 0, 0.0f, D, E, &m, &nsplit, W, iblock,
           isplit, work, iwork, &info);
    assert_info_success(info);
    assert_int_equal(m, N);
    assert_int_equal(nsplit, 3);
    for (INT i = 0; i < N; i++) {
        assert_int_equal(iblock[i], blocks[i]);
    }

    cstein(N, D, E, m, W, iblock, isplit, Zref, N, work, iwork, ifail, &info);
    assert_info_success(info);

    for (INT c = 0; c < 4; c++) {
        for (INT i = 0; i < N; i++) {
            iblkf[i] = flags[c][i] ? -(iblock[i] + 2) : iblock[i];
        }
        for (INT i = 0; i < N * N; i++) {
            Z[i] = CMPLXF(7.0f, 7.0f);
        }
        cstein(N, D, E, m, W, iblkf, isplit, Z, N, work, iwork, ifail, &info);
        if (info != 0) {
            fail_msg("case %d: info = %d", (int)c, (int)info);
        }
        for (INT i = 0; i < N * N; i++) {
            if (Z[i] != Zref[i]) {
                fail_msg("case %d: Z[%d] differs from the unflagged result",
                         (int)c, (int)i);
            }
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_flagged_blocks),
    };
    (void)cmocka_run_group_tests_name("zstein_unconverged", tests, NULL, NULL);
    return 0;
}
