/**
 * @file test_sgeqp3_fixed.c
 * @brief SGEQP3 regression tests for fixed columns and empty matrices.
 *
 * test_fixed_minlwork runs SGEQP3 with leading fixed columns, LWORK = 3*N+1
 * and N above the SGEQRF crossover, and checks that WORK is not written past
 * LWORK. test_empty_jpvt checks that JPVT is returned as a permutation for
 * M = 0, N > 0.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

#define THRESH 30.0f

static void test_fixed_minlwork(void** state)
{
    (void)state;
    const INT m = 174;
    const INT n = 174;
    const INT nfxd = 44;
    const INT lda = m;
    const INT minmn = m < n ? m : n;
    const INT lwork = 3 * n + 1;
    const INT nguard = 2 * nfxd;
    const f32 canary = 12345.0f;
    const INT lwchk = m * n + n;
    INT info;
    uint64_t rng_state[4];

    f32* A = malloc(lda * n * sizeof(f32));
    f32* AF = malloc(lda * n * sizeof(f32));
    f32* tau = malloc(minmn * sizeof(f32));
    f32* work = malloc((lwork + nguard) * sizeof(f32));
    f32* wchk = malloc(lwchk * sizeof(f32));
    INT* jpvt = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(AF); assert_non_null(tau);
    assert_non_null(work); assert_non_null(wchk); assert_non_null(jpvt);

    rng_seed(rng_state, 0x6E9F3ULL);
    for (INT i = 0; i < lda * n; i++) {
        A[i] = rng_uniform_symmetric_f32(rng_state);
        AF[i] = A[i];
    }
    for (INT j = 0; j < n; j++) {
        jpvt[j] = (j < nfxd) ? 1 : 0;
    }
    for (INT i = 0; i < lwork + nguard; i++) {
        work[i] = canary;
    }

    sgeqp3(m, n, AF, lda, jpvt, tau, work, lwork, &info);
    assert_info_success(info);

    INT nover = 0;
    for (INT i = lwork; i < lwork + nguard; i++) {
        if (work[i] != canary) nover++;
    }
    if (nover != 0) {
        fail_msg("%d of %d guard elements past WORK(LWORK) were written, lwork = %d",
                 (int)nover, (int)nguard, (int)lwork);
    }

    for (INT j = 0; j < nfxd; j++) {
        assert_int_equal(jpvt[j], j);
    }

    f32 resid = sqpt01(m, n, minmn, A, AF, lda, tau, jpvt, wchk, lwchk);
    assert_residual_below(resid, THRESH);

    free(A); free(AF); free(tau); free(work); free(wchk); free(jpvt);
}

static void test_empty_jpvt(void** state)
{
    (void)state;
    const INT n = 4;
    const INT jpvt_in[2][4] = { {0, 0, 0, 0}, {0, 0, 7, 0} };
    const INT jpvt_ex[2][4] = { {0, 1, 2, 3}, {2, 1, 0, 3} };
    f32 A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    f32 tau[1] = {0.0f};
    f32 work[1];
    INT jpvt[4];
    INT info;

    for (INT k = 0; k < 2; k++) {
        for (INT j = 0; j < n; j++) {
            jpvt[j] = jpvt_in[k][j];
        }
        sgeqp3(0, n, A, 1, jpvt, tau, work, 1, &info);
        assert_info_success(info);
        for (INT j = 0; j < n; j++) {
            assert_int_equal(jpvt[j], jpvt_ex[k][j]);
        }
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_fixed_minlwork),
        cmocka_unit_test(test_empty_jpvt),
    };
    (void)cmocka_run_group_tests_name("dgeqp3_fixed", tests, NULL, NULL);
    return 0;
}
