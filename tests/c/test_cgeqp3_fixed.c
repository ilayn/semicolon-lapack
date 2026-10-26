/**
 * @file test_cgeqp3_fixed.c
 * @brief CGEQP3 regression tests for fixed columns and empty matrices.
 *
 * test_fixed_minlwork runs CGEQP3 with leading fixed columns, LWORK = N+1
 * and N above the CGEQRF crossover, and checks that WORK is not written past
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
    const INT lwork = n + 1;
    const INT nguard = 2 * nfxd;
    const c64 canary = CMPLXF(12345.0f, -12345.0f);
    const INT lwchk = m * n + n;
    INT info;
    uint64_t rng_state[4];

    c64* A = malloc(lda * n * sizeof(c64));
    c64* AF = malloc(lda * n * sizeof(c64));
    c64* tau = malloc(minmn * sizeof(c64));
    c64* work = malloc((lwork + nguard) * sizeof(c64));
    c64* wchk = malloc(lwchk * sizeof(c64));
    f32* rwork = malloc(2 * n * sizeof(f32));
    INT* jpvt = malloc(n * sizeof(INT));
    assert_non_null(A); assert_non_null(AF); assert_non_null(tau);
    assert_non_null(work); assert_non_null(wchk); assert_non_null(rwork);
    assert_non_null(jpvt);

    rng_seed(rng_state, 0x6E9F3ULL);
    for (INT i = 0; i < lda * n; i++) {
        f32 ar = rng_uniform_symmetric_f32(rng_state);
        f32 ai = rng_uniform_symmetric_f32(rng_state);
        A[i] = CMPLXF(ar, ai);
        AF[i] = A[i];
    }
    for (INT j = 0; j < n; j++) {
        jpvt[j] = (j < nfxd) ? 1 : 0;
    }
    for (INT i = 0; i < lwork + nguard; i++) {
        work[i] = canary;
    }

    cgeqp3(m, n, AF, lda, jpvt, tau, work, lwork, rwork, &info);
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

    f32 resid = cqpt01(m, n, minmn, A, AF, lda, tau, jpvt, wchk, lwchk);
    assert_residual_below(resid, THRESH);

    free(A); free(AF); free(tau); free(work); free(wchk); free(rwork);
    free(jpvt);
}

static void test_empty_jpvt(void** state)
{
    (void)state;
    const INT n = 4;
    const INT jpvt_in[2][4] = { {0, 0, 0, 0}, {0, 0, 7, 0} };
    const INT jpvt_ex[2][4] = { {0, 1, 2, 3}, {2, 1, 0, 3} };
    c64 A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    c64 tau[1] = {0.0f};
    c64 work[1];
    f32 rwork[8];
    INT jpvt[4];
    INT info;

    for (INT k = 0; k < 2; k++) {
        for (INT j = 0; j < n; j++) {
            jpvt[j] = jpvt_in[k][j];
        }
        cgeqp3(0, n, A, 1, jpvt, tau, work, 1, rwork, &info);
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
    (void)cmocka_run_group_tests_name("cgeqp3_fixed", tests, NULL, NULL);
    return 0;
}
