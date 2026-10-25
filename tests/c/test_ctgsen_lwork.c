/**
 * @file test_ctgsen_lwork.c
 * @brief CTGSEN regression test for the minimum LWORK of IJOB = 1, 2 and 4.
 *
 * For IJOB = 1, 2 and 4, CTGSEN passes WORK(2*M*(N-M)+1) to CTGSYL, which
 * stores WORK(1) there, so LWORK must be at least 2*M*(N-M) + 1 (LAPACK
 * commit 116393587). Each case calls CTGSEN with exactly the queried LWORK
 * and checks that the element after WORK is left untouched.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"
#include <stdlib.h>

#define CANARY -12345.0f

static void run_tgsen(const INT ijob)
{
    const INT n = 20;
    INT m, info, lwork, liwork, iwq;
    f32 pl = -1.0f, pr = -1.0f, dif[2] = {-1.0f, -1.0f};
    c64 wq;
    uint64_t rng_state[4];

    c64* A = calloc(n * n, sizeof(c64));
    c64* B = calloc(n * n, sizeof(c64));
    c64* Q = calloc(n * n, sizeof(c64));
    c64* Z = calloc(n * n, sizeof(c64));
    c64* alpha = malloc(n * sizeof(c64));
    c64* beta = malloc(n * sizeof(c64));
    INT* select = calloc(n, sizeof(INT));
    assert_non_null(A); assert_non_null(B); assert_non_null(Q);
    assert_non_null(Z); assert_non_null(alpha); assert_non_null(beta);
    assert_non_null(select);

    rng_seed(rng_state, 0x7653EULL);
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i <= j; i++) {
            f32 ar = rng_uniform_symmetric_f32(rng_state);
            f32 ai = rng_uniform_symmetric_f32(rng_state);
            f32 br = rng_uniform_symmetric_f32(rng_state);
            f32 bi = rng_uniform_symmetric_f32(rng_state);
            A[i + j * n] = CMPLXF(ar, ai);
            B[i + j * n] = CMPLXF(br, bi);
        }
        A[j + j * n] += (f32)(j + 2);
        B[j + j * n] += 2.0f;
        Q[j + j * n] = 1.0f;
        Z[j + j * n] = 1.0f;
    }
    for (INT k = 1; k < n; k += 2) {
        select[k] = 1;
    }

    ctgsen(ijob, 1, 1, select, n, A, n, B, n, alpha, beta, Q, n, Z, n,
           &m, &pl, &pr, dif, &wq, -1, &iwq, -1, &info);
    assert_info_success(info);
    assert_int_equal(m, n / 2);
    lwork = (INT)crealf(wq);
    liwork = iwq;

    c64* work = malloc((lwork + 1) * sizeof(c64));
    INT* iwork = malloc(liwork * sizeof(INT));
    assert_non_null(work); assert_non_null(iwork);
    work[lwork] = CANARY;

    ctgsen(ijob, 1, 1, select, n, A, n, B, n, alpha, beta, Q, n, Z, n,
           &m, &pl, &pr, dif, work, lwork, iwork, liwork, &info);
    assert_info_success(info);

    if (work[lwork] != CANARY) {
        fail_msg("IJOB = %d, LWORK = %d: WORK(LWORK+1) was overwritten",
                 (int)ijob, (int)lwork);
    }
    if (ijob == 1 || ijob == 4) {
        assert_true(pl > 0.0f && pl <= 1.0f);
        assert_true(pr > 0.0f && pr <= 1.0f);
    }
    if (ijob == 2 || ijob == 4) {
        assert_true(dif[0] > 0.0f);
        assert_true(dif[1] > 0.0f);
    }

    free(A); free(B); free(Q); free(Z); free(alpha); free(beta);
    free(select); free(work); free(iwork);
}

static void test_ijob1(void** state)
{
    (void)state;
    run_tgsen(1);
}

static void test_ijob2(void** state)
{
    (void)state;
    run_tgsen(2);
}

static void test_ijob4(void** state)
{
    (void)state;
    run_tgsen(4);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_ijob1),
        cmocka_unit_test(test_ijob2),
        cmocka_unit_test(test_ijob4),
    };
    (void)cmocka_run_group_tests_name("ztgsen_lwork", tests, NULL, NULL);
    return 0;
}
