/**
 * @file test_dgeqrf_work.c
 * @brief DGEQRF, DGELQF, DGERQF and DGEQLF regression test for WORK(1).
 *
 * The reference DGEQRF and DGELQF store WORK(1) only on a workspace query,
 * on the K = 0 quick return and on exit, so a call that fails with
 * INFO = -7 (LWORK too small) leaves WORK(1) unchanged. The reference
 * DGERQF and DGEQLF store the optimal LWORK in WORK(1) before testing
 * LWORK, so the same call returns it. Each case also checks WORK(1) = 1
 * for K = 0 and that INFO = -4 leaves WORK(1) unchanged in all four.
 */

#include "test_harness.h"
#include "verify.h"
#include <string.h>

#define GUARD (-12345.0)

typedef void (*fact_t)(const INT, const INT, f64* restrict, const INT,
                       f64* restrict, f64* restrict, const INT, INT*);

typedef struct {
    const char* name;
    fact_t fn;
    INT keeps_on_lwork_error;
} routine_t;

static const routine_t ROUTINES[4] = {
    {"DGEQRF", dgeqrf, 1},
    {"DGELQF", dgelqf, 1},
    {"DGERQF", dgerqf, 0},
    {"DGEQLF", dgeqlf, 0},
};

static void call_expect(const routine_t* r, const INT m, const INT n,
                        f64* A, const INT lda, f64* work, const INT lwork,
                        const INT infot, INT* info)
{
    f64 tau[8];
    strcpy(xerbla_srnamt, r->name);
    xerbla_infot = infot;
    xerbla_lerr = 0;
    xerbla_ok = 1;
    r->fn(m, n, A, lda, tau, work, lwork, info);
    const INT lerr = xerbla_lerr;
    const INT ok = xerbla_ok;
    xerbla_srnamt[0] = '\0';
    xerbla_lerr = 0;
    xerbla_ok = 1;
    if (infot != 0) {
        assert_true(lerr);
        assert_true(ok);
    } else {
        assert_false(lerr);
    }
}

static void test_lwork_error(void** state)
{
    (void)state;
    const INT m = 6, n = 6;
    for (INT ir = 0; ir < 4; ir++) {
        const routine_t* r = &ROUTINES[ir];
        f64 A[36], work[64];
        INT info;
        for (INT i = 0; i < m * n; i++) {
            A[i] = (f64)(i % 7) - 3.0;
        }

        call_expect(r, m, n, A, m, work, -1, 0, &info);
        assert_info_success(info);
        const f64 lwkopt = work[0];
        assert_true(lwkopt >= 1.0);

        work[0] = GUARD;
        call_expect(r, m, n, A, m, work, 1, 7, &info);
        assert_int_equal(info, -7);
        if (r->keeps_on_lwork_error) {
            if (work[0] != GUARD) {
                fail_msg("%s: INFO = -7 changed WORK(1) to %g", r->name,
                         (double)work[0]);
            }
        } else {
            if (work[0] != lwkopt) {
                fail_msg("%s: INFO = -7 left WORK(1) = %g, expected %g",
                         r->name, (double)work[0], (double)lwkopt);
            }
        }
    }
}

static void test_k0_and_lda(void** state)
{
    (void)state;
    for (INT ir = 0; ir < 4; ir++) {
        const routine_t* r = &ROUTINES[ir];
        f64 A[16], work[16];
        INT info;

        work[0] = GUARD;
        call_expect(r, 0, 4, A, 1, work, 1, 0, &info);
        assert_info_success(info);
        assert_true(work[0] == 1.0);

        work[0] = GUARD;
        call_expect(r, 4, 0, A, 4, work, 1, 0, &info);
        assert_info_success(info);
        assert_true(work[0] == 1.0);

        work[0] = GUARD;
        call_expect(r, 3, 3, A, 2, work, 16, 4, &info);
        assert_int_equal(info, -4);
        assert_true(work[0] == GUARD);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test(test_lwork_error),
        cmocka_unit_test(test_k0_and_lda),
    };
    (void)cmocka_run_group_tests_name("dgeqrf_work", tests, NULL, NULL);
    return 0;
}
