/**
 * @file test_sgelqf.c
 * @brief CMocka test suite for sgelqf (LQ factorization).
 *
 * Verification:
 *   slqt01: norm(L - A*Q') / (N * norm(A) * eps) and
 *           norm(I - Q*Q') / (N * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

typedef struct {
    int m, n;
    int lda;
    f32 *A, *AF, *Q, *L;
    f32 *tau, *work, *rwork;
    f32 *d, *genwork;
    int lwork;
    uint64_t seed;
} lq_fixture_t;

static uint64_t g_seed = 3001;

static int lq_setup(void **state, int m, int n)
{
    lq_fixture_t *fix = malloc(sizeof(lq_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    int maxmn = m > n ? m : n;
    fix->lda = maxmn;
    fix->seed = g_seed++;

    int minmn = m < n ? m : n;
    fix->lwork = maxmn * 64;

    fix->A = calloc(fix->lda * maxmn, sizeof(f32));
    fix->AF = calloc(fix->lda * maxmn, sizeof(f32));
    fix->Q = calloc(fix->lda * maxmn, sizeof(f32));
    fix->L = calloc(fix->lda * maxmn, sizeof(f32));
    fix->tau = calloc(minmn, sizeof(f32));
    fix->work = calloc(fix->lwork, sizeof(f32));
    fix->rwork = calloc(maxmn, sizeof(f32));
    fix->d = calloc(minmn, sizeof(f32));
    fix->genwork = calloc(3 * maxmn, sizeof(f32));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->Q);
    assert_non_null(fix->L);
    assert_non_null(fix->tau);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);

    *state = fix;
    return 0;
}

static int lq_teardown(void **state)
{
    lq_fixture_t *fix = *state;
    if (fix) {
        free(fix->A); free(fix->AF); free(fix->Q); free(fix->L);
        free(fix->tau); free(fix->work); free(fix->rwork);
        free(fix->d); free(fix->genwork);
        free(fix);
    }
    return 0;
}

static int setup_5x5(void **state) { return lq_setup(state, 5, 5); }
static int setup_10x10(void **state) { return lq_setup(state, 10, 10); }
static int setup_20x20(void **state) { return lq_setup(state, 20, 20); }
static int setup_50x50(void **state) { return lq_setup(state, 50, 50); }
static int setup_10x5(void **state) { return lq_setup(state, 10, 5); }
static int setup_20x10(void **state) { return lq_setup(state, 20, 10); }
static int setup_5x10(void **state) { return lq_setup(state, 5, 10); }
static int setup_10x20(void **state) { return lq_setup(state, 10, 20); }

static void run_lqt01(lq_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode, info;
    f32 anorm, cndnum;
    f32 result[2];

    slatb4("SGE", imat, fix->m, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->m, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
    assert_int_equal(info, 0);

    slqt01(fix->m, fix->n, fix->A, fix->AF, fix->Q, fix->L, fix->lda,
           fix->tau, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

static void test_wellcond(void **state)
{
    lq_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_lqt01(fix, imat);
    }
}

static void test_illcond(void **state)
{
    lq_fixture_t *fix = *state;
    if (fix->m < 3) { skip_test("requires m >= 3"); return; }
    for (int imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_lqt01(fix, imat);
    }
}

static void test_scaled(void **state)
{
    lq_fixture_t *fix = *state;
    for (int imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_lqt01(fix, imat);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x5, lq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x10, lq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x20, lq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_50x50, lq_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x10, lq_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_20x20, lq_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_10x10, lq_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_20x20, lq_teardown),
        /* Tall (m > n): min(m,n)=n reflectors */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x5, lq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x10, lq_teardown),
        /* Wide (m < n): min(m,n)=m reflectors */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x10, lq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x20, lq_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x20, lq_teardown),
    };
    return cmocka_run_group_tests_name("dgelqf", tests, NULL, NULL);
}
