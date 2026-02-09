/**
 * @file test_dgerqf.c
 * @brief CMocka test suite for dgerqf (RQ factorization).
 *
 * Verification:
 *   drqt01: norm(R - A*Q') / (N * norm(A) * eps) and
 *           norm(I - Q*Q') / (N * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

typedef struct {
    int m, n;
    int lda;
    double *A, *AF, *Q, *R;
    double *tau, *work, *rwork;
    double *d, *genwork;
    int lwork;
    uint64_t seed;
} rq_fixture_t;

static uint64_t g_seed = 5001;

static int rq_setup(void **state, int m, int n)
{
    rq_fixture_t *fix = malloc(sizeof(rq_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    int maxmn = m > n ? m : n;
    fix->lda = maxmn;
    fix->seed = g_seed++;

    int minmn = m < n ? m : n;
    fix->lwork = maxmn * 64;

    fix->A = calloc(fix->lda * maxmn, sizeof(double));
    fix->AF = calloc(fix->lda * maxmn, sizeof(double));
    fix->Q = calloc(fix->lda * maxmn, sizeof(double));
    fix->R = calloc(fix->lda * maxmn, sizeof(double));
    fix->tau = calloc(minmn, sizeof(double));
    fix->work = calloc(fix->lwork, sizeof(double));
    fix->rwork = calloc(maxmn, sizeof(double));
    fix->d = calloc(minmn, sizeof(double));
    fix->genwork = calloc(3 * maxmn, sizeof(double));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->Q);
    assert_non_null(fix->R);
    assert_non_null(fix->tau);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);

    *state = fix;
    return 0;
}

static int rq_teardown(void **state)
{
    rq_fixture_t *fix = *state;
    if (fix) {
        free(fix->A); free(fix->AF); free(fix->Q); free(fix->R);
        free(fix->tau); free(fix->work); free(fix->rwork);
        free(fix->d); free(fix->genwork);
        free(fix);
    }
    return 0;
}

static int setup_5x5(void **state) { return rq_setup(state, 5, 5); }
static int setup_10x10(void **state) { return rq_setup(state, 10, 10); }
static int setup_20x20(void **state) { return rq_setup(state, 20, 20); }
static int setup_50x50(void **state) { return rq_setup(state, 50, 50); }
static int setup_10x5(void **state) { return rq_setup(state, 10, 5); }
static int setup_20x10(void **state) { return rq_setup(state, 20, 10); }
static int setup_5x10(void **state) { return rq_setup(state, 5, 10); }
static int setup_10x20(void **state) { return rq_setup(state, 10, 20); }

static void run_rqt01(rq_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode, info;
    double anorm, cndnum;
    double result[2];

    dlatb4("DGE", imat, fix->m, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->m, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
    assert_int_equal(info, 0);

    drqt01(fix->m, fix->n, fix->A, fix->AF, fix->Q, fix->R, fix->lda,
           fix->tau, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

static void test_wellcond(void **state)
{
    rq_fixture_t *fix = *state;
    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_rqt01(fix, imat);
    }
}

static void test_illcond(void **state)
{
    rq_fixture_t *fix = *state;
    if (fix->m < 3) { skip_test("requires m >= 3"); return; }
    for (int imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_rqt01(fix, imat);
    }
}

static void test_scaled(void **state)
{
    rq_fixture_t *fix = *state;
    for (int imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_rqt01(fix, imat);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x5, rq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x10, rq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x20, rq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_50x50, rq_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x10, rq_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_20x20, rq_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_10x10, rq_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_20x20, rq_teardown),
        /* Tall (m > n) */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x5, rq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x10, rq_teardown),
        /* Wide (m < n) */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x10, rq_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x20, rq_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x20, rq_teardown),
    };
    return cmocka_run_group_tests_name("dgerqf", tests, NULL, NULL);
}
