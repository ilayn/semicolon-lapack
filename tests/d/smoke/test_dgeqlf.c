/**
 * @file test_dgeqlf.c
 * @brief CMocka test suite for dgeqlf (QL factorization).
 *
 * Verification:
 *   dqlt01: norm(L - Q'*A) / (M * norm(A) * eps) and
 *           norm(I - Q'*Q) / (M * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
typedef struct {
    INT m, n;
    INT lda;
    f64 *A, *AF, *Q, *L;
    f64 *tau, *work, *rwork;
    f64 *d, *genwork;
    INT lwork;
    uint64_t seed;
} ql_fixture_t;

static uint64_t g_seed = 4001;

static int ql_setup(void **state, INT m, INT n)
{
    ql_fixture_t *fix = malloc(sizeof(ql_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    INT maxmn = m > n ? m : n;
    fix->lda = maxmn;
    fix->seed = g_seed++;

    INT minmn = m < n ? m : n;
    fix->lwork = maxmn * 64;

    fix->A = calloc(fix->lda * maxmn, sizeof(f64));
    fix->AF = calloc(fix->lda * maxmn, sizeof(f64));
    fix->Q = calloc(fix->lda * maxmn, sizeof(f64));
    fix->L = calloc(fix->lda * maxmn, sizeof(f64));
    fix->tau = calloc(minmn, sizeof(f64));
    fix->work = calloc(fix->lwork, sizeof(f64));
    fix->rwork = calloc(maxmn, sizeof(f64));
    fix->d = calloc(minmn, sizeof(f64));
    fix->genwork = calloc(3 * maxmn, sizeof(f64));

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

static int ql_teardown(void **state)
{
    ql_fixture_t *fix = *state;
    if (fix) {
        free(fix->A); free(fix->AF); free(fix->Q); free(fix->L);
        free(fix->tau); free(fix->work); free(fix->rwork);
        free(fix->d); free(fix->genwork);
        free(fix);
    }
    return 0;
}

static int setup_5x5(void **state) { return ql_setup(state, 5, 5); }
static int setup_10x10(void **state) { return ql_setup(state, 10, 10); }
static int setup_20x20(void **state) { return ql_setup(state, 20, 20); }
static int setup_50x50(void **state) { return ql_setup(state, 50, 50); }
static int setup_10x5(void **state) { return ql_setup(state, 10, 5); }
static int setup_20x10(void **state) { return ql_setup(state, 20, 10); }
static int setup_5x10(void **state) { return ql_setup(state, 5, 10); }
static int setup_10x20(void **state) { return ql_setup(state, 10, 20); }

static void run_qlt01(ql_fixture_t *fix, INT imat)
{
    char type, dist;
    INT kl, ku, mode, info;
    f64 anorm, cndnum;
    f64 result[2];

    dlatb4("DGE", imat, fix->m, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->m, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
    assert_int_equal(info, 0);

    dqlt01(fix->m, fix->n, fix->A, fix->AF, fix->Q, fix->L, fix->lda,
           fix->tau, fix->work, fix->lwork, fix->rwork, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

static void test_wellcond(void **state)
{
    ql_fixture_t *fix = *state;
    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_qlt01(fix, imat);
    }
}

static void test_illcond(void **state)
{
    ql_fixture_t *fix = *state;
    if (fix->m < 3) { skip_test("requires m >= 3"); return; }
    for (INT imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_qlt01(fix, imat);
    }
}

static void test_scaled(void **state)
{
    ql_fixture_t *fix = *state;
    for (INT imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        run_qlt01(fix, imat);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x5, ql_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x10, ql_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x20, ql_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_50x50, ql_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_10x10, ql_teardown),
        cmocka_unit_test_setup_teardown(test_illcond, setup_20x20, ql_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_10x10, ql_teardown),
        cmocka_unit_test_setup_teardown(test_scaled, setup_20x20, ql_teardown),
        /* Tall */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x5, ql_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_20x10, ql_teardown),
        /* Wide */
        cmocka_unit_test_setup_teardown(test_wellcond, setup_5x10, ql_teardown),
        cmocka_unit_test_setup_teardown(test_wellcond, setup_10x20, ql_teardown),
    };
    return cmocka_run_group_tests_name("dgeqlf", tests, NULL, NULL);
}
