/**
 * @file test_dgecon.c
 * @brief CMocka test suite for dgecon (condition number estimation).
 *
 * Tests the condition number estimation routine dgecon which estimates
 * the reciprocal of the condition number using the LU factorization.
 *
 * Verification: dget06 computes the ratio of estimated to actual condition
 * numbers. A ratio close to 1 indicates a good estimate.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
/* Routines under test */
/* Utilities */
/*
 * Test fixture
 */
typedef struct {
    INT n;
    INT lda;
    f64 *A;       /* Original matrix */
    f64 *AFAC;    /* Factored matrix */
    f64 *AINV;    /* Inverse */
    f64 *d;       /* Singular values for dlatms */
    f64 *work;    /* Workspace */
    f64 *rwork;   /* Workspace for dlange */
    INT* ipiv;       /* Pivot indices */
    INT* iwork;      /* Integer workspace for dgecon */
    uint64_t seed;
} dgecon_fixture_t;

static uint64_t g_seed = 2718;

static int dgecon_setup(void **state, INT n)
{
    dgecon_fixture_t *fix = malloc(sizeof(dgecon_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    /* Workspace needs: dgetri needs n*n, dgecon needs 4*n, dlatms needs 3*n */
    INT lwork = (n * n > 4 * n) ? n * n : 4 * n;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->AFAC = malloc(fix->lda * n * sizeof(f64));
    fix->AINV = malloc(fix->lda * n * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(lwork * sizeof(f64));
    fix->rwork = malloc(n * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(INT));
    fix->iwork = malloc(n * sizeof(INT));

    assert_non_null(fix->A);
    assert_non_null(fix->AFAC);
    assert_non_null(fix->AINV);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

static int dgecon_teardown(void **state)
{
    dgecon_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AFAC);
        free(fix->AINV);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->ipiv);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

static int setup_2(void **state) { return dgecon_setup(state, 2); }
static int setup_3(void **state) { return dgecon_setup(state, 3); }
static int setup_5(void **state) { return dgecon_setup(state, 5); }
static int setup_10(void **state) { return dgecon_setup(state, 10); }
static int setup_20(void **state) { return dgecon_setup(state, 20); }

/**
 * Core test logic: generate matrix, compute true and estimated condition numbers.
 */
static void run_dgecon_test(dgecon_fixture_t *fix, INT imat)
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm_param, cndnum;
    INT info;
    INT lwork = fix->n * fix->n;

    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    memcpy(fix->AFAC, fix->A, fix->lda * fix->n * sizeof(f64));

    /* Factor A */
    dgetrf(fix->n, fix->n, fix->AFAC, fix->lda, fix->ipiv, &info);
    assert_info_success(info);

    /* Compute inverse for true condition number */
    memcpy(fix->AINV, fix->AFAC, fix->lda * fix->n * sizeof(f64));
    dgetri(fix->n, fix->AINV, fix->lda, fix->ipiv, fix->work, lwork, &info);
    assert_info_success(info);

    /* Test 1-norm condition number */
    f64 anorm_1 = dlange("1", fix->n, fix->n, fix->A, fix->lda, fix->rwork);
    f64 ainvnm_1 = dlange("1", fix->n, fix->n, fix->AINV, fix->lda, fix->rwork);
    f64 rcondc_1 = (anorm_1 > 0.0 && ainvnm_1 > 0.0) ?
                      (1.0 / anorm_1) / ainvnm_1 : 0.0;

    f64 rcond_est_1;
    dgecon("1", fix->n, fix->AFAC, fix->lda, anorm_1, &rcond_est_1,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc_1 > 0.0) {
        f64 ratio_1 = dget06(rcond_est_1, rcondc_1);
        assert_residual_ok(ratio_1);
    }

    /* Test infinity-norm condition number */
    f64 anorm_i = dlange("I", fix->n, fix->n, fix->A, fix->lda, fix->rwork);
    f64 ainvnm_i = dlange("I", fix->n, fix->n, fix->AINV, fix->lda, fix->rwork);
    f64 rcondc_i = (anorm_i > 0.0 && ainvnm_i > 0.0) ?
                      (1.0 / anorm_i) / ainvnm_i : 0.0;

    f64 rcond_est_i;
    dgecon("I", fix->n, fix->AFAC, fix->lda, anorm_i, &rcond_est_i,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc_i > 0.0) {
        f64 ratio_i = dget06(rcond_est_i, rcondc_i);
        assert_residual_ok(ratio_i);
    }
}

/*
 * Test well-conditioned matrices (types 1-4).
 */
static void test_dgecon_wellcond(void **state)
{
    dgecon_fixture_t *fix = *state;

    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        run_dgecon_test(fix, imat);
    }
}

/*
 * Test ill-conditioned matrices (types 8-9).
 * Only run for n >= 3.
 */
static void test_dgecon_illcond(void **state)
{
    dgecon_fixture_t *fix = *state;

    if (fix->n < 3) {
        skip();
    }

    for (INT imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        run_dgecon_test(fix, imat);
    }
}

#define DGECON_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgecon_wellcond, setup_fn, dgecon_teardown), \
    cmocka_unit_test_setup_teardown(test_dgecon_illcond, setup_fn, dgecon_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DGECON_TESTS(setup_2),
        DGECON_TESTS(setup_3),
        DGECON_TESTS(setup_5),
        DGECON_TESTS(setup_10),
        DGECON_TESTS(setup_20),
    };

    return cmocka_run_group_tests_name("dgecon", tests, NULL, NULL);
}
