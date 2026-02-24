/**
 * @file test_dgetri.c
 * @brief CMocka test suite for dgetri (matrix inverse using LU factorization).
 *
 * Tests the matrix inverse routine dgetri which computes the inverse of a
 * matrix using the LU factorization computed by dgetrf.
 *
 * Verification: dget03 computes ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
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
    f64 *AINV;    /* Inverse */
    f64 *d;       /* Singular values for dlatms */
    f64 *work;    /* Workspace */
    f64 *rwork;   /* Workspace for dget03 */
    INT* ipiv;       /* Pivot indices */
    uint64_t seed;
} dgetri_fixture_t;

static uint64_t g_seed = 3141;

static int dgetri_setup(void **state, INT n)
{
    dgetri_fixture_t *fix = malloc(sizeof(dgetri_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    INT lwork = n * n;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->AINV = malloc(fix->lda * n * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(lwork * sizeof(f64));
    fix->rwork = malloc(n * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(INT));

    assert_non_null(fix->A);
    assert_non_null(fix->AINV);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->ipiv);

    *state = fix;
    return 0;
}

static int dgetri_teardown(void **state)
{
    dgetri_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AINV);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->ipiv);
        free(fix);
    }
    return 0;
}

static int setup_2(void **state) { return dgetri_setup(state, 2); }
static int setup_3(void **state) { return dgetri_setup(state, 3); }
static int setup_5(void **state) { return dgetri_setup(state, 5); }
static int setup_10(void **state) { return dgetri_setup(state, 10); }
static int setup_20(void **state) { return dgetri_setup(state, 20); }

/**
 * Core test logic: generate matrix, factorize, invert, verify.
 */
static f64 run_dgetri_test(dgetri_fixture_t *fix, INT imat)
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info;
    INT lwork = fix->n * fix->n;

    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A to AINV for factorization */
    memcpy(fix->AINV, fix->A, fix->lda * fix->n * sizeof(f64));

    /* Factor */
    dgetrf(fix->n, fix->n, fix->AINV, fix->lda, fix->ipiv, &info);
    assert_info_success(info);

    /* Compute inverse */
    dgetri(fix->n, fix->AINV, fix->lda, fix->ipiv, fix->work, lwork, &info);
    assert_info_success(info);

    /* Verify */
    f64 rcond, resid;
    dget03(fix->n, fix->A, fix->lda, fix->AINV, fix->lda,
           fix->work, fix->n, fix->rwork, &rcond, &resid);

    return resid;
}

/*
 * Test well-conditioned matrices (types 1-4).
 */
static void test_dgetri_wellcond(void **state)
{
    dgetri_fixture_t *fix = *state;
    f64 resid;

    for (INT imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        resid = run_dgetri_test(fix, imat);
        assert_residual_ok(resid);
    }
}

/*
 * Test ill-conditioned matrices (types 8-9).
 * Only run for n >= 3.
 */
static void test_dgetri_illcond(void **state)
{
    dgetri_fixture_t *fix = *state;

    if (fix->n < 3) {
        skip();
    }

    f64 resid;
    for (INT imat = 8; imat <= 9; imat++) {
        fix->seed = g_seed++;
        resid = run_dgetri_test(fix, imat);
        assert_residual_ok(resid);
    }
}

/*
 * Test scaled matrices (types 10-11).
 * Only run for n >= 2.
 */
static void test_dgetri_scaled(void **state)
{
    dgetri_fixture_t *fix = *state;

    if (fix->n < 2) {
        skip();
    }

    f64 resid;
    for (INT imat = 10; imat <= 11; imat++) {
        fix->seed = g_seed++;
        resid = run_dgetri_test(fix, imat);
        assert_residual_ok(resid);
    }
}

#define DGETRI_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgetri_wellcond, setup_fn, dgetri_teardown), \
    cmocka_unit_test_setup_teardown(test_dgetri_illcond, setup_fn, dgetri_teardown), \
    cmocka_unit_test_setup_teardown(test_dgetri_scaled, setup_fn, dgetri_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DGETRI_TESTS(setup_2),
        DGETRI_TESTS(setup_3),
        DGETRI_TESTS(setup_5),
        DGETRI_TESTS(setup_10),
        DGETRI_TESTS(setup_20),
    };

    return cmocka_run_group_tests_name("dgetri", tests, NULL, NULL);
}
