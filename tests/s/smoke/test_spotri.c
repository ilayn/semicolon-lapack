/**
 * @file test_spotri.c
 * @brief CMocka test suite for spotri (inverse using Cholesky factorization).
 *
 * Tests the matrix inverse routine spotri which computes the inverse of a
 * symmetric positive definite matrix using the Cholesky factorization
 * computed by spotrf.
 *
 * Verification: spot03 computes ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
/* Routines under test */
/*
 * Test fixture
 */
typedef struct {
    INT n;
    INT lda;
    f32* A;       /* Original matrix */
    f32* AINV;    /* Inverse */
    f32* d;
    f32* work;    /* Workspace for spot03 (n*n) and slatms */
    f32* rwork;
    uint64_t seed;
} dpotri_fixture_t;

static uint64_t g_seed = 5300;

static int dpotri_setup(void** state, INT n)
{
    dpotri_fixture_t* fix = malloc(sizeof(dpotri_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->AINV = malloc(fix->lda * n * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(fix->lda * n * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));

    assert_non_null(fix->A);
    assert_non_null(fix->AINV);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);

    *state = fix;
    return 0;
}

static int dpotri_teardown(void** state)
{
    dpotri_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AINV);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dpotri_setup(state, 5); }
static int setup_10(void** state) { return dpotri_setup(state, 10); }
static int setup_20(void** state) { return dpotri_setup(state, 20); }

/**
 * Core test logic: generate matrix, factorize, invert, verify.
 */
static f32 run_dpotri_test(dpotri_fixture_t* fix, INT imat, const char* uplo)
{
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    INT info;

    slatb4("SPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A to AINV for factorization */
    memcpy(fix->AINV, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Factor */
    spotrf(uplo, fix->n, fix->AINV, fix->lda, &info);
    assert_info_success(info);

    /* Compute inverse */
    spotri(uplo, fix->n, fix->AINV, fix->lda, &info);
    assert_info_success(info);

    /* Verify */
    f32 rcond, resid;
    spot03(uplo, fix->n, fix->A, fix->lda, fix->AINV, fix->lda,
           fix->work, fix->n, fix->rwork, &rcond, &resid);
    return resid;
}

static void test_dpotri_wellcond_upper(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dpotri_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

static void test_dpotri_wellcond_lower(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dpotri_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

static void test_dpotri_illcond_upper(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (INT imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dpotri_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

static void test_dpotri_illcond_lower(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (INT imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dpotri_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

#define DPOTRI_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dpotri_wellcond_upper, setup_fn, dpotri_teardown), \
    cmocka_unit_test_setup_teardown(test_dpotri_wellcond_lower, setup_fn, dpotri_teardown), \
    cmocka_unit_test_setup_teardown(test_dpotri_illcond_upper, setup_fn, dpotri_teardown), \
    cmocka_unit_test_setup_teardown(test_dpotri_illcond_lower, setup_fn, dpotri_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DPOTRI_TESTS(setup_5),
        DPOTRI_TESTS(setup_10),
        DPOTRI_TESTS(setup_20),
    };

    return cmocka_run_group_tests_name("dpotri", tests, NULL, NULL);
}
