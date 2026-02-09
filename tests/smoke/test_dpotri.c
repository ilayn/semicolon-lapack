/**
 * @file test_dpotri.c
 * @brief CMocka test suite for dpotri (inverse using Cholesky factorization).
 *
 * Tests the matrix inverse routine dpotri which computes the inverse of a
 * symmetric positive definite matrix using the Cholesky factorization
 * computed by dpotrf.
 *
 * Verification: dpot03 computes ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dpotrf(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* info);
extern void dpotri(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* info);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    double* A;       /* Original matrix */
    double* AINV;    /* Inverse */
    double* d;
    double* work;    /* Workspace for dpot03 (n*n) and dlatms */
    double* rwork;
    uint64_t seed;
} dpotri_fixture_t;

static uint64_t g_seed = 5300;

static int dpotri_setup(void** state, int n)
{
    dpotri_fixture_t* fix = malloc(sizeof(dpotri_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AINV = malloc(fix->lda * n * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(fix->lda * n * sizeof(double));
    fix->rwork = malloc(n * sizeof(double));

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
static double run_dpotri_test(dpotri_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;

    dlatb4("DPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A to AINV for factorization */
    memcpy(fix->AINV, fix->A, fix->lda * fix->n * sizeof(double));

    /* Factor */
    dpotrf(uplo, fix->n, fix->AINV, fix->lda, &info);
    assert_info_success(info);

    /* Compute inverse */
    dpotri(uplo, fix->n, fix->AINV, fix->lda, &info);
    assert_info_success(info);

    /* Verify */
    double rcond, resid;
    dpot03(uplo, fix->n, fix->A, fix->lda, fix->AINV, fix->lda,
           fix->work, fix->n, fix->rwork, &rcond, &resid);
    return resid;
}

static void test_dpotri_wellcond_upper(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        double resid = run_dpotri_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

static void test_dpotri_wellcond_lower(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        double resid = run_dpotri_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

static void test_dpotri_illcond_upper(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (int imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        double resid = run_dpotri_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

static void test_dpotri_illcond_lower(void** state)
{
    dpotri_fixture_t* fix = *state;
    for (int imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        double resid = run_dpotri_test(fix, imat, "L");
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
