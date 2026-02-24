/**
 * @file test_dpocon.c
 * @brief CMocka test suite for dpocon (condition number estimation).
 *
 * Tests the condition number estimation routine dpocon which estimates the
 * reciprocal of the condition number of a symmetric positive definite matrix
 * using the Cholesky factorization.
 *
 * Verification: dget06 compares estimated vs true reciprocal condition number.
 *
 * Tests both UPLO='U' and UPLO='L' for DPO matrix types 1-7.
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
    f64* A;       /* Original matrix */
    f64* AFAC;    /* Factored matrix */
    f64* AINV;    /* Inverse for true condition number */
    f64* d;
    f64* work;
    f64* rwork;
    INT* iwork;
    uint64_t seed;
} dpocon_fixture_t;

static uint64_t g_seed = 5400;

static int dpocon_setup(void** state, INT n)
{
    dpocon_fixture_t* fix = malloc(sizeof(dpocon_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->AFAC = malloc(fix->lda * n * sizeof(f64));
    fix->AINV = malloc(fix->lda * n * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(3 * n * sizeof(f64));
    fix->rwork = malloc(n * sizeof(f64));
    fix->iwork = malloc(n * sizeof(INT));

    assert_non_null(fix->A);
    assert_non_null(fix->AFAC);
    assert_non_null(fix->AINV);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

static int dpocon_teardown(void** state)
{
    dpocon_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AFAC);
        free(fix->AINV);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dpocon_setup(state, 5); }
static int setup_10(void** state) { return dpocon_setup(state, 10); }
static int setup_20(void** state) { return dpocon_setup(state, 20); }

/**
 * Core test logic: generate matrix, compute true and estimated condition numbers.
 */
static void run_dpocon_test(dpocon_fixture_t* fix, INT imat, const char* uplo)
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm_param, cndnum;
    INT info;

    dlatb4("DPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Factor A */
    memcpy(fix->AFAC, fix->A, fix->lda * fix->n * sizeof(f64));
    dpotrf(uplo, fix->n, fix->AFAC, fix->lda, &info);
    assert_info_success(info);

    /* Compute inverse for true condition number */
    memcpy(fix->AINV, fix->AFAC, fix->lda * fix->n * sizeof(f64));
    dpotri(uplo, fix->n, fix->AINV, fix->lda, &info);
    assert_info_success(info);

    /* Compute true reciprocal condition number */
    f64 anorm_1 = dlansy("1", uplo, fix->n, fix->A, fix->lda, fix->rwork);
    f64 ainvnm_1 = dlansy("1", uplo, fix->n, fix->AINV, fix->lda, fix->rwork);
    f64 rcondc = (anorm_1 > 0.0 && ainvnm_1 > 0.0) ?
                    (1.0 / anorm_1) / ainvnm_1 : 0.0;

    /* Estimate condition number */
    f64 rcond_est;
    dpocon(uplo, fix->n, fix->AFAC, fix->lda, anorm_1, &rcond_est,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc > 0.0) {
        f64 ratio = dget06(rcond_est, rcondc);
        assert_residual_ok(ratio);
    }
}

static void test_dpocon_wellcond_upper(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dpocon_test(fix, imat, "U");
    }
}

static void test_dpocon_wellcond_lower(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dpocon_test(fix, imat, "L");
    }
}

static void test_dpocon_illcond_upper(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (INT imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        run_dpocon_test(fix, imat, "U");
    }
}

static void test_dpocon_illcond_lower(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (INT imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        run_dpocon_test(fix, imat, "L");
    }
}

#define DPOCON_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dpocon_wellcond_upper, setup_fn, dpocon_teardown), \
    cmocka_unit_test_setup_teardown(test_dpocon_wellcond_lower, setup_fn, dpocon_teardown), \
    cmocka_unit_test_setup_teardown(test_dpocon_illcond_upper, setup_fn, dpocon_teardown), \
    cmocka_unit_test_setup_teardown(test_dpocon_illcond_lower, setup_fn, dpocon_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DPOCON_TESTS(setup_5),
        DPOCON_TESTS(setup_10),
        DPOCON_TESTS(setup_20),
    };

    return cmocka_run_group_tests_name("dpocon", tests, NULL, NULL);
}
