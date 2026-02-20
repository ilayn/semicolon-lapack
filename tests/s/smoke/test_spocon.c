/**
 * @file test_spocon.c
 * @brief CMocka test suite for spocon (condition number estimation).
 *
 * Tests the condition number estimation routine spocon which estimates the
 * reciprocal of the condition number of a symmetric positive definite matrix
 * using the Cholesky factorization.
 *
 * Verification: sget06 compares estimated vs true reciprocal condition number.
 *
 * Tests both UPLO='U' and UPLO='L' for SPO matrix types 1-7.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

/* Routines under test */
extern void spotrf(const char* uplo, const int n, f32* const restrict A,
                   const int lda, int* info);
extern void spotri(const char* uplo, const int n, f32* const restrict A,
                   const int lda, int* info);
extern void spocon(const char* uplo, const int n,
                   const f32* const restrict A, const int lda,
                   const f32 anorm, f32* rcond,
                   f32* const restrict work, int* const restrict iwork,
                   int* info);

/* Utilities */
extern f32 slamch(const char* cmach);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    f32* A;       /* Original matrix */
    f32* AFAC;    /* Factored matrix */
    f32* AINV;    /* Inverse for true condition number */
    f32* d;
    f32* work;
    f32* rwork;
    int* iwork;
    uint64_t seed;
} dpocon_fixture_t;

static uint64_t g_seed = 5400;

static int dpocon_setup(void** state, int n)
{
    dpocon_fixture_t* fix = malloc(sizeof(dpocon_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->AFAC = malloc(fix->lda * n * sizeof(f32));
    fix->AINV = malloc(fix->lda * n * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(3 * n * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));
    fix->iwork = malloc(n * sizeof(int));

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
static void run_dpocon_test(dpocon_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm_param, cndnum;
    int info;

    slatb4("SPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Factor A */
    memcpy(fix->AFAC, fix->A, fix->lda * fix->n * sizeof(f32));
    spotrf(uplo, fix->n, fix->AFAC, fix->lda, &info);
    assert_info_success(info);

    /* Compute inverse for true condition number */
    memcpy(fix->AINV, fix->AFAC, fix->lda * fix->n * sizeof(f32));
    spotri(uplo, fix->n, fix->AINV, fix->lda, &info);
    assert_info_success(info);

    /* Compute true reciprocal condition number */
    f32 anorm_1 = slansy("1", uplo, fix->n, fix->A, fix->lda, fix->rwork);
    f32 ainvnm_1 = slansy("1", uplo, fix->n, fix->AINV, fix->lda, fix->rwork);
    f32 rcondc = (anorm_1 > 0.0f && ainvnm_1 > 0.0f) ?
                    (1.0f / anorm_1) / ainvnm_1 : 0.0f;

    /* Estimate condition number */
    f32 rcond_est;
    spocon(uplo, fix->n, fix->AFAC, fix->lda, anorm_1, &rcond_est,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc > 0.0f) {
        f32 ratio = sget06(rcond_est, rcondc);
        assert_residual_ok(ratio);
    }
}

static void test_dpocon_wellcond_upper(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dpocon_test(fix, imat, "U");
    }
}

static void test_dpocon_wellcond_lower(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dpocon_test(fix, imat, "L");
    }
}

static void test_dpocon_illcond_upper(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (int imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        run_dpocon_test(fix, imat, "U");
    }
}

static void test_dpocon_illcond_lower(void** state)
{
    dpocon_fixture_t* fix = *state;
    for (int imat = 6; imat <= 7; imat++) {
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
