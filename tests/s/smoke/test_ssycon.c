/**
 * @file test_ssycon.c
 * @brief CMocka test suite for ssycon (condition number estimation for
 *        symmetric indefinite matrices).
 *
 * Tests the condition number estimation routine ssycon which estimates
 * the reciprocal of the condition number using the Bunch-Kaufman
 * factorization from ssytrf.
 *
 * Verification: sget06 compares estimated vs true reciprocal condition number.
 * The true reciprocal condition number is 1/cndnum from slatb4.
 *
 * Tests both UPLO='U' and UPLO='L' for SSY matrix types 1-10.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - condition estimation needs looser tolerance (LAPACK dtest.in uses 30) */
#define THRESH 30.0f
/* Routines under test */
/*
 * Test fixture
 */
typedef struct {
    INT n;
    INT lda;
    f32* A;       /* Original matrix */
    f32* AFAC;    /* Factored matrix */
    INT* ipiv;       /* Pivot indices from ssytrf */
    f32* d;       /* Singular values for slatms */
    f32* work;    /* Workspace */
    f32* rwork;   /* Workspace for slansy */
    INT* iwork;      /* Integer workspace for ssycon */
    uint64_t seed;
} dsycon_fixture_t;

static uint64_t g_seed = 7400;

static int dsycon_setup(void** state, INT n)
{
    dsycon_fixture_t* fix = malloc(sizeof(dsycon_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    /* Workspace: max(n*64, 2*n) for ssytrf/ssycon, slatms needs 3*n */
    INT lwork = n * 64;
    if (lwork < 2 * n) {
        lwork = 2 * n;
    }
    if (lwork < 3 * n) {
        lwork = 3 * n;
    }

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->AFAC = malloc(fix->lda * n * sizeof(f32));
    fix->ipiv = malloc(n * sizeof(INT));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(lwork * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));
    fix->iwork = malloc(n * sizeof(INT));

    assert_non_null(fix->A);
    assert_non_null(fix->AFAC);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

static int dsycon_teardown(void** state)
{
    dsycon_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AFAC);
        free(fix->ipiv);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dsycon_setup(state, 5); }
static int setup_10(void** state) { return dsycon_setup(state, 10); }
static int setup_20(void** state) { return dsycon_setup(state, 20); }
static int setup_50(void** state) { return dsycon_setup(state, 50); }

/**
 * Core test logic: generate symmetric matrix, factor with ssytrf,
 * estimate condition number with ssycon, compare to true condition number.
 */
static void run_dsycon_test(dsycon_fixture_t* fix, INT imat, const char* uplo)
{
    char type, dist;
    INT kl, ku, mode;
    f32 anorm_param, cndnum;
    INT info;
    INT lwork = fix->n * 64;
    if (lwork < 2 * fix->n) {
        lwork = 2 * fix->n;
    }

    slatb4("SSY", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param,
           &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum,
           anorm_param, kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A into AFAC for factoring */
    memcpy(fix->AFAC, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Factor A via Bunch-Kaufman */
    ssytrf(uplo, fix->n, fix->AFAC, fix->lda, fix->ipiv, fix->work, lwork,
           &info);
    assert_info_success(info);

    /* Compute norm of original matrix */
    f32 anorm_1 = slansy("1", uplo, fix->n, fix->A, fix->lda, fix->rwork);

    /* True reciprocal condition number: 1/cndnum from slatb4 */
    f32 rcondc = 1.0f / cndnum;

    /* Estimate condition number via ssycon */
    f32 rcond_est;
    ssycon(uplo, fix->n, fix->AFAC, fix->lda, fix->ipiv, anorm_1, &rcond_est,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc > 0.0f) {
        f32 ratio = sget06(rcond_est, rcondc);
        assert_residual_ok(ratio);
    }
}

/*
 * Test well-conditioned matrices (types 1-6, cndnum = 2).
 * UPLO = 'U'
 */
static void test_dsycon_wellcond_upper(void** state)
{
    dsycon_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        run_dsycon_test(fix, imat, "U");
    }
}

/*
 * Test well-conditioned matrices (types 1-6, cndnum = 2).
 * UPLO = 'L'
 */
static void test_dsycon_wellcond_lower(void** state)
{
    dsycon_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        run_dsycon_test(fix, imat, "L");
    }
}

/*
 * Test ill-conditioned matrices (types 7-8).
 * UPLO = 'U'
 */
static void test_dsycon_illcond_upper(void** state)
{
    dsycon_fixture_t* fix = *state;
    for (INT imat = 7; imat <= 8; imat++) {
        fix->seed = g_seed++;
        run_dsycon_test(fix, imat, "U");
    }
}

/*
 * Test ill-conditioned matrices (types 7-8).
 * UPLO = 'L'
 */
static void test_dsycon_illcond_lower(void** state)
{
    dsycon_fixture_t* fix = *state;
    for (INT imat = 7; imat <= 8; imat++) {
        fix->seed = g_seed++;
        run_dsycon_test(fix, imat, "L");
    }
}

/*
 * Test scaled matrices (types 9-10: near underflow and overflow).
 * UPLO = 'U'
 */
static void test_dsycon_scaled_upper(void** state)
{
    dsycon_fixture_t* fix = *state;
    for (INT imat = 9; imat <= 10; imat++) {
        fix->seed = g_seed++;
        run_dsycon_test(fix, imat, "U");
    }
}

/*
 * Test scaled matrices (types 9-10: near underflow and overflow).
 * UPLO = 'L'
 */
static void test_dsycon_scaled_lower(void** state)
{
    dsycon_fixture_t* fix = *state;
    for (INT imat = 9; imat <= 10; imat++) {
        fix->seed = g_seed++;
        run_dsycon_test(fix, imat, "L");
    }
}

#define DSYCON_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsycon_wellcond_upper, setup_fn, dsycon_teardown), \
    cmocka_unit_test_setup_teardown(test_dsycon_wellcond_lower, setup_fn, dsycon_teardown), \
    cmocka_unit_test_setup_teardown(test_dsycon_illcond_upper, setup_fn, dsycon_teardown), \
    cmocka_unit_test_setup_teardown(test_dsycon_illcond_lower, setup_fn, dsycon_teardown), \
    cmocka_unit_test_setup_teardown(test_dsycon_scaled_upper, setup_fn, dsycon_teardown), \
    cmocka_unit_test_setup_teardown(test_dsycon_scaled_lower, setup_fn, dsycon_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSYCON_TESTS(setup_5),
        DSYCON_TESTS(setup_10),
        DSYCON_TESTS(setup_20),
        DSYCON_TESTS(setup_50),
    };

    return cmocka_run_group_tests_name("dsycon", tests, NULL, NULL);
}
