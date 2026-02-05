/**
 * @file test_dsycon.c
 * @brief CMocka test suite for dsycon (condition number estimation for
 *        symmetric indefinite matrices).
 *
 * Tests the condition number estimation routine dsycon which estimates
 * the reciprocal of the condition number using the Bunch-Kaufman
 * factorization from dsytrf.
 *
 * Verification: dget06 compares estimated vs true reciprocal condition number.
 * The true reciprocal condition number is 1/cndnum from dlatb4.
 *
 * Tests both UPLO='U' and UPLO='L' for DSY matrix types 1-10.
 */

#include "test_harness.h"

/* Test threshold - condition estimation needs looser tolerance (LAPACK dtest.in uses 30) */
#define THRESH 30.0
#include <cblas.h>

/* Routines under test */
extern void dsytrf(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* const restrict ipiv,
                   double* const restrict work, const int lwork, int* info);
extern void dsycon(const char* uplo, const int n,
                   const double* const restrict A, const int lda,
                   const int* const restrict ipiv, const double anorm,
                   double* rcond, double* const restrict work,
                   int* const restrict iwork, int* info);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);

/* Verification routine */
extern double dget06(const double rcond, const double rcondc);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   uint64_t seed, const char* sym, double* d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    double* A;       /* Original matrix */
    double* AFAC;    /* Factored matrix */
    int* ipiv;       /* Pivot indices from dsytrf */
    double* d;       /* Singular values for dlatms */
    double* work;    /* Workspace */
    double* rwork;   /* Workspace for dlansy */
    int* iwork;      /* Integer workspace for dsycon */
    uint64_t seed;
} dsycon_fixture_t;

static uint64_t g_seed = 7400;

static int dsycon_setup(void** state, int n)
{
    dsycon_fixture_t* fix = malloc(sizeof(dsycon_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    /* Workspace: max(n*64, 2*n) for dsytrf/dsycon, dlatms needs 3*n */
    int lwork = n * 64;
    if (lwork < 2 * n) {
        lwork = 2 * n;
    }
    if (lwork < 3 * n) {
        lwork = 3 * n;
    }

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AFAC = malloc(fix->lda * n * sizeof(double));
    fix->ipiv = malloc(n * sizeof(int));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(lwork * sizeof(double));
    fix->rwork = malloc(n * sizeof(double));
    fix->iwork = malloc(n * sizeof(int));

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
 * Core test logic: generate symmetric matrix, factor with dsytrf,
 * estimate condition number with dsycon, compare to true condition number.
 */
static void run_dsycon_test(dsycon_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;
    int info;
    int lwork = fix->n * 64;
    if (lwork < 2 * fix->n) {
        lwork = 2 * fix->n;
    }

    dlatb4("DSY", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param,
           &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    dlatms(fix->n, fix->n, &dist, fix->seed, sym_str, fix->d, mode, cndnum,
           anorm_param, kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    /* Copy A into AFAC for factoring */
    memcpy(fix->AFAC, fix->A, fix->lda * fix->n * sizeof(double));

    /* Factor A via Bunch-Kaufman */
    dsytrf(uplo, fix->n, fix->AFAC, fix->lda, fix->ipiv, fix->work, lwork,
           &info);
    assert_info_success(info);

    /* Compute norm of original matrix */
    double anorm_1 = dlansy("1", uplo, fix->n, fix->A, fix->lda, fix->rwork);

    /* True reciprocal condition number: 1/cndnum from dlatb4 */
    double rcondc = 1.0 / cndnum;

    /* Estimate condition number via dsycon */
    double rcond_est;
    dsycon(uplo, fix->n, fix->AFAC, fix->lda, fix->ipiv, anorm_1, &rcond_est,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc > 0.0) {
        double ratio = dget06(rcond_est, rcondc);
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
    for (int imat = 1; imat <= 6; imat++) {
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
    for (int imat = 1; imat <= 6; imat++) {
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
    for (int imat = 7; imat <= 8; imat++) {
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
    for (int imat = 7; imat <= 8; imat++) {
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
    for (int imat = 9; imat <= 10; imat++) {
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
    for (int imat = 9; imat <= 10; imat++) {
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
