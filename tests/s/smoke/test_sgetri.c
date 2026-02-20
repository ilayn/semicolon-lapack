/**
 * @file test_sgetri.c
 * @brief CMocka test suite for sgetri (matrix inverse using LU factorization).
 *
 * Tests the matrix inverse routine sgetri which computes the inverse of a
 * matrix using the LU factorization computed by sgetrf.
 *
 * Verification: sget03 computes ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

/* Routines under test */
extern void sgetrf(const int m, const int n, f32 * const restrict A,
                   const int lda, int * const restrict ipiv, int *info);
extern void sgetri(const int n, f32 * const restrict A, const int lda,
                   const int * const restrict ipiv, f32 * const restrict work,
                   const int lwork, int *info);

/* Utilities */
extern f32 slamch(const char *cmach);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    f32 *A;       /* Original matrix */
    f32 *AINV;    /* Inverse */
    f32 *d;       /* Singular values for slatms */
    f32 *work;    /* Workspace */
    f32 *rwork;   /* Workspace for sget03 */
    int *ipiv;       /* Pivot indices */
    uint64_t seed;
} dgetri_fixture_t;

static uint64_t g_seed = 3141;

static int dgetri_setup(void **state, int n)
{
    dgetri_fixture_t *fix = malloc(sizeof(dgetri_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    int lwork = n * n;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->AINV = malloc(fix->lda * n * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(lwork * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));
    fix->ipiv = malloc(n * sizeof(int));

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
static f32 run_dgetri_test(dgetri_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info;
    int lwork = fix->n * fix->n;

    slatb4("SGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A to AINV for factorization */
    memcpy(fix->AINV, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Factor */
    sgetrf(fix->n, fix->n, fix->AINV, fix->lda, fix->ipiv, &info);
    assert_info_success(info);

    /* Compute inverse */
    sgetri(fix->n, fix->AINV, fix->lda, fix->ipiv, fix->work, lwork, &info);
    assert_info_success(info);

    /* Verify */
    f32 rcond, resid;
    sget03(fix->n, fix->A, fix->lda, fix->AINV, fix->lda,
           fix->work, fix->n, fix->rwork, &rcond, &resid);

    return resid;
}

/*
 * Test well-conditioned matrices (types 1-4).
 */
static void test_dgetri_wellcond(void **state)
{
    dgetri_fixture_t *fix = *state;
    f32 resid;

    for (int imat = 1; imat <= 4; imat++) {
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

    f32 resid;
    for (int imat = 8; imat <= 9; imat++) {
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

    f32 resid;
    for (int imat = 10; imat <= 11; imat++) {
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
