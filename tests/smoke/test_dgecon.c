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
#include <cblas.h>

/* Routines under test */
extern void dgetrf(const int m, const int n, double * const restrict A,
                   const int lda, int * const restrict ipiv, int *info);
extern void dgecon(const char *norm, const int n, const double * const restrict A,
                   const int lda, const double anorm, double *rcond,
                   double * const restrict work, int * const restrict iwork,
                   int *info);
extern void dgetri(const int n, double * const restrict A, const int lda,
                   const int * const restrict ipiv, double * const restrict work,
                   const int lwork, int *info);

/* Utilities */
extern double dlamch(const char *cmach);
extern double dlange(const char *norm, const int m, const int n,
                     const double * const restrict A, const int lda,
                     double * const restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    double *A;       /* Original matrix */
    double *AFAC;    /* Factored matrix */
    double *AINV;    /* Inverse */
    double *d;       /* Singular values for dlatms */
    double *work;    /* Workspace */
    double *rwork;   /* Workspace for dlange */
    int *ipiv;       /* Pivot indices */
    int *iwork;      /* Integer workspace for dgecon */
    uint64_t seed;
} dgecon_fixture_t;

static uint64_t g_seed = 2718;

static int dgecon_setup(void **state, int n)
{
    dgecon_fixture_t *fix = malloc(sizeof(dgecon_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    /* Workspace needs: dgetri needs n*n, dgecon needs 4*n, dlatms needs 3*n */
    int lwork = (n * n > 4 * n) ? n * n : 4 * n;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AFAC = malloc(fix->lda * n * sizeof(double));
    fix->AINV = malloc(fix->lda * n * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(lwork * sizeof(double));
    fix->rwork = malloc(n * sizeof(double));
    fix->ipiv = malloc(n * sizeof(int));
    fix->iwork = malloc(n * sizeof(int));

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
static void run_dgecon_test(dgecon_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;
    int info;
    int lwork = fix->n * fix->n;

    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    memcpy(fix->AFAC, fix->A, fix->lda * fix->n * sizeof(double));

    /* Factor A */
    dgetrf(fix->n, fix->n, fix->AFAC, fix->lda, fix->ipiv, &info);
    assert_info_success(info);

    /* Compute inverse for true condition number */
    memcpy(fix->AINV, fix->AFAC, fix->lda * fix->n * sizeof(double));
    dgetri(fix->n, fix->AINV, fix->lda, fix->ipiv, fix->work, lwork, &info);
    assert_info_success(info);

    /* Test 1-norm condition number */
    double anorm_1 = dlange("1", fix->n, fix->n, fix->A, fix->lda, fix->rwork);
    double ainvnm_1 = dlange("1", fix->n, fix->n, fix->AINV, fix->lda, fix->rwork);
    double rcondc_1 = (anorm_1 > 0.0 && ainvnm_1 > 0.0) ?
                      (1.0 / anorm_1) / ainvnm_1 : 0.0;

    double rcond_est_1;
    dgecon("1", fix->n, fix->AFAC, fix->lda, anorm_1, &rcond_est_1,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc_1 > 0.0) {
        double ratio_1 = dget06(rcond_est_1, rcondc_1);
        assert_residual_ok(ratio_1);
    }

    /* Test infinity-norm condition number */
    double anorm_i = dlange("I", fix->n, fix->n, fix->A, fix->lda, fix->rwork);
    double ainvnm_i = dlange("I", fix->n, fix->n, fix->AINV, fix->lda, fix->rwork);
    double rcondc_i = (anorm_i > 0.0 && ainvnm_i > 0.0) ?
                      (1.0 / anorm_i) / ainvnm_i : 0.0;

    double rcond_est_i;
    dgecon("I", fix->n, fix->AFAC, fix->lda, anorm_i, &rcond_est_i,
           fix->work, fix->iwork, &info);
    assert_info_success(info);

    if (rcondc_i > 0.0) {
        double ratio_i = dget06(rcond_est_i, rcondc_i);
        assert_residual_ok(ratio_i);
    }
}

/*
 * Test well-conditioned matrices (types 1-4).
 */
static void test_dgecon_wellcond(void **state)
{
    dgecon_fixture_t *fix = *state;

    for (int imat = 1; imat <= 4; imat++) {
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

    for (int imat = 8; imat <= 9; imat++) {
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
