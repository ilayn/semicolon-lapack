/**
 * @file test_dsytri.c
 * @brief CMocka test suite for dsytri (symmetric indefinite matrix inverse).
 *
 * Tests the matrix inverse routine dsytri which computes the inverse of a
 * symmetric indefinite matrix using the factorization computed by dsytrf.
 *
 * Verification: dget03 computes ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps)
 *
 * Since dsytri only computes the upper or lower triangle of the inverse,
 * the result must be expanded to a full symmetric matrix before calling dget03.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dsytrf(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* const restrict ipiv,
                   double* const restrict work, const int lwork, int* info);
extern void dsytri(const char* uplo, const int n, double* const restrict A,
                   const int lda, const int* const restrict ipiv,
                   double* const restrict work, int* info);

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    double* A;         /* Original matrix */
    double* AFAC;      /* Factored matrix (overwritten by dsytri) */
    double* AINV;      /* Full symmetric inverse */
    double* work_fac;  /* Workspace for dsytrf */
    double* work_inv;  /* Workspace for dsytri */
    double* work_ver;  /* Workspace for dget03 */
    double* rwork;     /* Workspace for dget03 */
    double* d;         /* Singular values for dlatms */
    int* ipiv;         /* Pivot indices */
    uint64_t seed;
} dsytri_fixture_t;

static uint64_t g_seed = 7300;

static int dsytri_setup(void** state, int n)
{
    dsytri_fixture_t* fix = malloc(sizeof(dsytri_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    int lwork_fac = n * 64;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AFAC = malloc(fix->lda * n * sizeof(double));
    fix->AINV = malloc(fix->lda * n * sizeof(double));
    fix->work_fac = malloc(lwork_fac * sizeof(double));
    fix->work_inv = malloc(n * sizeof(double));
    fix->work_ver = malloc(fix->lda * n * sizeof(double));
    fix->rwork = malloc(n * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->ipiv = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->AFAC);
    assert_non_null(fix->AINV);
    assert_non_null(fix->work_fac);
    assert_non_null(fix->work_inv);
    assert_non_null(fix->work_ver);
    assert_non_null(fix->rwork);
    assert_non_null(fix->d);
    assert_non_null(fix->ipiv);

    *state = fix;
    return 0;
}

static int dsytri_teardown(void** state)
{
    dsytri_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AFAC);
        free(fix->AINV);
        free(fix->work_fac);
        free(fix->work_inv);
        free(fix->work_ver);
        free(fix->rwork);
        free(fix->d);
        free(fix->ipiv);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dsytri_setup(state, 5); }
static int setup_10(void** state) { return dsytri_setup(state, 10); }
static int setup_20(void** state) { return dsytri_setup(state, 20); }

/**
 * Core test logic: generate symmetric matrix, factor, invert, verify.
 *
 * @param fix   Test fixture
 * @param imat  Matrix type (1-6)
 * @param uplo  "U" or "L"
 * @return      Normalized residual
 */
static double run_dsytri_test(dsytri_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;
    int n = fix->n;
    int lda = fix->lda;
    int lwork_fac = n * 64;

    dlatb4("DSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(n, n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, lda, fix->work_fac, &info, rng_state);
    assert_int_equal(info, 0);

    /* Copy A to AFAC for factorization */
    memcpy(fix->AFAC, fix->A, lda * n * sizeof(double));

    /* Factor with dsytrf */
    dsytrf(uplo, n, fix->AFAC, lda, fix->ipiv, fix->work_fac, lwork_fac, &info);
    assert_info_success(info);

    /* Compute inverse with dsytri (overwrites AFAC with inverse triangle) */
    dsytri(uplo, n, fix->AFAC, lda, fix->ipiv, fix->work_inv, &info);
    assert_info_success(info);

    /* Copy the triangular inverse to AINV and expand to full symmetric */
    memcpy(fix->AINV, fix->AFAC, lda * n * sizeof(double));

    if (uplo[0] == 'U') {
        for (int j = 0; j < n; j++)
            for (int i = j + 1; i < n; i++)
                fix->AINV[i + j * lda] = fix->AINV[j + i * lda];
    } else {
        for (int j = 0; j < n; j++)
            for (int i = 0; i < j; i++)
                fix->AINV[i + j * lda] = fix->AINV[j + i * lda];
    }

    /* Verify: ||I - A*AINV|| / (N * ||A|| * ||AINV|| * eps) */
    double rcond, resid;
    dget03(n, fix->A, lda, fix->AINV, lda,
           fix->work_ver, n, fix->rwork, &rcond, &resid);

    return resid;
}

/*
 * Test well-conditioned matrices (types 1-6) with UPLO='U'.
 */
static void test_dsytri_upper(void** state)
{
    dsytri_fixture_t* fix = *state;
    double resid;

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        resid = run_dsytri_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

/*
 * Test well-conditioned matrices (types 1-6) with UPLO='L'.
 */
static void test_dsytri_lower(void** state)
{
    dsytri_fixture_t* fix = *state;
    double resid;

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        resid = run_dsytri_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

#define DSYTRI_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsytri_upper, setup_fn, dsytri_teardown), \
    cmocka_unit_test_setup_teardown(test_dsytri_lower, setup_fn, dsytri_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSYTRI_TESTS(setup_5),
        DSYTRI_TESTS(setup_10),
        DSYTRI_TESTS(setup_20),
    };

    return cmocka_run_group_tests_name("dsytri", tests, NULL, NULL);
}
