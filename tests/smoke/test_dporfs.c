/**
 * @file test_dporfs.c
 * @brief CMocka test suite for dporfs (iterative refinement for Cholesky).
 *
 * Tests the iterative refinement routine dporfs which improves the computed
 * solution and provides error bounds for a symmetric positive definite system.
 *
 * Verification: dpot05 tests forward error (FERR) and backward error (BERR).
 *
 * Tests both UPLO='U' and UPLO='L', with NRHS=1,2,5.
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
extern void dpotrs(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   double* const restrict B, const int ldb, int* info);
extern void dporfs(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   const double* const restrict AF, const int ldaf,
                   const double* const restrict B, const int ldb,
                   double* const restrict X, const int ldx,
                   double* const restrict ferr, double* const restrict berr,
                   double* const restrict work, int* const restrict iwork,
                   int* info);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda, ldb;
    double* A;       /* Original matrix */
    double* AF;      /* Factored matrix */
    double* B;       /* RHS */
    double* X;       /* Computed solution */
    double* XACT;    /* Exact solution */
    double* ferr;
    double* berr;
    double* d;
    double* work;
    int* iwork;
    double reslts[2];
    uint64_t seed;
} dporfs_fixture_t;

static uint64_t g_seed = 5600;

static int dporfs_setup(void** state, int n, int nrhs)
{
    dporfs_fixture_t* fix = malloc(sizeof(dporfs_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AF = malloc(fix->lda * n * sizeof(double));
    fix->B = malloc(fix->ldb * nrhs * sizeof(double));
    fix->X = malloc(fix->ldb * nrhs * sizeof(double));
    fix->XACT = malloc(fix->ldb * nrhs * sizeof(double));
    fix->ferr = malloc(nrhs * sizeof(double));
    fix->berr = malloc(nrhs * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(3 * n * sizeof(double));
    fix->iwork = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

static int dporfs_teardown(void** state)
{
    dporfs_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->B);
        free(fix->X);
        free(fix->XACT);
        free(fix->ferr);
        free(fix->berr);
        free(fix->d);
        free(fix->work);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dporfs_setup(state, 5, 1); }
static int setup_10(void** state) { return dporfs_setup(state, 10, 1); }
static int setup_20(void** state) { return dporfs_setup(state, 20, 1); }
static int setup_5_nrhs2(void** state) { return dporfs_setup(state, 5, 2); }
static int setup_10_nrhs2(void** state) { return dporfs_setup(state, 10, 2); }
static int setup_20_nrhs2(void** state) { return dporfs_setup(state, 20, 2); }
static int setup_5_nrhs5(void** state) { return dporfs_setup(state, 5, 5); }
static int setup_10_nrhs5(void** state) { return dporfs_setup(state, 10, 5); }
static int setup_20_nrhs5(void** state) { return dporfs_setup(state, 20, 5); }

/**
 * Core test logic: generate matrix, solve, refine, verify error bounds.
 */
static void run_dporfs_test(dporfs_fixture_t* fix, int imat, const char* uplo)
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

    /* Generate exact solution */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0 + (double)i / fix->n;
        }
    }

    /* Compute B = A * XACT */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0, fix->B, fix->ldb);

    /* Factor A */
    memcpy(fix->AF, fix->A, fix->lda * fix->n * sizeof(double));
    dpotrf(uplo, fix->n, fix->AF, fix->lda, &info);
    assert_info_success(info);

    /* Solve A*X = B */
    memcpy(fix->X, fix->B, fix->ldb * fix->nrhs * sizeof(double));
    dpotrs(uplo, fix->n, fix->nrhs, fix->AF, fix->lda,
           fix->X, fix->ldb, &info);
    assert_info_success(info);

    /* Iterative refinement */
    dporfs(uplo, fix->n, fix->nrhs, fix->A, fix->lda, fix->AF, fix->lda,
           fix->B, fix->ldb, fix->X, fix->ldb,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Verify error bounds */
    dpot05(uplo, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, fix->X, fix->ldb, fix->XACT, fix->ldb,
           fix->ferr, fix->berr, fix->reslts);
    assert_residual_ok(fix->reslts[0]);
    assert_residual_ok(fix->reslts[1]);
}

static void test_dporfs_wellcond_upper(void** state)
{
    dporfs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dporfs_test(fix, imat, "U");
    }
}

static void test_dporfs_wellcond_lower(void** state)
{
    dporfs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dporfs_test(fix, imat, "L");
    }
}

#define DPORFS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dporfs_wellcond_upper, setup_fn, dporfs_teardown), \
    cmocka_unit_test_setup_teardown(test_dporfs_wellcond_lower, setup_fn, dporfs_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DPORFS_TESTS(setup_5),
        DPORFS_TESTS(setup_10),
        DPORFS_TESTS(setup_20),
        DPORFS_TESTS(setup_5_nrhs2),
        DPORFS_TESTS(setup_10_nrhs2),
        DPORFS_TESTS(setup_20_nrhs2),
        DPORFS_TESTS(setup_5_nrhs5),
        DPORFS_TESTS(setup_10_nrhs5),
        DPORFS_TESTS(setup_20_nrhs5),
    };

    return cmocka_run_group_tests_name("dporfs", tests, NULL, NULL);
}
