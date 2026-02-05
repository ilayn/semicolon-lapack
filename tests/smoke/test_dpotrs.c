/**
 * @file test_dpotrs.c
 * @brief CMocka test suite for dpotrs (solve using Cholesky factorization).
 *
 * Tests the solve routine dpotrs which solves A*X = B using the Cholesky
 * factorization computed by dpotrf.
 *
 * Verification: dpot02 computes ||B - A*X|| / (||A|| * ||X|| * eps)
 *
 * Tests both UPLO='U' and UPLO='L', with NRHS=1,2,5.
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dpotrf(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* info);
extern void dpotrs(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   double* const restrict B, const int ldb, int* info);

/* Verification routine */
extern void dpot02(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   const double* const restrict X, const int ldx,
                   double* const restrict B, const int ldb,
                   double* const restrict rwork, double* resid);

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
    int n, nrhs;
    int lda, ldb;
    double* A;       /* Original matrix */
    double* AF;      /* Factored matrix */
    double* B;       /* RHS (overwritten with solution) */
    double* B_orig;  /* Original B for verification */
    double* X;       /* Known solution */
    double* d;       /* Singular values for dlatms */
    double* work;    /* Workspace for dlatms */
    double* rwork;   /* Workspace for dpot02 */
    uint64_t seed;
} dpotrs_fixture_t;

static uint64_t g_seed = 5100;

static int dpotrs_setup(void** state, int n, int nrhs)
{
    dpotrs_fixture_t* fix = malloc(sizeof(dpotrs_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AF = malloc(fix->lda * n * sizeof(double));
    fix->B = malloc(fix->ldb * nrhs * sizeof(double));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(double));
    fix->X = malloc(fix->ldb * nrhs * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(3 * n * sizeof(double));
    fix->rwork = malloc(n * sizeof(double));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->B_orig);
    assert_non_null(fix->X);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);

    *state = fix;
    return 0;
}

static int dpotrs_teardown(void** state)
{
    dpotrs_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->B);
        free(fix->B_orig);
        free(fix->X);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dpotrs_setup(state, 5, 1); }
static int setup_10(void** state) { return dpotrs_setup(state, 10, 1); }
static int setup_20(void** state) { return dpotrs_setup(state, 20, 1); }
static int setup_5_nrhs2(void** state) { return dpotrs_setup(state, 5, 2); }
static int setup_10_nrhs2(void** state) { return dpotrs_setup(state, 10, 2); }
static int setup_20_nrhs2(void** state) { return dpotrs_setup(state, 20, 2); }
static int setup_5_nrhs5(void** state) { return dpotrs_setup(state, 5, 5); }
static int setup_10_nrhs5(void** state) { return dpotrs_setup(state, 10, 5); }
static int setup_20_nrhs5(void** state) { return dpotrs_setup(state, 20, 5); }

/**
 * Core test logic: generate matrix, factorize, solve, verify.
 */
static double run_dpotrs_test(dpotrs_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;

    dlatb4("DPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    dlatms(fix->n, fix->n, &dist, fix->seed, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    /* Generate known solution X */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->X[i + j * fix->ldb] = 1.0 + (double)i / fix->n;
        }
    }

    /* Compute B = A * X (A is symmetric) */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->X, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(double));

    /* Factor A */
    memcpy(fix->AF, fix->A, fix->lda * fix->n * sizeof(double));
    dpotrf(uplo, fix->n, fix->AF, fix->lda, &info);
    assert_info_success(info);

    /* Solve A*X = B */
    dpotrs(uplo, fix->n, fix->nrhs, fix->AF, fix->lda,
           fix->B, fix->ldb, &info);
    assert_info_success(info);

    /* Verify: dpot02 computes ||B_orig - A*X_computed|| / (||A||*||X||*eps) */
    double resid;
    dpot02(uplo, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, fix->B_orig, fix->ldb, fix->rwork, &resid);
    return resid;
}

/*
 * Test well-conditioned matrices (types 1-5) with both UPLO
 */
static void test_dpotrs_wellcond_upper(void** state)
{
    dpotrs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        double resid = run_dpotrs_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

static void test_dpotrs_wellcond_lower(void** state)
{
    dpotrs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        double resid = run_dpotrs_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

#define DPOTRS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dpotrs_wellcond_upper, setup_fn, dpotrs_teardown), \
    cmocka_unit_test_setup_teardown(test_dpotrs_wellcond_lower, setup_fn, dpotrs_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DPOTRS_TESTS(setup_5),
        DPOTRS_TESTS(setup_10),
        DPOTRS_TESTS(setup_20),
        DPOTRS_TESTS(setup_5_nrhs2),
        DPOTRS_TESTS(setup_10_nrhs2),
        DPOTRS_TESTS(setup_20_nrhs2),
        DPOTRS_TESTS(setup_5_nrhs5),
        DPOTRS_TESTS(setup_10_nrhs5),
        DPOTRS_TESTS(setup_20_nrhs5),
    };

    return cmocka_run_group_tests_name("dpotrs", tests, NULL, NULL);
}
