/**
 * @file test_dposv.c
 * @brief CMocka test suite for dposv (combined Cholesky factor+solve).
 *
 * Tests the combined driver dposv which factors A and solves A*X = B.
 *
 * Verification: dpot02 computes ||B - A*X|| / (||A|| * ||X|| * eps)
 *
 * Tests both UPLO='U' and UPLO='L', with NRHS=1,2,5.
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dposv(const char* uplo, const int n, const int nrhs,
                  double* const restrict A, const int lda,
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
    double* AF;      /* Matrix for factorization (overwritten) */
    double* B;       /* RHS (overwritten with solution) */
    double* B_orig;  /* Original B for verification */
    double* X;       /* Known solution */
    double* d;
    double* work;
    double* rwork;
    uint64_t seed;
} dposv_fixture_t;

static uint64_t g_seed = 5200;

static int dposv_setup(void** state, int n, int nrhs)
{
    dposv_fixture_t* fix = malloc(sizeof(dposv_fixture_t));
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

static int dposv_teardown(void** state)
{
    dposv_fixture_t* fix = *state;
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

static int setup_5(void** state) { return dposv_setup(state, 5, 1); }
static int setup_10(void** state) { return dposv_setup(state, 10, 1); }
static int setup_20(void** state) { return dposv_setup(state, 20, 1); }
static int setup_5_nrhs2(void** state) { return dposv_setup(state, 5, 2); }
static int setup_10_nrhs2(void** state) { return dposv_setup(state, 10, 2); }
static int setup_20_nrhs2(void** state) { return dposv_setup(state, 20, 2); }
static int setup_5_nrhs5(void** state) { return dposv_setup(state, 5, 5); }
static int setup_10_nrhs5(void** state) { return dposv_setup(state, 10, 5); }
static int setup_20_nrhs5(void** state) { return dposv_setup(state, 20, 5); }

/**
 * Core test logic: generate matrix, call dposv, verify.
 */
static double run_dposv_test(dposv_fixture_t* fix, int imat, const char* uplo)
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

    /* Compute B = A * X */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->X, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(double));

    /* Copy A to AF (dposv will overwrite) */
    memcpy(fix->AF, fix->A, fix->lda * fix->n * sizeof(double));

    /* Call dposv */
    dposv(uplo, fix->n, fix->nrhs, fix->AF, fix->lda, fix->B, fix->ldb, &info);
    assert_info_success(info);

    /* Verify */
    double resid;
    dpot02(uplo, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, fix->B_orig, fix->ldb, fix->rwork, &resid);
    return resid;
}

static void test_dposv_wellcond_upper(void** state)
{
    dposv_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        double resid = run_dposv_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

static void test_dposv_wellcond_lower(void** state)
{
    dposv_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        double resid = run_dposv_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

#define DPOSV_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dposv_wellcond_upper, setup_fn, dposv_teardown), \
    cmocka_unit_test_setup_teardown(test_dposv_wellcond_lower, setup_fn, dposv_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DPOSV_TESTS(setup_5),
        DPOSV_TESTS(setup_10),
        DPOSV_TESTS(setup_20),
        DPOSV_TESTS(setup_5_nrhs2),
        DPOSV_TESTS(setup_10_nrhs2),
        DPOSV_TESTS(setup_20_nrhs2),
        DPOSV_TESTS(setup_5_nrhs5),
        DPOSV_TESTS(setup_10_nrhs5),
        DPOSV_TESTS(setup_20_nrhs5),
    };

    return cmocka_run_group_tests_name("dposv", tests, NULL, NULL);
}
