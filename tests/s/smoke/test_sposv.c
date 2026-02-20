/**
 * @file test_sposv.c
 * @brief CMocka test suite for sposv (combined Cholesky factor+solve).
 *
 * Tests the combined driver sposv which factors A and solves A*X = B.
 *
 * Verification: spot02 computes ||B - A*X|| / (||A|| * ||X|| * eps)
 *
 * Tests both UPLO='U' and UPLO='L', with NRHS=1,2,5.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

/* Routine under test */
extern void sposv(const char* uplo, const int n, const int nrhs,
                  f32* const restrict A, const int lda,
                  f32* const restrict B, const int ldb, int* info);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda, ldb;
    f32* A;       /* Original matrix */
    f32* AF;      /* Matrix for factorization (overwritten) */
    f32* B;       /* RHS (overwritten with solution) */
    f32* B_orig;  /* Original B for verification */
    f32* X;       /* Known solution */
    f32* d;
    f32* work;
    f32* rwork;
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

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->AF = malloc(fix->lda * n * sizeof(f32));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->X = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(3 * n * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));

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
 * Core test logic: generate matrix, call sposv, verify.
 */
static f32 run_dposv_test(dposv_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info;

    slatb4("SPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate known solution X */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->X[i + j * fix->ldb] = 1.0f + (f32)i / fix->n;
        }
    }

    /* Compute B = A * X */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0f, fix->A, fix->lda,
                fix->X, fix->ldb, 0.0f, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(f32));

    /* Copy A to AF (sposv will overwrite) */
    memcpy(fix->AF, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Call sposv */
    sposv(uplo, fix->n, fix->nrhs, fix->AF, fix->lda, fix->B, fix->ldb, &info);
    assert_info_success(info);

    /* Verify */
    f32 resid;
    spot02(uplo, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, fix->B_orig, fix->ldb, fix->rwork, &resid);
    return resid;
}

static void test_dposv_wellcond_upper(void** state)
{
    dposv_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dposv_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

static void test_dposv_wellcond_lower(void** state)
{
    dposv_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dposv_test(fix, imat, "L");
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
