/**
 * @file test_dgesv.c
 * @brief CMocka test suite for dgesv (solve general linear system A*X = B).
 *
 * Tests the combined solve routine dgesv which factors A using LU with
 * partial pivoting (dgetrf) and solves the system (dgetrs) in one call.
 *
 * Verification:
 * - dget02: ||B - A*X|| / (||A|| * ||X|| * eps)
 * - dget04: ||X_computed - X_exact|| scaled by condition
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dgesv(const int n, const int nrhs, f64 * const restrict A,
                  const int lda, int * const restrict ipiv,
                  f64 * const restrict B, const int ldb, int *info);

/* Utilities */
extern f64 dlamch(const char *cmach);
extern f64 dlange(const char *norm, const int m, const int n,
                     const f64 * const restrict A, const int lda,
                     f64 * const restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda, ldb;
    f64 *A;       /* Original matrix */
    f64 *A_copy;  /* Copy for dgesv (gets overwritten) */
    f64 *B;       /* Right-hand side (gets overwritten with solution) */
    f64 *B_orig;  /* Original B for verification */
    f64 *XACT;    /* Known exact solution */
    f64 *d;       /* Singular values for dlatms */
    f64 *work;    /* Workspace */
    f64 *rwork;   /* Workspace for dget02 */
    int *ipiv;       /* Pivot indices */
    uint64_t seed;
} dgesv_fixture_t;

static uint64_t g_seed = 1729;

static int dgesv_setup(void **state, int n, int nrhs)
{
    dgesv_fixture_t *fix = malloc(sizeof(dgesv_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->A_copy = malloc(fix->lda * n * sizeof(f64));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->XACT = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(3 * n * sizeof(f64));
    fix->rwork = malloc(n * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->A_copy);
    assert_non_null(fix->B);
    assert_non_null(fix->B_orig);
    assert_non_null(fix->XACT);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->ipiv);

    *state = fix;
    return 0;
}

static int dgesv_teardown(void **state)
{
    dgesv_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->A_copy);
        free(fix->B);
        free(fix->B_orig);
        free(fix->XACT);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->ipiv);
        free(fix);
    }
    return 0;
}

/* Size-specific setups */
static int setup_2(void **state) { return dgesv_setup(state, 2, 1); }
static int setup_3(void **state) { return dgesv_setup(state, 3, 1); }
static int setup_5(void **state) { return dgesv_setup(state, 5, 1); }
static int setup_10(void **state) { return dgesv_setup(state, 10, 1); }
static int setup_20(void **state) { return dgesv_setup(state, 20, 1); }
static int setup_5_nrhs2(void **state) { return dgesv_setup(state, 5, 2); }
static int setup_10_nrhs2(void **state) { return dgesv_setup(state, 10, 2); }
static int setup_20_nrhs2(void **state) { return dgesv_setup(state, 20, 2); }
static int setup_5_nrhs5(void **state) { return dgesv_setup(state, 5, 5); }
static int setup_10_nrhs5(void **state) { return dgesv_setup(state, 10, 5); }
static int setup_20_nrhs5(void **state) { return dgesv_setup(state, 20, 5); }

/**
 * Core test logic for well-conditioned matrices (type 4).
 */
static void test_dgesv_wellcond(void **state)
{
    dgesv_fixture_t *fix = *state;
    int info;

    fix->seed = g_seed++;

    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;

    dlatb4("DGE", 4, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix A */
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);
    memcpy(fix->A_copy, fix->A, fix->lda * fix->n * sizeof(f64));

    /* Generate known exact solution */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0 + (f64)i / fix->n + (f64)j / fix->nrhs;
        }
    }

    /* Compute B = A * XACT */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                fix->n, fix->nrhs, fix->n, 1.0, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(f64));

    /* Solve */
    dgesv(fix->n, fix->nrhs, fix->A_copy, fix->lda, fix->ipiv,
          fix->B, fix->ldb, &info);
    assert_info_success(info);

    /* Test 1: Solution residual */
    f64 *B_copy = malloc(fix->ldb * fix->nrhs * sizeof(f64));
    assert_non_null(B_copy);
    memcpy(B_copy, fix->B_orig, fix->ldb * fix->nrhs * sizeof(f64));

    f64 resid_02;
    dget02("N", fix->n, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, B_copy, fix->ldb, fix->rwork, &resid_02);
    assert_residual_ok(resid_02);
    free(B_copy);

    /* Test 2: Solution accuracy */
    f64 rcond = 1.0 / (cndnum > 0.0 ? cndnum : 1.0);
    f64 resid_04;
    dget04(fix->n, fix->nrhs, fix->B, fix->ldb, fix->XACT, fix->ldb,
           rcond, &resid_04);
    assert_residual_ok(resid_04);
}

/**
 * Test ill-conditioned matrices (type 8).
 * Only run for n >= 5.
 */
static void test_dgesv_illcond(void **state)
{
    dgesv_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip();
    }

    int info;
    fix->seed = g_seed++;

    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;

    dlatb4("DGE", 8, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);
    memcpy(fix->A_copy, fix->A, fix->lda * fix->n * sizeof(f64));

    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0 + (f64)i / fix->n + (f64)j / fix->nrhs;
        }
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                fix->n, fix->nrhs, fix->n, 1.0, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(f64));

    dgesv(fix->n, fix->nrhs, fix->A_copy, fix->lda, fix->ipiv,
          fix->B, fix->ldb, &info);
    assert_info_success(info);

    f64 *B_copy = malloc(fix->ldb * fix->nrhs * sizeof(f64));
    assert_non_null(B_copy);
    memcpy(B_copy, fix->B_orig, fix->ldb * fix->nrhs * sizeof(f64));

    f64 resid_02;
    dget02("N", fix->n, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, B_copy, fix->ldb, fix->rwork, &resid_02);
    assert_residual_ok(resid_02);
    free(B_copy);
}

/**
 * Sanity check: simple known system.
 */
static void test_dgesv_simple(void **state)
{
    (void)state;

    f64 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    f64 b[3] = {4, 10, 24};
    int ipiv[3];
    int info;

    dgesv(3, 1, A, 3, ipiv, b, 3, &info);

    assert_info_success(info);
    assert_true(fabs(b[0] - 1.0) < 1e-10);
    assert_true(fabs(b[1] - 1.0) < 1e-10);
    assert_true(fabs(b[2] - 1.0) < 1e-10);
}

/**
 * Test singular matrix detection.
 */
static void test_dgesv_singular(void **state)
{
    (void)state;

    f64 A[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    f64 b[3] = {1, 2, 3};
    int ipiv[3];
    int info;

    dgesv(3, 1, A, 3, ipiv, b, 3, &info);

    assert_info_singular(info);
}

#define DGESV_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgesv_wellcond, setup_fn, dgesv_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesv_illcond, setup_fn, dgesv_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Sanity checks */
        cmocka_unit_test(test_dgesv_simple),
        cmocka_unit_test(test_dgesv_singular),

        /* Single RHS */
        DGESV_TESTS(setup_2),
        DGESV_TESTS(setup_3),
        DGESV_TESTS(setup_5),
        DGESV_TESTS(setup_10),
        DGESV_TESTS(setup_20),

        /* Multiple RHS (nrhs=2) */
        DGESV_TESTS(setup_5_nrhs2),
        DGESV_TESTS(setup_10_nrhs2),
        DGESV_TESTS(setup_20_nrhs2),

        /* Multiple RHS (nrhs=5) */
        DGESV_TESTS(setup_5_nrhs5),
        DGESV_TESTS(setup_10_nrhs5),
        DGESV_TESTS(setup_20_nrhs5),
    };

    return cmocka_run_group_tests_name("dgesv", tests, NULL, NULL);
}
