/**
 * @file test_sgesv.c
 * @brief CMocka test suite for sgesv (solve general linear system A*X = B).
 *
 * Tests the combined solve routine sgesv which factors A using LU with
 * partial pivoting (sgetrf) and solves the system (sgetrs) in one call.
 *
 * Verification:
 * - sget02: ||B - A*X|| / (||A|| * ||X|| * eps)
 * - sget04: ||X_computed - X_exact|| scaled by condition
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include "semicolon_cblas.h"

/* Routine under test */
/* Utilities */
/*
 * Test fixture
 */
typedef struct {
    INT n, nrhs;
    INT lda, ldb;
    f32 *A;       /* Original matrix */
    f32 *A_copy;  /* Copy for sgesv (gets overwritten) */
    f32 *B;       /* Right-hand side (gets overwritten with solution) */
    f32 *B_orig;  /* Original B for verification */
    f32 *XACT;    /* Known exact solution */
    f32 *d;       /* Singular values for slatms */
    f32 *work;    /* Workspace */
    f32 *rwork;   /* Workspace for sget02 */
    INT* ipiv;       /* Pivot indices */
    uint64_t seed;
} dgesv_fixture_t;

static uint64_t g_seed = 1729;

static int dgesv_setup(void **state, INT n, INT nrhs)
{
    dgesv_fixture_t *fix = malloc(sizeof(dgesv_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->A_copy = malloc(fix->lda * n * sizeof(f32));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->XACT = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(3 * n * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));
    fix->ipiv = malloc(n * sizeof(INT));

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
    INT info;

    fix->seed = g_seed++;

    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;

    slatb4("SGE", 4, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix A */
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);
    memcpy(fix->A_copy, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Generate known exact solution */
    for (INT j = 0; j < fix->nrhs; j++) {
        for (INT i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0f + (f32)i / fix->n + (f32)j / fix->nrhs;
        }
    }

    /* Compute B = A * XACT */
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                fix->n, fix->nrhs, fix->n, 1.0f, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0f, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(f32));

    /* Solve */
    sgesv(fix->n, fix->nrhs, fix->A_copy, fix->lda, fix->ipiv,
          fix->B, fix->ldb, &info);
    assert_info_success(info);

    /* Test 1: Solution residual */
    f32 *B_copy = malloc(fix->ldb * fix->nrhs * sizeof(f32));
    assert_non_null(B_copy);
    memcpy(B_copy, fix->B_orig, fix->ldb * fix->nrhs * sizeof(f32));

    f32 resid_02;
    sget02("N", fix->n, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, B_copy, fix->ldb, fix->rwork, &resid_02);
    assert_residual_ok(resid_02);
    free(B_copy);

    /* Test 2: Solution accuracy */
    f32 rcond = 1.0f / (cndnum > 0.0f ? cndnum : 1.0f);
    f32 resid_04;
    sget04(fix->n, fix->nrhs, fix->B, fix->ldb, fix->XACT, fix->ldb,
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

    INT info;
    fix->seed = g_seed++;

    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;

    slatb4("SGE", 8, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);
    memcpy(fix->A_copy, fix->A, fix->lda * fix->n * sizeof(f32));

    for (INT j = 0; j < fix->nrhs; j++) {
        for (INT i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0f + (f32)i / fix->n + (f32)j / fix->nrhs;
        }
    }

    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                fix->n, fix->nrhs, fix->n, 1.0f, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0f, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(f32));

    sgesv(fix->n, fix->nrhs, fix->A_copy, fix->lda, fix->ipiv,
          fix->B, fix->ldb, &info);
    assert_info_success(info);

    f32 *B_copy = malloc(fix->ldb * fix->nrhs * sizeof(f32));
    assert_non_null(B_copy);
    memcpy(B_copy, fix->B_orig, fix->ldb * fix->nrhs * sizeof(f32));

    f32 resid_02;
    sget02("N", fix->n, fix->n, fix->nrhs, fix->A, fix->lda,
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

    f32 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    f32 b[3] = {4, 10, 24};
    INT ipiv[3];
    INT info;

    sgesv(3, 1, A, 3, ipiv, b, 3, &info);

    assert_info_success(info);
    assert_true(fabsf(b[0] - 1.0f) < 1e-5f);
    assert_true(fabsf(b[1] - 1.0f) < 1e-5f);
    assert_true(fabsf(b[2] - 1.0f) < 1e-5f);
}

/**
 * Test singular matrix detection.
 */
static void test_dgesv_singular(void **state)
{
    (void)state;

    f32 A[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    f32 b[3] = {1, 2, 3};
    INT ipiv[3];
    INT info;

    sgesv(3, 1, A, 3, ipiv, b, 3, &info);

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
