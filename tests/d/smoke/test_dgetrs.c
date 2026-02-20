/**
 * @file test_dgetrs.c
 * @brief CMocka test suite for dgetrs (solve using LU factorization).
 *
 * Tests the triangular solve routine dgetrs which solves A*X = B or A'*X = B
 * using the LU factorization computed by dgetrf.
 *
 * Verification: dget02 computes ||B - op(A)*X|| / (||A|| * ||X|| * eps)
 *
 * Tests all three transpose options: 'N', 'T', 'C'
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dgetrf(const int m, const int n, f64 * const restrict A,
                   const int lda, int * const restrict ipiv, int *info);
extern void dgetrs(const char *trans, const int n, const int nrhs,
                   const f64 * const restrict A, const int lda,
                   const int * const restrict ipiv, f64 * const restrict B,
                   const int ldb, int *info);

/* Utilities */
extern f64 dlamch(const char *cmach);

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int n, nrhs;
    int lda, ldb;
    f64 *A;       /* Original matrix */
    f64 *A_fact;  /* Factored matrix */
    f64 *B;       /* Right-hand side */
    f64 *B_orig;  /* Original B for verification */
    f64 *X;       /* Known solution */
    f64 *d;       /* Singular values for dlatms */
    f64 *work;    /* Workspace for dlatms */
    f64 *rwork;   /* Workspace for dget02 */
    int *ipiv;       /* Pivot indices */
    uint64_t seed;   /* RNG seed */
} dgetrs_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 2024;

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dgetrs_setup(void **state, int n, int nrhs)
{
    dgetrs_fixture_t *fix = malloc(sizeof(dgetrs_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->A_fact = malloc(fix->lda * n * sizeof(f64));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->X = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(3 * n * sizeof(f64));
    fix->rwork = malloc(n * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->A_fact);
    assert_non_null(fix->B);
    assert_non_null(fix->B_orig);
    assert_non_null(fix->X);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->ipiv);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgetrs_teardown(void **state)
{
    dgetrs_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->A_fact);
        free(fix->B);
        free(fix->B_orig);
        free(fix->X);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->ipiv);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions (nrhs=1) */
static int setup_2(void **state) { return dgetrs_setup(state, 2, 1); }
static int setup_3(void **state) { return dgetrs_setup(state, 3, 1); }
static int setup_5(void **state) { return dgetrs_setup(state, 5, 1); }
static int setup_10(void **state) { return dgetrs_setup(state, 10, 1); }
static int setup_20(void **state) { return dgetrs_setup(state, 20, 1); }

/* Multi-RHS setups */
static int setup_5_nrhs2(void **state) { return dgetrs_setup(state, 5, 2); }
static int setup_10_nrhs2(void **state) { return dgetrs_setup(state, 10, 2); }
static int setup_20_nrhs2(void **state) { return dgetrs_setup(state, 20, 2); }
static int setup_5_nrhs5(void **state) { return dgetrs_setup(state, 5, 5); }
static int setup_10_nrhs5(void **state) { return dgetrs_setup(state, 10, 5); }
static int setup_20_nrhs5(void **state) { return dgetrs_setup(state, 20, 5); }

/**
 * Core test logic: generate matrix, factorize, solve, verify.
 */
static f64 run_dgetrs_test(dgetrs_fixture_t *fix, int imat, const char* trans)
{
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info;

    /* Get matrix parameters */
    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate test matrix A */
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate known solution X */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->X[i + j * fix->ldb] = 1.0 + (f64)i / fix->n;
        }
    }

    /* Compute B = op(A) * X */
    CBLAS_TRANSPOSE cblas_trans = (trans[0] == 'N') ? CblasNoTrans : CblasTrans;
    cblas_dgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                fix->n, fix->nrhs, fix->n, 1.0, fix->A, fix->lda,
                fix->X, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * fix->nrhs * sizeof(f64));

    /* Factor A */
    memcpy(fix->A_fact, fix->A, fix->lda * fix->n * sizeof(f64));
    dgetrf(fix->n, fix->n, fix->A_fact, fix->lda, fix->ipiv, &info);
    assert_info_success(info);

    /* Solve op(A) * X = B */
    dgetrs(trans, fix->n, fix->nrhs, fix->A_fact, fix->lda, fix->ipiv,
           fix->B, fix->ldb, &info);
    assert_info_success(info);

    /* Verify: dget02 computes ||B_orig - op(A)*X_computed|| / (||A||*||X||*eps) */
    f64 resid;
    dget02(trans, fix->n, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->B, fix->ldb, fix->B_orig, fix->ldb, fix->rwork, &resid);

    return resid;
}

/*
 * Test well-conditioned matrices (type 4) with no transpose.
 */
static void test_dgetrs_notrans(void **state)
{
    dgetrs_fixture_t *fix = *state;
    fix->seed = g_seed++;
    f64 resid = run_dgetrs_test(fix, 4, "N");
    assert_residual_ok(resid);
}

/*
 * Test well-conditioned matrices (type 4) with transpose.
 */
static void test_dgetrs_trans(void **state)
{
    dgetrs_fixture_t *fix = *state;
    fix->seed = g_seed++;
    f64 resid = run_dgetrs_test(fix, 4, "T");
    assert_residual_ok(resid);
}

/*
 * Test well-conditioned matrices (type 4) with conjugate transpose.
 * For real matrices, 'C' is same as 'T'.
 */
static void test_dgetrs_conjtrans(void **state)
{
    dgetrs_fixture_t *fix = *state;
    fix->seed = g_seed++;
    f64 resid = run_dgetrs_test(fix, 4, "C");
    assert_residual_ok(resid);
}

/*
 * Test ill-conditioned matrices (type 8).
 * Only run for n >= 5.
 */
static void test_dgetrs_illcond(void **state)
{
    dgetrs_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip();
    }

    fix->seed = g_seed++;
    f64 resid = run_dgetrs_test(fix, 8, "N");
    assert_residual_ok(resid);

    fix->seed = g_seed++;
    resid = run_dgetrs_test(fix, 8, "T");
    assert_residual_ok(resid);
}

#define DGETRS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgetrs_notrans, setup_fn, dgetrs_teardown), \
    cmocka_unit_test_setup_teardown(test_dgetrs_trans, setup_fn, dgetrs_teardown), \
    cmocka_unit_test_setup_teardown(test_dgetrs_conjtrans, setup_fn, dgetrs_teardown), \
    cmocka_unit_test_setup_teardown(test_dgetrs_illcond, setup_fn, dgetrs_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Single RHS */
        DGETRS_TESTS(setup_2),
        DGETRS_TESTS(setup_3),
        DGETRS_TESTS(setup_5),
        DGETRS_TESTS(setup_10),
        DGETRS_TESTS(setup_20),

        /* Multiple RHS (nrhs=2) */
        DGETRS_TESTS(setup_5_nrhs2),
        DGETRS_TESTS(setup_10_nrhs2),
        DGETRS_TESTS(setup_20_nrhs2),

        /* Multiple RHS (nrhs=5) */
        DGETRS_TESTS(setup_5_nrhs5),
        DGETRS_TESTS(setup_10_nrhs5),
        DGETRS_TESTS(setup_20_nrhs5),
    };

    return cmocka_run_group_tests_name("dgetrs", tests, NULL, NULL);
}
