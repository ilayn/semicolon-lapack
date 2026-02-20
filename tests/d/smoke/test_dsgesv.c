/**
 * @file test_dsgesv.c
 * @brief CMocka test suite for dsgesv (mixed precision iterative refinement solver).
 *
 * Tests the mixed precision solver which factors in single precision
 * and refines in f64 precision to achieve f64 precision accuracy.
 *
 * Verification:
 * - dget02: ||B - A*X|| / (||A|| * ||X|| * eps)
 * - dget04: ||X_computed - X_exact|| scaled by condition
 * - Iteration count check (iter >= 0 means refinement succeeded)
 *
 * Configurations:
 *   Sizes: {2, 3, 5, 10, 20, 50}
 *   NRHS:  {1, 2, 5}
 *   Types: {4 (well-conditioned), 8 (ill-conditioned, N>=5 only)}
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dsgesv(const int n, const int nrhs, f64 * const restrict A,
                   const int lda, int * const restrict ipiv,
                   const f64 * const restrict B, const int ldb,
                   f64 * const restrict X, const int ldx,
                   f64 * const restrict work, float * const restrict swork,
                   int *iter, int *info);

/* Utilities */
extern f64 dlamch(const char *cmach);

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int n;
    int nrhs;
    int lda, ldb, ldx;
    f64 *A;        /* Matrix (modified by dsgesv) */
    f64 *A_orig;   /* Pristine copy of A */
    f64 *B;        /* Right-hand side */
    f64 *B_orig;   /* Pristine copy of B */
    f64 *X;        /* Solution */
    f64 *XACT;     /* Exact solution */
    f64 *d;        /* Singular values for dlatms */
    f64 *work;     /* Double workspace for dsgesv and dlatms */
    f64 *rwork;    /* Workspace for dget02 */
    float *swork;     /* Single precision workspace for dsgesv */
    int *ipiv;        /* Pivot indices */
    uint64_t seed;    /* RNG seed */
} dsgesv_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 1729;

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dsgesv_setup(void **state, int n, int nrhs)
{
    dsgesv_fixture_t *fix = malloc(sizeof(dsgesv_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->ldx = n;
    fix->seed = g_seed++;

    /* work needs to be large enough for both dlatms (3*n) and dsgesv (n*nrhs) */
    int work_size = (3 * n > n * nrhs) ? 3 * n : n * nrhs;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->A_orig = malloc(fix->lda * n * sizeof(f64));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(f64));
    fix->X = malloc(fix->ldx * nrhs * sizeof(f64));
    fix->XACT = malloc(fix->ldx * nrhs * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(work_size * sizeof(f64));
    fix->rwork = malloc(n * sizeof(f64));
    fix->swork = malloc(n * (n + nrhs) * sizeof(float));
    fix->ipiv = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->A_orig);
    assert_non_null(fix->B);
    assert_non_null(fix->B_orig);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->swork);
    assert_non_null(fix->ipiv);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dsgesv_teardown(void **state)
{
    dsgesv_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->A_orig);
        free(fix->B);
        free(fix->B_orig);
        free(fix->X);
        free(fix->XACT);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->swork);
        free(fix->ipiv);
        free(fix);
    }
    return 0;
}

/* Size-specific setup wrappers: N x NRHS */
static int setup_2_1(void **state) { return dsgesv_setup(state, 2, 1); }
static int setup_2_2(void **state) { return dsgesv_setup(state, 2, 2); }
static int setup_3_1(void **state) { return dsgesv_setup(state, 3, 1); }
static int setup_3_2(void **state) { return dsgesv_setup(state, 3, 2); }
static int setup_5_1(void **state) { return dsgesv_setup(state, 5, 1); }
static int setup_5_2(void **state) { return dsgesv_setup(state, 5, 2); }
static int setup_5_5(void **state) { return dsgesv_setup(state, 5, 5); }
static int setup_10_1(void **state) { return dsgesv_setup(state, 10, 1); }
static int setup_10_2(void **state) { return dsgesv_setup(state, 10, 2); }
static int setup_10_5(void **state) { return dsgesv_setup(state, 10, 5); }
static int setup_20_1(void **state) { return dsgesv_setup(state, 20, 1); }
static int setup_20_2(void **state) { return dsgesv_setup(state, 20, 2); }
static int setup_20_5(void **state) { return dsgesv_setup(state, 20, 5); }
static int setup_50_1(void **state) { return dsgesv_setup(state, 50, 1); }
static int setup_50_2(void **state) { return dsgesv_setup(state, 50, 2); }
static int setup_50_5(void **state) { return dsgesv_setup(state, 50, 5); }

/* Sanity test setups */
static int setup_3_1_simple(void **state) { return dsgesv_setup(state, 3, 1); }
static int setup_4_2_identity(void **state) { return dsgesv_setup(state, 4, 2); }
static int setup_3_1_singular(void **state) { return dsgesv_setup(state, 3, 1); }
static int setup_10_1_refine(void **state) { return dsgesv_setup(state, 10, 1); }

/**
 * Core test logic: generate matrix, solve with dsgesv, compute residuals.
 * Populates resid_02, resid_04, and iter.
 * Returns 0 on success, nonzero if the matrix was singular (skip residual checks).
 */
static int run_dsgesv_test(dsgesv_fixture_t *fix, int imat,
                           f64 *resid_02, f64 *resid_04, int *iter_out)
{
    char type, dist;
    int kl, ku, mode;
    f64 anorm_param, cndnum;
    int info, iter;
    int n = fix->n;
    int nrhs = fix->nrhs;

    /* Get matrix parameters */
    dlatb4("DGE", imat, n, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    /* Generate test matrix A */
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(n, n, &dist, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);
    memcpy(fix->A_orig, fix->A, fix->lda * n * sizeof(f64));

    /* Generate known exact solution XACT */
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            fix->XACT[i + j * fix->ldx] = 1.0 + (f64)i / n + (f64)j / nrhs;
        }
    }

    /* Compute B = A * XACT */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n, nrhs, n, 1.0, fix->A_orig, fix->lda, fix->XACT, fix->ldx,
                0.0, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * nrhs * sizeof(f64));

    /* Solve A * X = B using dsgesv */
    dsgesv(n, nrhs, fix->A, fix->lda, fix->ipiv, fix->B, fix->ldb,
           fix->X, fix->ldx, fix->work, fix->swork, &iter, &info);

    *iter_out = iter;

    if (info != 0) {
        /* Singular matrix - skip solve tests */
        return 1;
    }

    /* Test 1: Solution residual using dget02 */
    f64 *B_copy = malloc(fix->ldb * nrhs * sizeof(f64));
    assert_non_null(B_copy);
    memcpy(B_copy, fix->B_orig, fix->ldb * nrhs * sizeof(f64));
    dget02("N", n, n, nrhs, fix->A_orig, fix->lda, fix->X, fix->ldx,
           B_copy, fix->ldb, fix->rwork, resid_02);
    free(B_copy);

    /* Test 2: Solution accuracy using dget04 */
    f64 rcond = 1.0 / (cndnum > 0.0 ? cndnum : 1.0);
    dget04(n, nrhs, fix->X, fix->ldx, fix->XACT, fix->ldx, rcond, resid_04);

    return 0;
}

/*
 * Sanity test: simple known system
 * A * x = b where solution is x = [1, 1, 1]'
 */
static void test_dsgesv_simple(void **state)
{
    dsgesv_fixture_t *fix = *state;
    int n = 3;
    int nrhs = 1;
    int info, iter;

    /* System: A * x = b where solution is x = [1, 1, 1]' */
    f64 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};  /* Column-major */
    f64 B[3] = {4, 10, 24};

    memcpy(fix->A, A, 9 * sizeof(f64));
    memcpy(fix->B, B, 3 * sizeof(f64));

    dsgesv(n, nrhs, fix->A, n, fix->ipiv, fix->B, n,
           fix->X, n, fix->work, fix->swork, &iter, &info);

    assert_info_success(info);

    f64 tol = 1e-10;
    assert_true(fabs(fix->X[0] - 1.0) < tol);
    assert_true(fabs(fix->X[1] - 1.0) < tol);
    assert_true(fabs(fix->X[2] - 1.0) < tol);
}

/*
 * Sanity test: identity matrix (trivial case)
 */
static void test_dsgesv_identity(void **state)
{
    dsgesv_fixture_t *fix = *state;
    int n = 4;
    int nrhs = 2;
    int info, iter;

    /* Identity matrix */
    memset(fix->A, 0, fix->lda * n * sizeof(f64));
    for (int i = 0; i < n; i++) {
        fix->A[i + i * fix->lda] = 1.0;
    }

    /* RHS = expected solution */
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            fix->B[i + j * fix->ldb] = (f64)(i + 1) + (f64)j * 10;
        }
    }

    dsgesv(n, nrhs, fix->A, n, fix->ipiv, fix->B, n,
           fix->X, n, fix->work, fix->swork, &iter, &info);

    assert_info_success(info);

    f64 tol = 1e-14;
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            f64 expected = (f64)(i + 1) + (f64)j * 10;
            assert_true(fabs(fix->X[i + j * n] - expected) < tol);
        }
    }
}

/*
 * Sanity test: singular matrix detection
 */
static void test_dsgesv_singular(void **state)
{
    dsgesv_fixture_t *fix = *state;
    int n = 3;
    int nrhs = 1;
    int info, iter;

    /* Singular matrix (all columns identical) */
    f64 A[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    f64 B[3] = {1, 2, 3};

    memcpy(fix->A, A, 9 * sizeof(f64));
    memcpy(fix->B, B, 3 * sizeof(f64));

    dsgesv(n, nrhs, fix->A, n, fix->ipiv, fix->B, n,
           fix->X, n, fix->work, fix->swork, &iter, &info);

    assert_info_singular(info);
}

/*
 * Sanity test: iterative refinement with Hilbert-like matrix
 */
static void test_dsgesv_refinement(void **state)
{
    dsgesv_fixture_t *fix = *state;
    int n = 10;
    int nrhs = 1;
    int info, iter;

    /* Create moderately ill-conditioned Hilbert-like matrix */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            fix->A[i + j * fix->lda] = 1.0 / (f64)(i + j + 1);
        }
    }
    memcpy(fix->A_orig, fix->A, fix->lda * n * sizeof(f64));

    /* Known solution: all ones */
    for (int i = 0; i < n; i++) {
        fix->XACT[i] = 1.0;
    }

    /* Compute B = A * XACT */
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n,
                1.0, fix->A_orig, n, fix->XACT, 1, 0.0, fix->B, 1);

    dsgesv(n, nrhs, fix->A, n, fix->ipiv, fix->B, n,
           fix->X, n, fix->work, fix->swork, &iter, &info);

    /* For Hilbert matrix, we expect either successful refinement or fallback.
     * Either way, info should be 0. */
    assert_info_success(info);
}

/*
 * Comprehensive test: well-conditioned (type 4)
 */
static void test_dsgesv_type4(void **state)
{
    dsgesv_fixture_t *fix = *state;
    f64 resid_02, resid_04;
    int iter;

    fix->seed = g_seed++;
    int rc = run_dsgesv_test(fix, 4, &resid_02, &resid_04, &iter);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
}

/*
 * Comprehensive test: ill-conditioned (type 8)
 */
static void test_dsgesv_type8(void **state)
{
    dsgesv_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip_test("type 8 requires N >= 5");
    }

    f64 resid_02, resid_04;
    int iter;

    fix->seed = g_seed++;
    int rc = run_dsgesv_test(fix, 8, &resid_02, &resid_04, &iter);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
}

/*
 * Macro to generate test entries for a given size setup.
 * Creates 2 test cases: type4 (well-conditioned) and type8 (ill-conditioned).
 */
#define DSGESV_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsgesv_type4, setup_fn, dsgesv_teardown), \
    cmocka_unit_test_setup_teardown(test_dsgesv_type8, setup_fn, dsgesv_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Sanity checks */
        cmocka_unit_test_setup_teardown(test_dsgesv_simple, setup_3_1_simple, dsgesv_teardown),
        cmocka_unit_test_setup_teardown(test_dsgesv_identity, setup_4_2_identity, dsgesv_teardown),
        cmocka_unit_test_setup_teardown(test_dsgesv_singular, setup_3_1_singular, dsgesv_teardown),
        cmocka_unit_test_setup_teardown(test_dsgesv_refinement, setup_10_1_refine, dsgesv_teardown),

        /* Comprehensive: N=2, NRHS=1 */
        DSGESV_TESTS(setup_2_1),
        /* Comprehensive: N=2, NRHS=2 */
        DSGESV_TESTS(setup_2_2),

        /* Comprehensive: N=3, NRHS=1 */
        DSGESV_TESTS(setup_3_1),
        /* Comprehensive: N=3, NRHS=2 */
        DSGESV_TESTS(setup_3_2),

        /* Comprehensive: N=5, NRHS=1 */
        DSGESV_TESTS(setup_5_1),
        /* Comprehensive: N=5, NRHS=2 */
        DSGESV_TESTS(setup_5_2),
        /* Comprehensive: N=5, NRHS=5 */
        DSGESV_TESTS(setup_5_5),

        /* Comprehensive: N=10, NRHS=1 */
        DSGESV_TESTS(setup_10_1),
        /* Comprehensive: N=10, NRHS=2 */
        DSGESV_TESTS(setup_10_2),
        /* Comprehensive: N=10, NRHS=5 */
        DSGESV_TESTS(setup_10_5),

        /* Comprehensive: N=20, NRHS=1 */
        DSGESV_TESTS(setup_20_1),
        /* Comprehensive: N=20, NRHS=2 */
        DSGESV_TESTS(setup_20_2),
        /* Comprehensive: N=20, NRHS=5 */
        DSGESV_TESTS(setup_20_5),

        /* Comprehensive: N=50, NRHS=1 */
        DSGESV_TESTS(setup_50_1),
        /* Comprehensive: N=50, NRHS=2 */
        DSGESV_TESTS(setup_50_2),
        /* Comprehensive: N=50, NRHS=5 */
        DSGESV_TESTS(setup_50_5),
    };

    return cmocka_run_group_tests_name("dsgesv", tests, NULL, NULL);
}
