/**
 * @file test_dgesc2.c
 * @brief CMocka test suite for dgesc2 (solve using LU with complete pivoting).
 *
 * Tests solving A * X = scale * B using complete pivoting LU from dgetc2.
 *
 * Verification: Compute ||b - A*x/scale|| / (||A|| * ||x|| * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dgetc2(const int n, f64 * const restrict A, const int lda,
                   int * const restrict ipiv, int * const restrict jpiv, int *info);
extern void dgesc2(const int n, const f64 * const restrict A, const int lda,
                   f64 * const restrict rhs, const int * const restrict ipiv,
                   const int * const restrict jpiv, f64 *scale);

/* Utilities */
extern f64 dlamch(const char *cmach);
extern f64 dlange(const char *norm, const int m, const int n,
                     const f64 * const restrict A, const int lda,
                     f64 * const restrict work);

/*
 * Compute ||b - A*x/scale|| / (||A|| * ||x|| * eps)
 */
static f64 compute_residual(int n, const f64 *A, int lda,
                               const f64 *x, const f64 *b,
                               f64 scale)
{
    f64 eps = dlamch("E");
    f64 *work = malloc(n * sizeof(f64));
    f64 *resid_vec = malloc(n * sizeof(f64));

    assert_non_null(work);
    assert_non_null(resid_vec);

    memcpy(resid_vec, b, n * sizeof(f64));
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n,
                -1.0 / scale, A, lda, x, 1, 1.0, resid_vec, 1);

    f64 anorm = dlange("1", n, n, A, lda, work);
    f64 xnorm = cblas_dnrm2(n, x, 1);
    f64 rnorm = cblas_dnrm2(n, resid_vec, 1);

    f64 result;
    if (anorm <= 0.0 || xnorm <= 0.0) {
        result = rnorm / eps;
    } else {
        result = rnorm / (anorm * xnorm * eps);
    }

    free(work);
    free(resid_vec);

    return result;
}

/*
 * Test fixture
 */
typedef struct {
    int n;
    int lda;
    f64 *A;       /* Factored matrix */
    f64 *A_orig;  /* Original matrix */
    f64 *b;       /* RHS (overwritten with solution) */
    f64 *b_orig;  /* Original RHS */
    f64 *x_exact; /* Known exact solution */
    f64 *d;       /* Singular values for dlatms */
    f64 *work;    /* Workspace */
    int *ipiv;       /* Row pivot indices */
    int *jpiv;       /* Column pivot indices */
    uint64_t seed;
} dgesc2_fixture_t;

static uint64_t g_seed = 1729;

static int dgesc2_setup(void **state, int n)
{
    dgesc2_fixture_t *fix = malloc(sizeof(dgesc2_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->lda = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->A_orig = malloc(fix->lda * n * sizeof(f64));
    fix->b = malloc(n * sizeof(f64));
    fix->b_orig = malloc(n * sizeof(f64));
    fix->x_exact = malloc(n * sizeof(f64));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(3 * n * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(int));
    fix->jpiv = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->A_orig);
    assert_non_null(fix->b);
    assert_non_null(fix->b_orig);
    assert_non_null(fix->x_exact);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->jpiv);

    *state = fix;
    return 0;
}

static int dgesc2_teardown(void **state)
{
    dgesc2_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->A_orig);
        free(fix->b);
        free(fix->b_orig);
        free(fix->x_exact);
        free(fix->d);
        free(fix->work);
        free(fix->ipiv);
        free(fix->jpiv);
        free(fix);
    }
    return 0;
}

static int setup_2(void **state) { return dgesc2_setup(state, 2); }
static int setup_3(void **state) { return dgesc2_setup(state, 3); }
static int setup_4(void **state) { return dgesc2_setup(state, 4); }
static int setup_5(void **state) { return dgesc2_setup(state, 5); }
static int setup_8(void **state) { return dgesc2_setup(state, 8); }

/**
 * Core test logic.
 */
static f64 run_dgesc2_test(dgesc2_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info;

    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    memcpy(fix->A_orig, fix->A, fix->lda * fix->n * sizeof(f64));

    /* Generate known exact solution */
    for (int i = 0; i < fix->n; i++) {
        fix->x_exact[i] = 1.0 + (f64)i / fix->n;
    }

    /* Compute b = A * x_exact */
    cblas_dgemv(CblasColMajor, CblasNoTrans, fix->n, fix->n,
                1.0, fix->A_orig, fix->lda, fix->x_exact, 1, 0.0, fix->b, 1);
    memcpy(fix->b_orig, fix->b, fix->n * sizeof(f64));

    /* Factor A using complete pivoting */
    dgetc2(fix->n, fix->A, fix->lda, fix->ipiv, fix->jpiv, &info);

    /* Solve A * x = scale * b */
    f64 scale;
    dgesc2(fix->n, fix->A, fix->lda, fix->b, fix->ipiv, fix->jpiv, &scale);

    /* Verify solution (b now contains x) */
    return compute_residual(fix->n, fix->A_orig, fix->lda, fix->b, fix->b_orig, scale);
}

/*
 * Test well-conditioned matrices (types 1-4).
 */
static void test_dgesc2_wellcond(void **state)
{
    dgesc2_fixture_t *fix = *state;
    f64 resid;

    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        resid = run_dgesc2_test(fix, imat);
        assert_residual_ok(resid);
    }
}

/*
 * Test ill-conditioned matrices (type 8).
 */
static void test_dgesc2_illcond(void **state)
{
    dgesc2_fixture_t *fix = *state;
    fix->seed = g_seed++;
    f64 resid = run_dgesc2_test(fix, 8);
    assert_residual_ok(resid);
}

/*
 * Sanity check: simple known system.
 */
static void test_dgesc2_simple(void **state)
{
    (void)state;

    f64 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    f64 A_orig[9];
    memcpy(A_orig, A, 9 * sizeof(f64));

    f64 b[3] = {4, 10, 24};
    f64 b_orig[3];
    memcpy(b_orig, b, 3 * sizeof(f64));

    int ipiv[3], jpiv[3];
    int info;
    f64 scale;

    dgetc2(3, A, 3, ipiv, jpiv, &info);
    dgesc2(3, A, 3, b, ipiv, jpiv, &scale);

    f64 resid = compute_residual(3, A_orig, 3, b, b_orig, scale);
    assert_residual_ok(resid);
}

/*
 * Sanity check: identity matrix.
 */
static void test_dgesc2_identity(void **state)
{
    (void)state;

    int n = 4;
    f64 *A = calloc(n * n, sizeof(f64));
    f64 *A_orig = calloc(n * n, sizeof(f64));
    f64 *b = malloc(n * sizeof(f64));
    f64 *b_orig = malloc(n * sizeof(f64));
    int *ipiv = malloc(n * sizeof(int));
    int *jpiv = malloc(n * sizeof(int));
    int info;
    f64 scale;

    assert_non_null(A);
    assert_non_null(A_orig);
    assert_non_null(b);
    assert_non_null(b_orig);

    for (int i = 0; i < n; i++) {
        A[i + i * n] = 1.0;
        A_orig[i + i * n] = 1.0;
        b[i] = (f64)(i + 1);
        b_orig[i] = (f64)(i + 1);
    }

    dgetc2(n, A, n, ipiv, jpiv, &info);
    dgesc2(n, A, n, b, ipiv, jpiv, &scale);

    f64 resid = compute_residual(n, A_orig, n, b, b_orig, scale);
    assert_residual_ok(resid);
    assert_true(fabs(scale - 1.0) < 1e-14);

    free(A);
    free(A_orig);
    free(b);
    free(b_orig);
    free(ipiv);
    free(jpiv);
}

#define DGESC2_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgesc2_wellcond, setup_fn, dgesc2_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesc2_illcond, setup_fn, dgesc2_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Sanity checks */
        cmocka_unit_test(test_dgesc2_simple),
        cmocka_unit_test(test_dgesc2_identity),

        /* Comprehensive tests */
        DGESC2_TESTS(setup_2),
        DGESC2_TESTS(setup_3),
        DGESC2_TESTS(setup_4),
        DGESC2_TESTS(setup_5),
        DGESC2_TESTS(setup_8),
    };

    return cmocka_run_group_tests_name("dgesc2", tests, NULL, NULL);
}
