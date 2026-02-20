/**
 * @file test_sgesc2.c
 * @brief CMocka test suite for sgesc2 (solve using LU with complete pivoting).
 *
 * Tests solving A * X = scale * B using complete pivoting LU from sgetc2.
 *
 * Verification: Compute ||b - A*x/scale|| / (||A|| * ||x|| * eps)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

/* Routines under test */
extern void sgetc2(const int n, f32 * const restrict A, const int lda,
                   int * const restrict ipiv, int * const restrict jpiv, int *info);
extern void sgesc2(const int n, const f32 * const restrict A, const int lda,
                   f32 * const restrict rhs, const int * const restrict ipiv,
                   const int * const restrict jpiv, f32 *scale);

/* Utilities */
extern f32 slamch(const char *cmach);
extern f32 slange(const char *norm, const int m, const int n,
                     const f32 * const restrict A, const int lda,
                     f32 * const restrict work);

/*
 * Compute ||b - A*x/scale|| / (||A|| * ||x|| * eps)
 */
static f32 compute_residual(int n, const f32 *A, int lda,
                               const f32 *x, const f32 *b,
                               f32 scale)
{
    f32 eps = slamch("E");
    f32 *work = malloc(n * sizeof(f32));
    f32 *resid_vec = malloc(n * sizeof(f32));

    assert_non_null(work);
    assert_non_null(resid_vec);

    memcpy(resid_vec, b, n * sizeof(f32));
    cblas_sgemv(CblasColMajor, CblasNoTrans, n, n,
                -1.0f / scale, A, lda, x, 1, 1.0f, resid_vec, 1);

    f32 anorm = slange("1", n, n, A, lda, work);
    f32 xnorm = cblas_snrm2(n, x, 1);
    f32 rnorm = cblas_snrm2(n, resid_vec, 1);

    f32 result;
    if (anorm <= 0.0f || xnorm <= 0.0f) {
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
    f32 *A;       /* Factored matrix */
    f32 *A_orig;  /* Original matrix */
    f32 *b;       /* RHS (overwritten with solution) */
    f32 *b_orig;  /* Original RHS */
    f32 *x_exact; /* Known exact solution */
    f32 *d;       /* Singular values for slatms */
    f32 *work;    /* Workspace */
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

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->A_orig = malloc(fix->lda * n * sizeof(f32));
    fix->b = malloc(n * sizeof(f32));
    fix->b_orig = malloc(n * sizeof(f32));
    fix->x_exact = malloc(n * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(3 * n * sizeof(f32));
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
static f32 run_dgesc2_test(dgesc2_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info;

    slatb4("SGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    memcpy(fix->A_orig, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Generate known exact solution */
    for (int i = 0; i < fix->n; i++) {
        fix->x_exact[i] = 1.0f + (f32)i / fix->n;
    }

    /* Compute b = A * x_exact */
    cblas_sgemv(CblasColMajor, CblasNoTrans, fix->n, fix->n,
                1.0f, fix->A_orig, fix->lda, fix->x_exact, 1, 0.0f, fix->b, 1);
    memcpy(fix->b_orig, fix->b, fix->n * sizeof(f32));

    /* Factor A using complete pivoting */
    sgetc2(fix->n, fix->A, fix->lda, fix->ipiv, fix->jpiv, &info);

    /* Solve A * x = scale * b */
    f32 scale;
    sgesc2(fix->n, fix->A, fix->lda, fix->b, fix->ipiv, fix->jpiv, &scale);

    /* Verify solution (b now contains x) */
    return compute_residual(fix->n, fix->A_orig, fix->lda, fix->b, fix->b_orig, scale);
}

/*
 * Test well-conditioned matrices (types 1-4).
 */
static void test_dgesc2_wellcond(void **state)
{
    dgesc2_fixture_t *fix = *state;
    f32 resid;

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
    f32 resid = run_dgesc2_test(fix, 8);
    assert_residual_ok(resid);
}

/*
 * Sanity check: simple known system.
 */
static void test_dgesc2_simple(void **state)
{
    (void)state;

    f32 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    f32 A_orig[9];
    memcpy(A_orig, A, 9 * sizeof(f32));

    f32 b[3] = {4, 10, 24};
    f32 b_orig[3];
    memcpy(b_orig, b, 3 * sizeof(f32));

    int ipiv[3], jpiv[3];
    int info;
    f32 scale;

    sgetc2(3, A, 3, ipiv, jpiv, &info);
    sgesc2(3, A, 3, b, ipiv, jpiv, &scale);

    f32 resid = compute_residual(3, A_orig, 3, b, b_orig, scale);
    assert_residual_ok(resid);
}

/*
 * Sanity check: identity matrix.
 */
static void test_dgesc2_identity(void **state)
{
    (void)state;

    int n = 4;
    f32 *A = calloc(n * n, sizeof(f32));
    f32 *A_orig = calloc(n * n, sizeof(f32));
    f32 *b = malloc(n * sizeof(f32));
    f32 *b_orig = malloc(n * sizeof(f32));
    int *ipiv = malloc(n * sizeof(int));
    int *jpiv = malloc(n * sizeof(int));
    int info;
    f32 scale;

    assert_non_null(A);
    assert_non_null(A_orig);
    assert_non_null(b);
    assert_non_null(b_orig);

    for (int i = 0; i < n; i++) {
        A[i + i * n] = 1.0f;
        A_orig[i + i * n] = 1.0f;
        b[i] = (f32)(i + 1);
        b_orig[i] = (f32)(i + 1);
    }

    sgetc2(n, A, n, ipiv, jpiv, &info);
    sgesc2(n, A, n, b, ipiv, jpiv, &scale);

    f32 resid = compute_residual(n, A_orig, n, b, b_orig, scale);
    assert_residual_ok(resid);
    assert_true(fabsf(scale - 1.0f) < 1e-14);

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
