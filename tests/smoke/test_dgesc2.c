/**
 * @file test_dgesc2.c
 * @brief CMocka test suite for dgesc2 (solve using LU with complete pivoting).
 *
 * Tests solving A * X = scale * B using complete pivoting LU from dgetc2.
 *
 * Verification: Compute ||b - A*x/scale|| / (||A|| * ||x|| * eps)
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dgetc2(const int n, double * const restrict A, const int lda,
                   int * const restrict ipiv, int * const restrict jpiv, int *info);
extern void dgesc2(const int n, const double * const restrict A, const int lda,
                   double * const restrict rhs, const int * const restrict ipiv,
                   const int * const restrict jpiv, double *scale);

/* Matrix generation */
extern void dlatb4(const char *path, const int imat, const int m, const int n,
                   char *type, int *kl, int *ku, double *anorm, int *mode,
                   double *cndnum, char *dist);
extern void dlatms(const int m, const int n, const char *dist,
                   uint64_t seed, const char *sym, double *d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char *pack,
                   double *A, const int lda, double *work, int *info);

/* Utilities */
extern double dlamch(const char *cmach);
extern double dlange(const char *norm, const int m, const int n,
                     const double * const restrict A, const int lda,
                     double * const restrict work);

/*
 * Compute ||b - A*x/scale|| / (||A|| * ||x|| * eps)
 */
static double compute_residual(int n, const double *A, int lda,
                               const double *x, const double *b,
                               double scale)
{
    double eps = dlamch("E");
    double *work = malloc(n * sizeof(double));
    double *resid_vec = malloc(n * sizeof(double));

    assert_non_null(work);
    assert_non_null(resid_vec);

    memcpy(resid_vec, b, n * sizeof(double));
    cblas_dgemv(CblasColMajor, CblasNoTrans, n, n,
                -1.0 / scale, A, lda, x, 1, 1.0, resid_vec, 1);

    double anorm = dlange("1", n, n, A, lda, work);
    double xnorm = cblas_dnrm2(n, x, 1);
    double rnorm = cblas_dnrm2(n, resid_vec, 1);

    double result;
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
    double *A;       /* Factored matrix */
    double *A_orig;  /* Original matrix */
    double *b;       /* RHS (overwritten with solution) */
    double *b_orig;  /* Original RHS */
    double *x_exact; /* Known exact solution */
    double *d;       /* Singular values for dlatms */
    double *work;    /* Workspace */
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

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->A_orig = malloc(fix->lda * n * sizeof(double));
    fix->b = malloc(n * sizeof(double));
    fix->b_orig = malloc(n * sizeof(double));
    fix->x_exact = malloc(n * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(3 * n * sizeof(double));
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
static double run_dgesc2_test(dgesc2_fixture_t *fix, int imat)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;

    dlatb4("DGE", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    dlatms(fix->n, fix->n, &dist, fix->seed, &type, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    memcpy(fix->A_orig, fix->A, fix->lda * fix->n * sizeof(double));

    /* Generate known exact solution */
    for (int i = 0; i < fix->n; i++) {
        fix->x_exact[i] = 1.0 + (double)i / fix->n;
    }

    /* Compute b = A * x_exact */
    cblas_dgemv(CblasColMajor, CblasNoTrans, fix->n, fix->n,
                1.0, fix->A_orig, fix->lda, fix->x_exact, 1, 0.0, fix->b, 1);
    memcpy(fix->b_orig, fix->b, fix->n * sizeof(double));

    /* Factor A using complete pivoting */
    dgetc2(fix->n, fix->A, fix->lda, fix->ipiv, fix->jpiv, &info);

    /* Solve A * x = scale * b */
    double scale;
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
    double resid;

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
    double resid = run_dgesc2_test(fix, 8);
    assert_residual_ok(resid);
}

/*
 * Sanity check: simple known system.
 */
static void test_dgesc2_simple(void **state)
{
    (void)state;

    double A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    double A_orig[9];
    memcpy(A_orig, A, 9 * sizeof(double));

    double b[3] = {4, 10, 24};
    double b_orig[3];
    memcpy(b_orig, b, 3 * sizeof(double));

    int ipiv[3], jpiv[3];
    int info;
    double scale;

    dgetc2(3, A, 3, ipiv, jpiv, &info);
    dgesc2(3, A, 3, b, ipiv, jpiv, &scale);

    double resid = compute_residual(3, A_orig, 3, b, b_orig, scale);
    assert_residual_ok(resid);
}

/*
 * Sanity check: identity matrix.
 */
static void test_dgesc2_identity(void **state)
{
    (void)state;

    int n = 4;
    double *A = calloc(n * n, sizeof(double));
    double *A_orig = calloc(n * n, sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *b_orig = malloc(n * sizeof(double));
    int *ipiv = malloc(n * sizeof(int));
    int *jpiv = malloc(n * sizeof(int));
    int info;
    double scale;

    assert_non_null(A);
    assert_non_null(A_orig);
    assert_non_null(b);
    assert_non_null(b_orig);

    for (int i = 0; i < n; i++) {
        A[i + i * n] = 1.0;
        A_orig[i + i * n] = 1.0;
        b[i] = (double)(i + 1);
        b_orig[i] = (double)(i + 1);
    }

    dgetc2(n, A, n, ipiv, jpiv, &info);
    dgesc2(n, A, n, b, ipiv, jpiv, &scale);

    double resid = compute_residual(n, A_orig, n, b, b_orig, scale);
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
