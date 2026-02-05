/**
 * @file test_dgeevx.c
 * @brief Test expert eigenvalue computation routine dgeevx.
 *
 * Tests the expert driver which provides:
 * - Balancing for better accuracy
 * - Reciprocal condition numbers for eigenvalues and eigenvectors
 */

#include "test_harness.h"

/* Test threshold - matches LAPACK dchkhs.f */
#define THRESH 30.0

#include "testutils/verify.h"
#include "testutils/test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cblas.h>

/* Test fixture */
typedef struct {
    int n;
    double* A;        /* Original matrix */
    double* Acopy;    /* Copy for verification */
    double* VL;       /* Left eigenvectors */
    double* VR;       /* Right eigenvectors */
    double* wr;       /* Real eigenvalues */
    double* wi;       /* Imaginary eigenvalues */
    double* scale;    /* Scaling factors from balancing */
    double* rconde;   /* Reciprocal condition numbers for eigenvalues */
    double* rcondv;   /* Reciprocal condition numbers for eigenvectors */
    double* work;     /* Workspace */
    int* iwork;       /* Integer workspace */
    uint64_t seed;
} dgeevx_fixture_t;

/* Forward declarations from semicolon_lapack */
extern void dgeevx(const char* balanc, const char* jobvl, const char* jobvr,
                   const char* sense, const int n, double* A, const int lda,
                   double* wr, double* wi,
                   double* VL, const int ldvl, double* VR, const int ldvr,
                   int* ilo, int* ihi, double* scale, double* abnrm,
                   double* rconde, double* rcondv,
                   double* work, const int lwork, int* iwork, int* info);
extern void dlacpy(const char* uplo, const int m, const int n,
                   const double* A, const int lda, double* B, const int ldb);
extern void dlaset(const char* uplo, const int m, const int n,
                   const double alpha, const double beta,
                   double* A, const int lda);
extern double dlamch(const char* cmach);
extern double dlange(const char* norm, const int m, const int n,
                     const double* A, const int lda, double* work);

/* Setup function parameterized by N */
static int setup_N(void** state, int n) {
    dgeevx_fixture_t* fix = malloc(sizeof(dgeevx_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0xFEEDFACEULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(double));
    fix->Acopy = malloc(n * n * sizeof(double));
    fix->VL = malloc(n * n * sizeof(double));
    fix->VR = malloc(n * n * sizeof(double));
    fix->wr = malloc(n * sizeof(double));
    fix->wi = malloc(n * sizeof(double));
    fix->scale = malloc(n * sizeof(double));
    fix->rconde = malloc(n * sizeof(double));
    fix->rcondv = malloc(n * sizeof(double));
    fix->iwork = malloc(2 * n * sizeof(int));

    /* Workspace: generous allocation */
    int lwork = 12 * n * n;
    fix->work = malloc(lwork * sizeof(double));

    if (!fix->A || !fix->Acopy || !fix->VL || !fix->VR ||
        !fix->wr || !fix->wi || !fix->scale ||
        !fix->rconde || !fix->rcondv || !fix->iwork || !fix->work) {
        free(fix->A); free(fix->Acopy); free(fix->VL);
        free(fix->VR); free(fix->wr); free(fix->wi);
        free(fix->scale); free(fix->rconde); free(fix->rcondv);
        free(fix->iwork); free(fix->work);
        free(fix);
        return -1;
    }

    rng_seed(fix->seed);
    *state = fix;
    return 0;
}

static int teardown(void** state) {
    dgeevx_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->Acopy);
        free(fix->VL);
        free(fix->VR);
        free(fix->wr);
        free(fix->wi);
        free(fix->scale);
        free(fix->rconde);
        free(fix->rcondv);
        free(fix->iwork);
        free(fix->work);
        free(fix);
    }
    return 0;
}

/* Setup wrappers for different sizes */
static int setup_5(void** state) { return setup_N(state, 5); }
static int setup_10(void** state) { return setup_N(state, 10); }
static int setup_20(void** state) { return setup_N(state, 20); }

/**
 * Generate random test matrix.
 */
static void generate_random_matrix(int n, double* A, int lda, double anorm)
{
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            A[i + j * lda] = anorm * rng_uniform_symmetric();
        }
    }
}

/**
 * Test with no balancing and no condition numbers.
 */
static void test_basic(void** state)
{
    dgeevx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int ilo, ihi, info;
    double abnrm;
    double result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and eigenvectors without balancing or condition numbers */
    int lwork = 10 * n * n;
    dgeevx("N", "V", "V", "N", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }

    /* Test right eigenvectors: | A*VR - VR*W | / ( |A| |VR| ulp ) */
    dget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
    /* Note: result[1] is max-norm normalization error. LAPACK's dgeevx normalizes
     * to Euclidean norm=1, not max-norm=1, so result[1] is not a pass/fail
     * criterion (per ddrvev.f testing methodology). */

    /* Test left eigenvectors: | VL'*A - W'*VL' | / ( |A| |VL| ulp ) */
    dget22("T", "N", "C", n, fix->Acopy, lda, fix->VL, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);

    /* ilo and ihi should indicate full range when no balancing (0-based) */
    assert_int_equal(ilo, 0);
    assert_int_equal(ihi, n - 1);
}

/**
 * Test with balancing.
 */
static void test_with_balancing(void** state)
{
    dgeevx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int ilo, ihi, info;
    double abnrm;
    double result[2];

    /* Generate matrix with varying magnitudes to benefit from balancing */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            double scale_factor = pow(10.0, (double)(j - i));
            fix->A[i + j * lda] = scale_factor * rng_uniform_symmetric();
        }
    }

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute with balancing */
    int lwork = 10 * n * n;
    dgeevx("B", "V", "V", "N", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Verify abnrm is the 1-norm of the balanced matrix */
    assert_true(abnrm > 0.0);
    assert_true(isfinite(abnrm));

    /* Scale factors should be positive */
    for (int j = 0; j < n; j++) {
        assert_true(fix->scale[j] > 0.0);
    }

    /* Test eigenvectors */
    dget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test with condition numbers for eigenvalues.
 */
static void test_condition_eigenvalues(void** state)
{
    dgeevx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int ilo, ihi, info;
    double abnrm;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);

    /* Compute with condition numbers for eigenvalues */
    int lwork = 10 * n * n;
    dgeevx("N", "V", "V", "E", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Reciprocal condition numbers should be in (0, 1] */
    for (int j = 0; j < n; j++) {
        assert_true(fix->rconde[j] > 0.0);
        assert_true(fix->rconde[j] <= 1.0);
    }
}

/**
 * Test with condition numbers for eigenvectors.
 */
static void test_condition_eigenvectors(void** state)
{
    dgeevx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int ilo, ihi, info;
    double abnrm;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);

    /* Compute with condition numbers for eigenvectors */
    int lwork = 10 * n * n;
    dgeevx("N", "V", "V", "V", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Reciprocal condition numbers (sep) should be positive */
    for (int j = 0; j < n; j++) {
        assert_true(fix->rcondv[j] > 0.0);
        assert_true(isfinite(fix->rcondv[j]));
    }
}

/**
 * Test with both condition numbers.
 */
static void test_condition_both(void** state)
{
    dgeevx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int ilo, ihi, info;
    double abnrm;
    double result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute with both condition numbers */
    int lwork = 10 * n * n;
    dgeevx("B", "V", "V", "B", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Both sets of condition numbers should be valid */
    for (int j = 0; j < n; j++) {
        assert_true(fix->rconde[j] > 0.0);
        assert_true(fix->rconde[j] <= 1.0);
        assert_true(fix->rcondv[j] > 0.0);
    }

    /* Verify eigenvectors */
    dget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test workspace query.
 */
static void test_workspace_query(void** state)
{
    dgeevx_fixture_t* fix = *state;
    int n = fix->n;
    int ilo, ihi, info;
    double abnrm;
    double work_query;

    /* Query optimal workspace for various configurations */
    dgeevx("B", "V", "V", "B", n, fix->A, n, fix->wr, fix->wi,
           fix->VL, n, fix->VR, n, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, &work_query, -1, fix->iwork, &info);

    assert_info_success(info);
    assert_true(work_query >= (double)n);
}

/**
 * Test with diagonal matrix (known eigenvalues).
 */
static void test_diagonal_matrix(void** state)
{
    dgeevx_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int ilo, ihi, info;
    double abnrm;

    const double ZERO = 0.0;

    /* Create diagonal matrix with known eigenvalues */
    dlaset("F", n, n, ZERO, ZERO, fix->A, lda);
    for (int j = 0; j < n; j++) {
        fix->A[j + j * lda] = (double)(j + 1);
    }

    /* Compute eigenvalues and condition numbers */
    int lwork = 10 * n * n;
    dgeevx("N", "V", "V", "B", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* All eigenvalues should be real (wi = 0) */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], ZERO, 1e-10);
    }

    /* For diagonal matrix, condition numbers for eigenvalues should be 1 */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->rconde[j], 1.0, 1e-10);
    }
}

int main(void)
{
    const struct CMUnitTest tests_n5[] = {
        cmocka_unit_test_setup_teardown(test_basic, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_with_balancing, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrix, setup_5, teardown),
    };

    const struct CMUnitTest tests_n10[] = {
        cmocka_unit_test_setup_teardown(test_basic, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_condition_eigenvalues, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_condition_eigenvectors, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_condition_both, setup_10, teardown),
    };

    const struct CMUnitTest tests_n20[] = {
        cmocka_unit_test_setup_teardown(test_basic, setup_20, teardown),
        cmocka_unit_test_setup_teardown(test_with_balancing, setup_20, teardown),
    };

    int result = 0;
    result += cmocka_run_group_tests_name("dgeevx_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeevx_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeevx_n20", tests_n20, NULL, NULL);

    return result;
}
