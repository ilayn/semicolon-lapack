/**
 * @file test_dgeev.c
 * @brief Test eigenvalue computation routine dgeev.
 *
 * Tests based on LAPACK TESTING/EIG/dchkhs.f, adapted to CMocka framework.
 * Uses dget22 to verify eigenvector correctness:
 *   | A*X - X*W | / ( |A| |X| ulp )  for right eigenvectors
 *   | Y'*A - W'*Y' | / ( |A| |Y| ulp )  for left eigenvectors
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
    double* A;       /* Original matrix */
    double* Acopy;   /* Copy for verification */
    double* VL;      /* Left eigenvectors */
    double* VR;      /* Right eigenvectors */
    double* wr;      /* Real eigenvalues */
    double* wi;      /* Imaginary eigenvalues */
    double* work;    /* Workspace */
    uint64_t seed;
} dgeev_fixture_t;

/* Forward declarations from semicolon_lapack */
extern void dgeev(const char* jobvl, const char* jobvr, const int n,
                  double* A, const int lda, double* wr, double* wi,
                  double* VL, const int ldvl, double* VR, const int ldvr,
                  double* work, const int lwork, int* info);
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
    dgeev_fixture_t* fix = malloc(sizeof(dgeev_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0xDEADBEEFULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(double));
    fix->Acopy = malloc(n * n * sizeof(double));
    fix->VL = malloc(n * n * sizeof(double));
    fix->VR = malloc(n * n * sizeof(double));
    fix->wr = malloc(n * sizeof(double));
    fix->wi = malloc(n * sizeof(double));

    /* Workspace: generous allocation */
    int lwork = 10 * n * n;
    fix->work = malloc(lwork * sizeof(double));

    if (!fix->A || !fix->Acopy || !fix->VL || !fix->VR ||
        !fix->wr || !fix->wi || !fix->work) {
        free(fix->A); free(fix->Acopy); free(fix->VL);
        free(fix->VR); free(fix->wr); free(fix->wi); free(fix->work);
        free(fix);
        return -1;
    }

    rng_seed(fix->seed);
    *state = fix;
    return 0;
}

static int teardown(void** state) {
    dgeev_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->Acopy);
        free(fix->VL);
        free(fix->VR);
        free(fix->wr);
        free(fix->wi);
        free(fix->work);
        free(fix);
    }
    return 0;
}

/* Setup wrappers for different sizes */
static int setup_5(void** state) { return setup_N(state, 5); }
static int setup_10(void** state) { return setup_N(state, 10); }
static int setup_20(void** state) { return setup_N(state, 20); }
static int setup_32(void** state) { return setup_N(state, 32); }

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
 * Test eigenvalue computation with both left and right eigenvectors.
 */
static void test_eigenvectors_both(dgeev_fixture_t* fix)
{
    int n = fix->n;
    int lda = n;
    int info;
    double result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and eigenvectors */
    int lwork = 8 * n * n;
    dgeev("V", "V", n, fix->A, lda, fix->wr, fix->wi,
          fix->VL, lda, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }

    /* Test right eigenvectors: | A*VR - VR*W | / ( |A| |VR| ulp ) */
    dget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);  /* Eigenvector accuracy */
    /* Note: result[1] is max-norm normalization error. LAPACK's dgeev normalizes
     * to Euclidean norm=1, not max-norm=1, so result[1] is not a pass/fail
     * criterion (per ddrvev.f - only result[0] is used for the test). */

    /* Test left eigenvectors: | VL'*A - W'*VL' | / ( |A| |VL| ulp ) */
    dget22("T", "N", "C", n, fix->Acopy, lda, fix->VL, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test eigenvalue computation with right eigenvectors only.
 */
static void test_eigenvectors_right(dgeev_fixture_t* fix)
{
    int n = fix->n;
    int lda = n;
    int info;
    double result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and right eigenvectors only */
    int lwork = 8 * n * n;
    dgeev("N", "V", n, fix->A, lda, fix->wr, fix->wi,
          NULL, 1, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* Test right eigenvectors */
    dget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test eigenvalue computation with left eigenvectors only.
 */
static void test_eigenvectors_left(dgeev_fixture_t* fix)
{
    int n = fix->n;
    int lda = n;
    int info;
    double result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and left eigenvectors only */
    int lwork = 8 * n * n;
    dgeev("V", "N", n, fix->A, lda, fix->wr, fix->wi,
          fix->VL, lda, NULL, 1, fix->work, lwork, &info);

    assert_info_success(info);

    /* Test left eigenvectors */
    dget22("T", "N", "C", n, fix->Acopy, lda, fix->VL, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test eigenvalue computation only (no eigenvectors).
 */
static void test_eigenvalues_only(void** state)
{
    dgeev_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int info;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0);

    /* Compute eigenvalues only */
    int lwork = 8 * n * n;
    dgeev("N", "N", n, fix->A, lda, fix->wr, fix->wi,
          NULL, 1, NULL, 1, fix->work, lwork, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (int j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }
}

/**
 * Test workspace query.
 */
static void test_workspace_query(void** state)
{
    dgeev_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    double work_query;

    /* Query optimal workspace for various configurations */
    dgeev("V", "V", n, fix->A, n, fix->wr, fix->wi,
          fix->VL, n, fix->VR, n, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (double)(3 * n));

    dgeev("N", "V", n, fix->A, n, fix->wr, fix->wi,
          NULL, 1, fix->VR, n, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (double)(3 * n));

    dgeev("N", "N", n, fix->A, n, fix->wr, fix->wi,
          NULL, 1, NULL, 1, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (double)(3 * n));
}

/**
 * Test with symmetric matrix (real eigenvalues).
 */
static void test_symmetric_matrix(void** state)
{
    dgeev_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int info;
    double result[2];

    /* Generate symmetric random matrix */
    for (int j = 0; j < n; j++) {
        for (int i = j; i < n; i++) {
            double val = rng_uniform_symmetric();
            fix->A[i + j * lda] = val;
            fix->A[j + i * lda] = val;
        }
    }

    /* Keep a copy */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and right eigenvectors */
    int lwork = 8 * n * n;
    dgeev("N", "V", n, fix->A, lda, fix->wr, fix->wi,
          NULL, 1, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* For symmetric matrix, all eigenvalues should be real */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], 0.0, 1e-10);
    }

    /* Test eigenvectors */
    dget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test with diagonal matrix.
 */
static void test_diagonal_matrix(void** state)
{
    dgeev_fixture_t* fix = *state;
    int n = fix->n;
    int lda = n;
    int info;

    const double ZERO = 0.0;

    /* Create diagonal matrix with known eigenvalues */
    dlaset("F", n, n, ZERO, ZERO, fix->A, lda);
    for (int j = 0; j < n; j++) {
        fix->A[j + j * lda] = (double)(j + 1);
    }

    /* Keep a copy */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and eigenvectors */
    int lwork = 8 * n * n;
    dgeev("V", "V", n, fix->A, lda, fix->wr, fix->wi,
          fix->VL, lda, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* All eigenvalues should be real (wi = 0) */
    for (int j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], ZERO, 1e-10);
    }

    /* Each eigenvalue should be in the set {1, 2, ..., n} */
    /* (order may differ) */
    int found[32] = {0};  /* Assuming n <= 32 */
    for (int j = 0; j < n; j++) {
        int idx = (int)(fix->wr[j] + 0.5) - 1;
        if (idx >= 0 && idx < n) {
            found[idx] = 1;
        }
    }
    for (int j = 0; j < n; j++) {
        assert_true(found[j]);
    }
}

/* Test wrappers */
static void test_both_n5(void** state) { test_eigenvectors_both(*state); }
static void test_both_n10(void** state) { test_eigenvectors_both(*state); }
static void test_both_n20(void** state) { test_eigenvectors_both(*state); }
static void test_both_n32(void** state) { test_eigenvectors_both(*state); }

static void test_right_n10(void** state) { test_eigenvectors_right(*state); }
static void test_left_n10(void** state) { test_eigenvectors_left(*state); }

int main(void)
{
    const struct CMUnitTest tests_n5[] = {
        cmocka_unit_test_setup_teardown(test_both_n5, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_eigenvalues_only, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_workspace_query, setup_5, teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrix, setup_5, teardown),
    };

    const struct CMUnitTest tests_n10[] = {
        cmocka_unit_test_setup_teardown(test_both_n10, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_right_n10, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_left_n10, setup_10, teardown),
        cmocka_unit_test_setup_teardown(test_symmetric_matrix, setup_10, teardown),
    };

    const struct CMUnitTest tests_n20[] = {
        cmocka_unit_test_setup_teardown(test_both_n20, setup_20, teardown),
    };

    const struct CMUnitTest tests_n32[] = {
        cmocka_unit_test_setup_teardown(test_both_n32, setup_32, teardown),
    };

    int result = 0;
    result += cmocka_run_group_tests_name("dgeev_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeev_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeev_n20", tests_n20, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeev_n32", tests_n32, NULL, NULL);

    return result;
}
