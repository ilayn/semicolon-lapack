/**
 * @file test_sgeev.c
 * @brief Test eigenvalue computation routine sgeev.
 *
 * Tests based on LAPACK TESTING/EIG/dchkhs.f, adapted to CMocka framework.
 * Uses sget22 to verify eigenvector correctness:
 *   | A*X - X*W | / ( |A| |X| ulp )  for right eigenvectors
 *   | Y'*A - W'*Y' | / ( |A| |Y| ulp )  for left eigenvectors
 */

#include "test_harness.h"

/* Test threshold - matches LAPACK dchkhs.f */
#define THRESH 30.0f

#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
/* Test fixture */
typedef struct {
    INT n;
    f32* A;       /* Original matrix */
    f32* Acopy;   /* Copy for verification */
    f32* VL;      /* Left eigenvectors */
    f32* VR;      /* Right eigenvectors */
    f32* wr;      /* Real eigenvalues */
    f32* wi;      /* Imaginary eigenvalues */
    f32* work;    /* Workspace */
    uint64_t seed;
    uint64_t rng_state[4];
} dgeev_fixture_t;

/* Forward declarations from semicolon_lapack */
/* Setup function parameterized by N */
static int setup_N(void** state, int n) {
    dgeev_fixture_t* fix = malloc(sizeof(dgeev_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0xDEADBEEFULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(f32));
    fix->Acopy = malloc(n * n * sizeof(f32));
    fix->VL = malloc(n * n * sizeof(f32));
    fix->VR = malloc(n * n * sizeof(f32));
    fix->wr = malloc(n * sizeof(f32));
    fix->wi = malloc(n * sizeof(f32));

    /* Workspace: generous allocation */
    INT lwork = 10 * n * n;
    fix->work = malloc(lwork * sizeof(f32));

    if (!fix->A || !fix->Acopy || !fix->VL || !fix->VR ||
        !fix->wr || !fix->wi || !fix->work) {
        free(fix->A); free(fix->Acopy); free(fix->VL);
        free(fix->VR); free(fix->wr); free(fix->wi); free(fix->work);
        free(fix);
        return -1;
    }

    rng_seed(fix->rng_state, fix->seed);
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
static void generate_random_matrix(INT n, f32* A, INT lda, f32 anorm,
                                   uint64_t state[static 4])
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            A[i + j * lda] = anorm * rng_uniform_symmetric_f32(state);
        }
    }
}

/**
 * Test eigenvalue computation with both left and right eigenvectors.
 */
static void test_eigenvectors_both(dgeev_fixture_t* fix)
{
    INT n = fix->n;
    INT lda = n;
    INT info;
    f32 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0f, fix->rng_state);

    /* Keep a copy for verification */
    slacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and eigenvectors */
    INT lwork = 8 * n * n;
    sgeev("V", "V", n, fix->A, lda, fix->wr, fix->wi,
          fix->VL, lda, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (INT j = 0; j < n; j++) {
        assert_true(isfinite(fix->wr[j]));
        assert_true(isfinite(fix->wi[j]));
    }

    /* Test right eigenvectors: | A*VR - VR*W | / ( |A| |VR| ulp ) */
    sget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);  /* Eigenvector accuracy */
    /* Note: result[1] is max-norm normalization error. LAPACK's sgeev normalizes
     * to Euclidean norm=1, not max-norm=1, so result[1] is not a pass/fail
     * criterion (per ddrvev.f - only result[0] is used for the test). */

    /* Test left eigenvectors: | VL'*A - W'*VL' | / ( |A| |VL| ulp ) */
    sget22("T", "N", "C", n, fix->Acopy, lda, fix->VL, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test eigenvalue computation with right eigenvectors only.
 */
static void test_eigenvectors_right(dgeev_fixture_t* fix)
{
    INT n = fix->n;
    INT lda = n;
    INT info;
    f32 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0f, fix->rng_state);

    /* Keep a copy for verification */
    slacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and right eigenvectors only */
    INT lwork = 8 * n * n;
    sgeev("N", "V", n, fix->A, lda, fix->wr, fix->wi,
          NULL, 1, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* Test right eigenvectors */
    sget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test eigenvalue computation with left eigenvectors only.
 */
static void test_eigenvectors_left(dgeev_fixture_t* fix)
{
    INT n = fix->n;
    INT lda = n;
    INT info;
    f32 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0f, fix->rng_state);

    /* Keep a copy for verification */
    slacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and left eigenvectors only */
    INT lwork = 8 * n * n;
    sgeev("V", "N", n, fix->A, lda, fix->wr, fix->wi,
          fix->VL, lda, NULL, 1, fix->work, lwork, &info);

    assert_info_success(info);

    /* Test left eigenvectors */
    sget22("T", "N", "C", n, fix->Acopy, lda, fix->VL, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test eigenvalue computation only (no eigenvectors).
 */
static void test_eigenvalues_only(void** state)
{
    dgeev_fixture_t* fix = *state;
    INT n = fix->n;
    INT lda = n;
    INT info;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0f, fix->rng_state);

    /* Compute eigenvalues only */
    INT lwork = 8 * n * n;
    sgeev("N", "N", n, fix->A, lda, fix->wr, fix->wi,
          NULL, 1, NULL, 1, fix->work, lwork, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (INT j = 0; j < n; j++) {
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
    INT n = fix->n;
    INT info;
    f32 work_query;

    /* Query optimal workspace for various configurations */
    sgeev("V", "V", n, fix->A, n, fix->wr, fix->wi,
          fix->VL, n, fix->VR, n, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (f32)(3 * n));

    sgeev("N", "V", n, fix->A, n, fix->wr, fix->wi,
          NULL, 1, fix->VR, n, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (f32)(3 * n));

    sgeev("N", "N", n, fix->A, n, fix->wr, fix->wi,
          NULL, 1, NULL, 1, &work_query, -1, &info);
    assert_info_success(info);
    assert_true(work_query >= (f32)(3 * n));
}

/**
 * Test with symmetric matrix (real eigenvalues).
 */
static void test_symmetric_matrix(void** state)
{
    dgeev_fixture_t* fix = *state;
    INT n = fix->n;
    INT lda = n;
    INT info;
    f32 result[2];

    /* Generate symmetric random matrix */
    for (INT j = 0; j < n; j++) {
        for (INT i = j; i < n; i++) {
            f32 val = rng_uniform_symmetric_f32(fix->rng_state);
            fix->A[i + j * lda] = val;
            fix->A[j + i * lda] = val;
        }
    }

    /* Keep a copy */
    slacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and right eigenvectors */
    INT lwork = 8 * n * n;
    sgeev("N", "V", n, fix->A, lda, fix->wr, fix->wi,
          NULL, 1, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* For symmetric matrix, all eigenvalues should be real */
    for (INT j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], 0.0f, 1e-10);
    }

    /* Test eigenvectors */
    sget22("N", "N", "N", n, fix->Acopy, lda, fix->VR, lda,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test with diagonal matrix.
 */
static void test_diagonal_matrix(void** state)
{
    dgeev_fixture_t* fix = *state;
    INT n = fix->n;
    INT lda = n;
    INT info;

    const f32 ZERO = 0.0f;

    /* Create diagonal matrix with known eigenvalues */
    slaset("F", n, n, ZERO, ZERO, fix->A, lda);
    for (INT j = 0; j < n; j++) {
        fix->A[j + j * lda] = (f32)(j + 1);
    }

    /* Keep a copy */
    slacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and eigenvectors */
    INT lwork = 8 * n * n;
    sgeev("V", "V", n, fix->A, lda, fix->wr, fix->wi,
          fix->VL, lda, fix->VR, lda, fix->work, lwork, &info);

    assert_info_success(info);

    /* All eigenvalues should be real (wi = 0) */
    for (INT j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], ZERO, 1e-10);
    }

    /* Each eigenvalue should be in the set {1, 2, ..., n} */
    /* (order may differ) */
    INT found[32] = {0};  /* Assuming n <= 32 */
    for (INT j = 0; j < n; j++) {
        INT idx = (INT)(fix->wr[j] + 0.5f) - 1;
        if (idx >= 0 && idx < n) {
            found[idx] = 1;
        }
    }
    for (INT j = 0; j < n; j++) {
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

    INT result = 0;
    result += cmocka_run_group_tests_name("dgeev_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeev_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeev_n20", tests_n20, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeev_n32", tests_n32, NULL, NULL);

    return result;
}
