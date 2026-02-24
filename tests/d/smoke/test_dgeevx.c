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

#include "verify.h"
#include "test_rng.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
/* Test fixture */
typedef struct {
    INT n;
    f64* A;        /* Original matrix */
    f64* Acopy;    /* Copy for verification */
    f64* VL;       /* Left eigenvectors */
    f64* VR;       /* Right eigenvectors */
    f64* wr;       /* Real eigenvalues */
    f64* wi;       /* Imaginary eigenvalues */
    f64* scale;    /* Scaling factors from balancing */
    f64* rconde;   /* Reciprocal condition numbers for eigenvalues */
    f64* rcondv;   /* Reciprocal condition numbers for eigenvectors */
    f64* work;     /* Workspace */
    INT* iwork;       /* Integer workspace */
    uint64_t seed;
    uint64_t rng_state[4];
} dgeevx_fixture_t;

/* Forward declarations from semicolon_lapack */
/* Setup function parameterized by N */
static int setup_N(void** state, int n) {
    dgeevx_fixture_t* fix = malloc(sizeof(dgeevx_fixture_t));
    if (!fix) return -1;

    fix->n = n;
    fix->seed = 0xFEEDFACEULL;

    /* Allocate matrices */
    fix->A = malloc(n * n * sizeof(f64));
    fix->Acopy = malloc(n * n * sizeof(f64));
    fix->VL = malloc(n * n * sizeof(f64));
    fix->VR = malloc(n * n * sizeof(f64));
    fix->wr = malloc(n * sizeof(f64));
    fix->wi = malloc(n * sizeof(f64));
    fix->scale = malloc(n * sizeof(f64));
    fix->rconde = malloc(n * sizeof(f64));
    fix->rcondv = malloc(n * sizeof(f64));
    fix->iwork = malloc(2 * n * sizeof(INT));

    /* Workspace: generous allocation */
    INT lwork = 12 * n * n;
    fix->work = malloc(lwork * sizeof(f64));

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

    rng_seed(fix->rng_state, fix->seed);
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
static void generate_random_matrix(INT n, f64* A, INT lda, f64 anorm,
                                   uint64_t state[static 4])
{
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            A[i + j * lda] = anorm * rng_uniform_symmetric(state);
        }
    }
}

/**
 * Test with no balancing and no condition numbers.
 */
static void test_basic(void** state)
{
    dgeevx_fixture_t* fix = *state;
    INT n = fix->n;
    INT lda = n;
    INT ilo, ihi, info;
    f64 abnrm;
    f64 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute eigenvalues and eigenvectors without balancing or condition numbers */
    INT lwork = 10 * n * n;
    dgeevx("N", "V", "V", "N", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Eigenvalues should be finite */
    for (INT j = 0; j < n; j++) {
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
    INT n = fix->n;
    INT lda = n;
    INT ilo, ihi, info;
    f64 abnrm;
    f64 result[2];

    /* Generate matrix with varying magnitudes to benefit from balancing */
    for (INT j = 0; j < n; j++) {
        for (INT i = 0; i < n; i++) {
            f64 scale_factor = pow(10.0, (f64)(j - i));
            fix->A[i + j * lda] = scale_factor * rng_uniform_symmetric(fix->rng_state);
        }
    }

    /* Keep a copy for verification */
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute with balancing */
    INT lwork = 10 * n * n;
    dgeevx("B", "V", "V", "N", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Verify abnrm is the 1-norm of the balanced matrix */
    assert_true(abnrm > 0.0);
    assert_true(isfinite(abnrm));

    /* Scale factors should be positive */
    for (INT j = 0; j < n; j++) {
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
    INT n = fix->n;
    INT lda = n;
    INT ilo, ihi, info;
    f64 abnrm;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Compute with condition numbers for eigenvalues */
    INT lwork = 10 * n * n;
    dgeevx("N", "V", "V", "E", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Reciprocal condition numbers should be in (0, 1] */
    for (INT j = 0; j < n; j++) {
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
    INT n = fix->n;
    INT lda = n;
    INT ilo, ihi, info;
    f64 abnrm;

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);

    /* Compute with condition numbers for eigenvectors */
    INT lwork = 10 * n * n;
    dgeevx("N", "V", "V", "V", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Reciprocal condition numbers (sep) should be positive */
    for (INT j = 0; j < n; j++) {
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
    INT n = fix->n;
    INT lda = n;
    INT ilo, ihi, info;
    f64 abnrm;
    f64 result[2];

    /* Generate random matrix */
    generate_random_matrix(n, fix->A, lda, 1.0, fix->rng_state);
    dlacpy(" ", n, n, fix->A, lda, fix->Acopy, lda);

    /* Compute with both condition numbers */
    INT lwork = 10 * n * n;
    dgeevx("B", "V", "V", "B", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* Both sets of condition numbers should be valid */
    for (INT j = 0; j < n; j++) {
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
    INT n = fix->n;
    INT ilo, ihi, info;
    f64 abnrm;
    f64 work_query;

    /* Query optimal workspace for various configurations */
    dgeevx("B", "V", "V", "B", n, fix->A, n, fix->wr, fix->wi,
           fix->VL, n, fix->VR, n, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, &work_query, -1, fix->iwork, &info);

    assert_info_success(info);
    assert_true(work_query >= (f64)n);
}

/**
 * Test with diagonal matrix (known eigenvalues).
 */
static void test_diagonal_matrix(void** state)
{
    dgeevx_fixture_t* fix = *state;
    INT n = fix->n;
    INT lda = n;
    INT ilo, ihi, info;
    f64 abnrm;

    const f64 ZERO = 0.0;

    /* Create diagonal matrix with known eigenvalues */
    dlaset("F", n, n, ZERO, ZERO, fix->A, lda);
    for (INT j = 0; j < n; j++) {
        fix->A[j + j * lda] = (f64)(j + 1);
    }

    /* Compute eigenvalues and condition numbers */
    INT lwork = 10 * n * n;
    dgeevx("N", "V", "V", "B", n, fix->A, lda, fix->wr, fix->wi,
           fix->VL, lda, fix->VR, lda, &ilo, &ihi, fix->scale, &abnrm,
           fix->rconde, fix->rcondv, fix->work, lwork, fix->iwork, &info);

    assert_info_success(info);

    /* All eigenvalues should be real (wi = 0) */
    for (INT j = 0; j < n; j++) {
        assert_double_equal(fix->wi[j], ZERO, 1e-10);
    }

    /* For diagonal matrix, condition numbers for eigenvalues should be 1 */
    for (INT j = 0; j < n; j++) {
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

    INT result = 0;
    result += cmocka_run_group_tests_name("dgeevx_n5", tests_n5, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeevx_n10", tests_n10, NULL, NULL);
    result += cmocka_run_group_tests_name("dgeevx_n20", tests_n20, NULL, NULL);

    return result;
}
