/**
 * @file test_strevc.c
 * @brief CMocka test suite for strevc (eigenvector computation from Schur form).
 *
 * Tests the eigenvector computation routine strevc using LAPACK's
 * verification methodology with normalized residuals.
 *
 * The tests verify:
 *   - Right eigenvectors: | T*R - R*W | / ( |T| |R| ulp )
 *   - Left eigenvectors:  | L'*T - W'*L' | / ( |T| |L| ulp )
 *   - Eigenvector normalization (max-norm should be 1)
 *
 * Test approach (based on LAPACK dchkhs.f):
 *   1. Generate a random matrix A
 *   2. Compute Schur form: A = Z*T*Z' using reference LAPACK
 *   3. Call strevc to compute eigenvectors from T
 *   4. Verify eigenvector equation and normalization
 */

#include "test_harness.h"

/* Test threshold - matches LAPACK dchkhs.f */
#define THRESH 30.0f

#include "test_rng.h"
#include "verify.h"

/* Routine under test */
/* Utilities */
/* Reference LAPACK for Schur decomposition (from OpenBLAS) */
/*
 * Test fixture
 */
typedef struct {
    INT n;
    f32* T;       /* Schur form matrix */
    f32* VR;      /* Right eigenvectors */
    f32* VL;      /* Left eigenvectors */
    f32* wr;      /* Real parts of eigenvalues */
    f32* wi;      /* Imaginary parts of eigenvalues */
    f32* work;    /* Workspace */
    INT* select;     /* Selection array */
    uint64_t rng_state[4]; /* RNG state */
} dtrevc_fixture_t;

static uint64_t g_seed = 2024;

/**
 * Setup fixture
 */
static int dtrevc_setup(void** state, INT n)
{
    dtrevc_fixture_t* fix = malloc(sizeof(dtrevc_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    rng_seed(fix->rng_state, g_seed++);  /* Initialize RNG */

    /* Allocate arrays */
    fix->T = malloc(n * n * sizeof(f32));
    fix->VR = malloc(n * n * sizeof(f32));
    fix->VL = malloc(n * n * sizeof(f32));
    fix->wr = malloc(n * sizeof(f32));
    fix->wi = malloc(n * sizeof(f32));
    fix->work = malloc((n * (n + 1) + 10 * n) * sizeof(f32));
    fix->select = malloc(n * sizeof(INT));

    assert_non_null(fix->T);
    assert_non_null(fix->VR);
    assert_non_null(fix->VL);
    assert_non_null(fix->wr);
    assert_non_null(fix->wi);
    assert_non_null(fix->work);
    assert_non_null(fix->select);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture
 */
static int dtrevc_teardown(void** state)
{
    dtrevc_fixture_t* fix = *state;
    if (fix) {
        free(fix->T);
        free(fix->VR);
        free(fix->VL);
        free(fix->wr);
        free(fix->wi);
        free(fix->work);
        free(fix->select);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_2(void** state) { return dtrevc_setup(state, 2); }
static int setup_3(void** state) { return dtrevc_setup(state, 3); }
static int setup_4(void** state) { return dtrevc_setup(state, 4); }
static int setup_5(void** state) { return dtrevc_setup(state, 5); }
static int setup_6(void** state) { return dtrevc_setup(state, 6); }
static int setup_10(void** state) { return dtrevc_setup(state, 10); }
static int setup_20(void** state) { return dtrevc_setup(state, 20); }

/**
 * Generate a random upper quasi-triangular (Schur form) matrix.
 *
 * The matrix has:
 * - 1x1 blocks for real eigenvalues on the diagonal
 * - 2x2 blocks for complex conjugate pairs
 * - Random values in the strictly upper triangular part
 */
static void generate_schur_matrix(f32* T, INT n, f32* wr, f32* wi,
                                  uint64_t state[static 4])
{
    /* Initialize to zero */
    for (INT i = 0; i < n * n; i++)
        T[i] = 0.0f;

    /* Place eigenvalues on diagonal - mix of real and complex */
    INT j = 0;
    while (j < n) {
        f32 u = rng_uniform_f32(state);

        if (j < n - 1 && u < 0.4f) {
            /* Create a 2x2 block for complex conjugate pair */
            f32 real_part = rng_uniform_symmetric_f32(state);
            f32 imag_part = 0.5f + rng_uniform_f32(state);  /* Ensure nonzero */

            /* 2x2 block:  [ a   b ]
             *             [-c   a ]  where eigenvalues are a +/- i*sqrt(b*c)
             * We want eigenvalues real_part +/- i*imag_part
             * So b*c = imag_part^2, choose b = imag_part, c = imag_part
             */
            T[j + j * n] = real_part;
            T[j + 1 + (j + 1) * n] = real_part;
            T[j + (j + 1) * n] = imag_part;
            T[j + 1 + j * n] = -imag_part;

            wr[j] = real_part;
            wi[j] = imag_part;
            wr[j + 1] = real_part;
            wi[j + 1] = -imag_part;

            j += 2;
        } else {
            /* Real eigenvalue */
            f32 eig = rng_uniform_symmetric_f32(state);
            T[j + j * n] = eig;
            wr[j] = eig;
            wi[j] = 0.0f;
            j += 1;
        }
    }

    /* Fill strictly upper triangular part with random values
     * BUT preserve the 2x2 block structure:
     * - Don't modify T(j, j+1) if it's the superdiagonal of a 2x2 block
     * A 2x2 block is identified by T(j+1, j) != 0
     */
    for (INT col = 0; col < n; col++) {
        for (INT row = 0; row < col; row++) {
            /* Check if (row, col) is within a 2x2 block */
            /* Case 1: row is first row of 2x2 block and col = row+1 (superdiagonal) */
            if (row + 1 < n && T[row + 1 + row * n] != 0.0f && col == row + 1) {
                continue;  /* Don't overwrite 2x2 block superdiagonal */
            }
            /* Case 2: row is second row of 2x2 block - skip column row-1 through row */
            if (row > 0 && T[row + (row - 1) * n] != 0.0f && col <= row) {
                continue;  /* Already handled by 2x2 block */
            }
            T[row + col * n] = 0.5f * rng_uniform_symmetric_f32(state);
        }
    }
}

/**
 * Test right eigenvectors (SIDE='R', HOWMNY='A')
 */
static void test_right_eigenvectors(void** state)
{
    dtrevc_fixture_t* fix = *state;
    INT n = fix->n;
    INT info, m;
    f32 result[2];

    /* Generate Schur matrix */
    generate_schur_matrix(fix->T, n, fix->wr, fix->wi, fix->rng_state);

    /* Compute all right eigenvectors */
    strevc("R", "A", fix->select, n, fix->T, n,
           fix->VL, n, fix->VR, n, n, &m, fix->work, &info);

    assert_info_success(info);
    assert_int_equal(m, n);

    /* Verify: | T*R - R*W | / (|T| |R| ulp) */
    sget22("N", "N", "N", n, fix->T, n, fix->VR, n,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
    /* Check normalization */
    assert_residual_ok(result[1]);
}

/**
 * Test left eigenvectors (SIDE='L', HOWMNY='A')
 */
static void test_left_eigenvectors(void** state)
{
    dtrevc_fixture_t* fix = *state;
    INT n = fix->n;
    INT info, m;
    f32 result[2];

    /* Generate Schur matrix */
    generate_schur_matrix(fix->T, n, fix->wr, fix->wi, fix->rng_state);

    /* Compute all left eigenvectors */
    strevc("L", "A", fix->select, n, fix->T, n,
           fix->VL, n, fix->VR, n, n, &m, fix->work, &info);

    assert_info_success(info);
    assert_int_equal(m, n);

    /* Verify: | L'*T - W'*L' | / (|T| |L| ulp)
     * Using transposed form: transa='T', transe='N', transw='C' */
    sget22("T", "N", "C", n, fix->T, n, fix->VL, n,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

/**
 * Test both left and right eigenvectors (SIDE='B', HOWMNY='A')
 */
static void test_both_eigenvectors(void** state)
{
    dtrevc_fixture_t* fix = *state;
    INT n = fix->n;
    INT info, m;
    f32 result[2];

    /* Generate Schur matrix */
    generate_schur_matrix(fix->T, n, fix->wr, fix->wi, fix->rng_state);

    /* Compute both left and right eigenvectors */
    strevc("B", "A", fix->select, n, fix->T, n,
           fix->VL, n, fix->VR, n, n, &m, fix->work, &info);

    assert_info_success(info);
    assert_int_equal(m, n);

    /* Verify right eigenvectors */
    sget22("N", "N", "N", n, fix->T, n, fix->VR, n,
           fix->wr, fix->wi, fix->work, result);
    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);

    /* Verify left eigenvectors */
    sget22("T", "N", "C", n, fix->T, n, fix->VL, n,
           fix->wr, fix->wi, fix->work, result);
    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

/**
 * Test selected eigenvectors (HOWMNY='S')
 */
static void test_selected_eigenvectors(void** state)
{
    dtrevc_fixture_t* fix = *state;
    INT n = fix->n;
    INT info, m;

    if (n < 3) {
        skip_test("n too small for selection test");
        return;
    }

    /* Generate Schur matrix */
    generate_schur_matrix(fix->T, n, fix->wr, fix->wi, fix->rng_state);

    /* Select every other eigenvalue (respecting complex pairs) */
    INT nselected = 0;
    for (INT j = 0; j < n; j++) {
        if (fix->wi[j] == 0.0f) {
            /* Real eigenvalue */
            fix->select[j] = (j % 2 == 0);
            if (fix->select[j]) nselected++;
        } else if (fix->wi[j] > 0.0f) {
            /* First of complex pair - select both or neither */
            INT sel = (j % 3 == 0);
            fix->select[j] = sel;
            fix->select[j + 1] = 0;  /* Will be set to 0 by strevc */
            if (sel) nselected += 2;
            j++;  /* Skip the conjugate */
        }
    }

    if (nselected == 0) {
        /* Ensure at least one is selected */
        fix->select[0] = 1;
        nselected = (fix->wi[0] == 0.0f) ? 1 : 2;
    }

    /* Compute selected right eigenvectors */
    strevc("R", "S", fix->select, n, fix->T, n,
           fix->VL, n, fix->VR, n, n, &m, fix->work, &info);

    assert_info_success(info);
    assert_int_equal(m, nselected);
}

/**
 * Test backtransform (HOWMNY='B')
 *
 * When HOWMNY='B', strevc backtransforms the eigenvectors by
 * the matrix already in VR/VL.
 */
static void test_backtransform(void** state)
{
    dtrevc_fixture_t* fix = *state;
    INT n = fix->n;
    INT info, m;
    f32 result[2];

    /* Generate Schur matrix */
    generate_schur_matrix(fix->T, n, fix->wr, fix->wi, fix->rng_state);

    /* Initialize VR to identity (simulating Q from Schur decomposition) */
    slaset("F", n, n, 0.0f, 1.0f, fix->VR, n);

    /* Compute right eigenvectors with backtransform */
    strevc("R", "B", fix->select, n, fix->T, n,
           fix->VL, n, fix->VR, n, n, &m, fix->work, &info);

    assert_info_success(info);
    assert_int_equal(m, n);

    /* When Q=I, the result should be the same as HOWMNY='A'
     * Verify: | T*R - R*W | / (|T| |R| ulp) */
    sget22("N", "N", "N", n, fix->T, n, fix->VR, n,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
    assert_residual_ok(result[1]);
}

/**
 * Test with diagonal matrix (all real eigenvalues)
 */
static void test_diagonal_matrix(void** state)
{
    dtrevc_fixture_t* fix = *state;
    INT n = fix->n;
    INT info, m;
    f32 result[2];

    /* Create diagonal matrix */
    for (INT i = 0; i < n * n; i++)
        fix->T[i] = 0.0f;

    for (INT i = 0; i < n; i++) {
        fix->T[i + i * n] = (f32)(i + 1);  /* Eigenvalues 1, 2, ..., n */
        fix->wr[i] = (f32)(i + 1);
        fix->wi[i] = 0.0f;
    }

    /* Compute right eigenvectors */
    strevc("R", "A", fix->select, n, fix->T, n,
           fix->VL, n, fix->VR, n, n, &m, fix->work, &info);

    assert_info_success(info);

    /* For diagonal matrix, eigenvectors should be standard basis vectors */
    sget22("N", "N", "N", n, fix->T, n, fix->VR, n,
           fix->wr, fix->wi, fix->work, result);

    assert_residual_ok(result[0]);
}

/**
 * Test with matrix having all complex eigenvalues
 */
static void test_all_complex_eigenvalues(void** state)
{
    dtrevc_fixture_t* fix = *state;
    INT n = fix->n;
    INT info, m;
    f32 result[2];

    if (n % 2 != 0) {
        skip_test("n must be even for all-complex test");
        return;
    }

    /* Create matrix with only complex eigenvalue pairs */
    for (INT i = 0; i < n * n; i++)
        fix->T[i] = 0.0f;

    for (INT j = 0; j < n; j += 2) {
        f32 real_part = (f32)(j + 1);
        f32 imag_part = 0.5f;

        fix->T[j + j * n] = real_part;
        fix->T[j + 1 + (j + 1) * n] = real_part;
        fix->T[j + (j + 1) * n] = imag_part;
        fix->T[j + 1 + j * n] = -imag_part;

        fix->wr[j] = real_part;
        fix->wi[j] = imag_part;
        fix->wr[j + 1] = real_part;
        fix->wi[j + 1] = -imag_part;
    }

    /* Compute both eigenvectors */
    strevc("B", "A", fix->select, n, fix->T, n,
           fix->VL, n, fix->VR, n, n, &m, fix->work, &info);

    assert_info_success(info);
    assert_int_equal(m, n);

    /* Verify */
    sget22("N", "N", "N", n, fix->T, n, fix->VR, n,
           fix->wr, fix->wi, fix->work, result);
    assert_residual_ok(result[0]);

    sget22("T", "N", "C", n, fix->T, n, fix->VL, n,
           fix->wr, fix->wi, fix->work, result);
    assert_residual_ok(result[0]);
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Right eigenvectors */
        cmocka_unit_test_setup_teardown(test_right_eigenvectors, setup_2, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_right_eigenvectors, setup_3, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_right_eigenvectors, setup_4, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_right_eigenvectors, setup_5, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_right_eigenvectors, setup_6, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_right_eigenvectors, setup_10, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_right_eigenvectors, setup_20, dtrevc_teardown),

        /* Left eigenvectors */
        cmocka_unit_test_setup_teardown(test_left_eigenvectors, setup_2, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_left_eigenvectors, setup_3, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_left_eigenvectors, setup_5, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_left_eigenvectors, setup_10, dtrevc_teardown),

        /* Both eigenvectors */
        cmocka_unit_test_setup_teardown(test_both_eigenvectors, setup_4, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_both_eigenvectors, setup_6, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_both_eigenvectors, setup_10, dtrevc_teardown),

        /* Selected eigenvectors */
        cmocka_unit_test_setup_teardown(test_selected_eigenvectors, setup_5, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_selected_eigenvectors, setup_10, dtrevc_teardown),

        /* Backtransform */
        cmocka_unit_test_setup_teardown(test_backtransform, setup_4, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_backtransform, setup_6, dtrevc_teardown),

        /* Diagonal matrix (all real eigenvalues) */
        cmocka_unit_test_setup_teardown(test_diagonal_matrix, setup_5, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_diagonal_matrix, setup_10, dtrevc_teardown),

        /* All complex eigenvalues */
        cmocka_unit_test_setup_teardown(test_all_complex_eigenvalues, setup_4, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_all_complex_eigenvalues, setup_6, dtrevc_teardown),
        cmocka_unit_test_setup_teardown(test_all_complex_eigenvalues, setup_10, dtrevc_teardown),
    };

    return cmocka_run_group_tests_name("dtrevc", tests, NULL, NULL);
}
