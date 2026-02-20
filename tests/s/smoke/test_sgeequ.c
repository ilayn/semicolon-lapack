/**
 * @file test_sgeequ.c
 * @brief CMocka test suite for sgeequ (row and column equilibration).
 *
 * Tests the equilibration routine sgeequ which computes row and column
 * scaling factors to equilibrate a general M-by-N matrix A.
 *
 * Verification:
 * - Row scaling: R(i) should make max|A(i,j)*R(i)| ~ 1
 * - Column scaling: C(j) should make max|A(i,j)*C(j)| ~ 1
 * - ROWCND near 1 means rows are well-balanced
 * - COLCND near 1 means columns are well-balanced
 *
 * Uses THRESH=0.1 for equilibration quality checks (not the standard 30.0).
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f

/* Routine under test */
extern void sgeequ(const int m, const int n, const f32 * const restrict A,
                   const int lda, f32 * const restrict R,
                   f32 * const restrict C, f32 *rowcnd, f32 *colcnd,
                   f32 *amax, int *info);

/* Utilities */
extern f32 slamch(const char *cmach);

/* Equilibration quality threshold */
static const f32 DGEEQU_THRESH = 0.1f;

/*
 * Test fixture: holds all allocated memory for a single test case.
 * CMocka passes this between setup -> test -> teardown.
 */
typedef struct {
    int m, n;
    int lda;
    f32 *A;       /* Matrix */
    f32 *R;       /* Row scaling factors */
    f32 *C;       /* Column scaling factors */
    f32 *row_max; /* Workspace for row max verification */
    f32 *col_max; /* Workspace for column max verification */
} dgeequ_fixture_t;

/**
 * Generate a matrix with varying row/column magnitudes.
 * Each row is scaled by row_scale^i and each column by col_scale^j.
 */
static void generate_unbalanced_matrix(int m, int n, f32 *A, int lda,
                                       f32 row_scale, f32 col_scale)
{
    for (int j = 0; j < n; j++) {
        f32 cscale = powf(col_scale, j);
        for (int i = 0; i < m; i++) {
            f32 rscale = powf(row_scale, i);
            A[i + j * lda] = rscale * cscale * (1.0f + (f32)(i + j) / (m + n));
        }
    }
}

/**
 * Setup fixture: allocate memory for given dimensions.
 * Called before each test function.
 */
static int dgeequ_setup(void **state, int m, int n)
{
    dgeequ_fixture_t *fix = malloc(sizeof(dgeequ_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    fix->lda = m;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->R = malloc(m * sizeof(f32));
    fix->C = malloc(n * sizeof(f32));
    fix->row_max = calloc(m, sizeof(f32));
    fix->col_max = calloc(n, sizeof(f32));

    assert_non_null(fix->A);
    assert_non_null(fix->R);
    assert_non_null(fix->C);
    assert_non_null(fix->row_max);
    assert_non_null(fix->col_max);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 * Called after each test function.
 */
static int dgeequ_teardown(void **state)
{
    dgeequ_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->R);
        free(fix->C);
        free(fix->row_max);
        free(fix->col_max);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions: square matrices */
static int setup_2x2(void **state) { return dgeequ_setup(state, 2, 2); }
static int setup_3x3(void **state) { return dgeequ_setup(state, 3, 3); }
static int setup_5x5(void **state) { return dgeequ_setup(state, 5, 5); }
static int setup_10x10(void **state) { return dgeequ_setup(state, 10, 10); }

/* Size-specific setup functions: rectangular M > N */
static int setup_5x3(void **state) { return dgeequ_setup(state, 5, 3); }
static int setup_7x5(void **state) { return dgeequ_setup(state, 7, 5); }
static int setup_12x10(void **state) { return dgeequ_setup(state, 12, 10); }

/* Size-specific setup functions: rectangular M < N */
static int setup_3x5(void **state) { return dgeequ_setup(state, 3, 5); }
static int setup_5x7(void **state) { return dgeequ_setup(state, 5, 7); }
static int setup_10x12(void **state) { return dgeequ_setup(state, 10, 12); }

/**
 * Test sgeequ with a matrix having known scaling properties.
 * Verifies: info==0, amax correctness, R/C positivity, row/column equilibration.
 */
static void test_dgeequ_equilibration(void **state)
{
    dgeequ_fixture_t *fix = *state;
    int m = fix->m, n = fix->n, lda = fix->lda;
    f32 rowcnd, colcnd, amax;
    int info;

    /* Generate matrix with exponentially varying row/column magnitudes */
    generate_unbalanced_matrix(m, n, fix->A, lda, 10.0f, 10.0f);

    /* Compute expected amax (largest element) */
    f32 expected_amax = 0.0f;
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            f32 val = fabsf(fix->A[i + j * lda]);
            if (val > expected_amax) expected_amax = val;
        }
    }

    /* Call sgeequ */
    sgeequ(m, n, fix->A, lda, fix->R, fix->C, &rowcnd, &colcnd, &amax, &info);

    /* info must be 0 (no zero rows/columns in unbalanced matrix) */
    assert_int_equal(info, 0);

    /* Verify amax matches expected */
    f32 amax_err = fabsf(amax - expected_amax) / expected_amax;
    assert_residual_below(amax_err, 1e-10);

    /* Verify that scaling factors are positive */
    int r_positive = 1;
    for (int i = 0; i < m; i++) {
        if (fix->R[i] <= 0.0f) r_positive = 0;
    }
    assert_true(r_positive);

    int c_positive = 1;
    for (int j = 0; j < n; j++) {
        if (fix->C[j] <= 0.0f) c_positive = 0;
    }
    assert_true(c_positive);

    /* Verify that after scaling by R, each row's max is approximately 1 */
    memset(fix->row_max, 0, m * sizeof(f32));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            f32 scaled = fabsf(fix->A[i + j * lda]) * fix->R[i];
            if (scaled > fix->row_max[i]) fix->row_max[i] = scaled;
        }
    }

    int row_equil = 1;
    for (int i = 0; i < m; i++) {
        if (fabsf(fix->row_max[i] - 1.0f) > DGEEQU_THRESH) row_equil = 0;
    }
    assert_true(row_equil);

    /* Verify that after scaling by R and C, each column's max is approximately 1.
     * C is computed to equilibrate the row-scaled matrix, so the correct
     * check is R(i)*A(i,j)*C(j), not just A(i,j)*C(j). */
    memset(fix->col_max, 0, n * sizeof(f32));
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            f32 scaled = fabsf(fix->A[i + j * lda]) * fix->R[i] * fix->C[j];
            if (scaled > fix->col_max[j]) fix->col_max[j] = scaled;
        }
    }

    int col_equil = 1;
    for (int j = 0; j < n; j++) {
        if (fabsf(fix->col_max[j] - 1.0f) > DGEEQU_THRESH) col_equil = 0;
    }
    assert_true(col_equil);
}

/**
 * Test sgeequ with identity matrix (should need no scaling).
 * Verifies: info==0, amax==1, rowcnd==1, colcnd==1.
 */
static void test_dgeequ_identity(void **state)
{
    dgeequ_fixture_t *fix = *state;
    int n = fix->n;
    int lda = fix->lda;
    f32 rowcnd, colcnd, amax;
    int info;

    /* Only applicable for square matrices */
    if (fix->m != fix->n) {
        skip();
    }

    /* Set up identity matrix */
    memset(fix->A, 0, lda * n * sizeof(f32));
    for (int i = 0; i < n; i++) {
        fix->A[i + i * lda] = 1.0f;
    }

    sgeequ(n, n, fix->A, lda, fix->R, fix->C, &rowcnd, &colcnd, &amax, &info);

    /* info must be 0 */
    assert_int_equal(info, 0);

    /* For identity: amax=1, rowcnd=1, colcnd=1 */
    assert_true((double)fabsf(amax - 1.0f) < 1e-14);
    assert_true((double)fabsf(rowcnd - 1.0f) < 1e-14);
    assert_true((double)fabsf(colcnd - 1.0f) < 1e-14);
}

/**
 * Test sgeequ with matrix containing a zero row.
 * Verifies: info==1 (first row is zero, 1-based index).
 */
static void test_dgeequ_zero_row(void **state)
{
    dgeequ_fixture_t *fix = *state;
    int m = fix->m, n = fix->n, lda = fix->lda;
    f32 rowcnd, colcnd, amax;
    int info;

    /* Only applicable for matrices with m >= 2 */
    if (m < 2) {
        skip();
    }

    /* Matrix with zero first row */
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
            fix->A[i + j * lda] = (i == 0) ? 0.0f : 1.0f;
        }
    }

    sgeequ(m, n, fix->A, lda, fix->R, fix->C, &rowcnd, &colcnd, &amax, &info);

    /* info should be 1 (first row is zero, 1-based reporting) */
    assert_int_equal(info, 1);
}

/*
 * Macro to generate test entries for a given size.
 * Creates 3 test cases: equilibration, identity, zero_row.
 */
#define DGEEQU_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgeequ_equilibration, setup_fn, dgeequ_teardown), \
    cmocka_unit_test_setup_teardown(test_dgeequ_identity, setup_fn, dgeequ_teardown), \
    cmocka_unit_test_setup_teardown(test_dgeequ_zero_row, setup_fn, dgeequ_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Square matrices */
        DGEEQU_TESTS(setup_2x2),
        DGEEQU_TESTS(setup_3x3),
        DGEEQU_TESTS(setup_5x5),
        DGEEQU_TESTS(setup_10x10),

        /* Rectangular M > N */
        DGEEQU_TESTS(setup_5x3),
        DGEEQU_TESTS(setup_7x5),
        DGEEQU_TESTS(setup_12x10),

        /* Rectangular M < N */
        DGEEQU_TESTS(setup_3x5),
        DGEEQU_TESTS(setup_5x7),
        DGEEQU_TESTS(setup_10x12),
    };

    return cmocka_run_group_tests_name("dgeequ", tests, NULL, NULL);
}
