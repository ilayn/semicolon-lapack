/**
 * @file test_sgttrs.c
 * @brief CMocka test suite for sgttrs (solve using tridiagonal LU factorization).
 *
 * Tests sgttrf + sgttrs combination using LAPACK's verification methodology.
 *
 * Verification: sgtt02 computes ||B - op(A)*X|| / (||op(A)|| * ||X|| * eps)
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include "test_rng.h"
#include "verify.h"

/* Routines under test */
/* Utilities */
/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    INT n, nrhs;
    INT ldb;
    f32 *DL;      /* Original sub-diagonal */
    f32 *D;       /* Original diagonal */
    f32 *DU;      /* Original super-diagonal */
    f32 *DLF;     /* Factored sub-diagonal */
    f32 *DF;      /* Factored diagonal */
    f32 *DUF;     /* Factored super-diagonal */
    f32 *DU2;     /* Second super-diagonal from factorization */
    INT* ipiv;       /* Pivot indices */
    f32 *XACT;    /* Exact solution */
    f32 *B;       /* Right-hand side / computed solution */
    f32 *B_copy;  /* Copy of RHS for verification */
    uint64_t seed;   /* RNG seed */
    uint64_t rng_state[4]; /* RNG state */
} dgttrs_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 3141;

/**
 * Generate a diagonally dominant tridiagonal matrix for testing.
 */
static void generate_gt_matrix(INT n, INT imat, f32 *DL, f32 *D, f32 *DU,
                                uint64_t state[static 4])
{
    char type, dist;
    INT kl, ku, mode;
    f32 anorm, cndnum;
    INT i;

    if (n <= 0) return;

    slatb4("SGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate diagonally dominant matrix for stability */
    for (i = 0; i < n; i++) {
        D[i] = 4.0f + rng_uniform_f32(state);
    }
    for (i = 0; i < n - 1; i++) {
        DL[i] = rng_uniform_f32(state) - 0.5f;
        DU[i] = rng_uniform_f32(state) - 0.5f;
    }

    /* Scale if needed */
    if (anorm != 1.0f) {
        for (i = 0; i < n; i++) {
            D[i] *= anorm;
        }
        for (i = 0; i < n - 1; i++) {
            DL[i] *= anorm;
            DU[i] *= anorm;
        }
    }
}

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dgttrs_setup(void **state, INT n, INT nrhs)
{
    dgttrs_fixture_t *fix = malloc(sizeof(dgttrs_fixture_t));
    assert_non_null(fix);

    INT m = (n > 1) ? n - 1 : 0;
    INT ldb = (n > 1) ? n : 1;

    fix->n = n;
    fix->nrhs = nrhs;
    fix->ldb = ldb;
    fix->seed = g_seed++;
    rng_seed(fix->rng_state, fix->seed);

    fix->DL = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->D = malloc(n * sizeof(f32));
    fix->DU = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->DLF = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->DF = malloc(n * sizeof(f32));
    fix->DUF = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->DU2 = malloc((n > 2 ? n - 2 : 1) * sizeof(f32));
    fix->ipiv = malloc(n * sizeof(INT));
    fix->XACT = malloc(ldb * nrhs * sizeof(f32));
    fix->B = malloc(ldb * nrhs * sizeof(f32));
    fix->B_copy = malloc(ldb * nrhs * sizeof(f32));

    assert_non_null(fix->DL);
    assert_non_null(fix->D);
    assert_non_null(fix->DU);
    assert_non_null(fix->DLF);
    assert_non_null(fix->DF);
    assert_non_null(fix->DUF);
    assert_non_null(fix->DU2);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->XACT);
    assert_non_null(fix->B);
    assert_non_null(fix->B_copy);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgttrs_teardown(void **state)
{
    dgttrs_fixture_t *fix = *state;
    if (fix) {
        free(fix->DL);
        free(fix->D);
        free(fix->DU);
        free(fix->DLF);
        free(fix->DF);
        free(fix->DUF);
        free(fix->DU2);
        free(fix->ipiv);
        free(fix->XACT);
        free(fix->B);
        free(fix->B_copy);
        free(fix);
    }
    return 0;
}

/* Size-specific setup wrappers: sizes {1,2,3,5,10,50} x nrhs {1,2,15} */
static int setup_n1_nrhs1(void **state) { return dgttrs_setup(state, 1, 1); }
static int setup_n2_nrhs1(void **state) { return dgttrs_setup(state, 2, 1); }
static int setup_n3_nrhs1(void **state) { return dgttrs_setup(state, 3, 1); }
static int setup_n5_nrhs1(void **state) { return dgttrs_setup(state, 5, 1); }
static int setup_n10_nrhs1(void **state) { return dgttrs_setup(state, 10, 1); }
static int setup_n50_nrhs1(void **state) { return dgttrs_setup(state, 50, 1); }
static int setup_n1_nrhs2(void **state) { return dgttrs_setup(state, 1, 2); }
static int setup_n2_nrhs2(void **state) { return dgttrs_setup(state, 2, 2); }
static int setup_n3_nrhs2(void **state) { return dgttrs_setup(state, 3, 2); }
static int setup_n5_nrhs2(void **state) { return dgttrs_setup(state, 5, 2); }
static int setup_n10_nrhs2(void **state) { return dgttrs_setup(state, 10, 2); }
static int setup_n50_nrhs2(void **state) { return dgttrs_setup(state, 50, 2); }
static int setup_n1_nrhs15(void **state) { return dgttrs_setup(state, 1, 15); }
static int setup_n2_nrhs15(void **state) { return dgttrs_setup(state, 2, 15); }
static int setup_n3_nrhs15(void **state) { return dgttrs_setup(state, 3, 15); }
static int setup_n5_nrhs15(void **state) { return dgttrs_setup(state, 5, 15); }
static int setup_n10_nrhs15(void **state) { return dgttrs_setup(state, 10, 15); }
static int setup_n50_nrhs15(void **state) { return dgttrs_setup(state, 50, 15); }

/**
 * Core test logic: generate matrix, factorize, solve, verify.
 * Returns residual for the caller to assert on.
 */
static f32 run_dgttrs_test(dgttrs_fixture_t *fix, INT imat, const char* trans)
{
    INT info;
    INT n = fix->n;
    INT nrhs = fix->nrhs;
    INT m = (n > 1) ? n - 1 : 0;
    INT ldb = fix->ldb;
    INT i, j;

    /* Generate test matrix */
    generate_gt_matrix(n, imat, fix->DL, fix->D, fix->DU, fix->rng_state);

    /* Copy to factored arrays */
    memcpy(fix->DLF, fix->DL, (m > 0 ? m : 1) * sizeof(f32));
    memcpy(fix->DF, fix->D, n * sizeof(f32));
    memcpy(fix->DUF, fix->DU, (m > 0 ? m : 1) * sizeof(f32));

    /* Factor */
    sgttrf(n, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv, &info);
    assert_info_success(info);

    /* Generate random solution XACT */
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->XACT[i + j * ldb] = rng_uniform_symmetric_f32(fix->rng_state);
        }
    }

    /* Compute B = op(A) * XACT */
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->B[i + j * ldb] = 0.0f;
        }
    }
    slagtm(trans, n, nrhs, 1.0f, fix->DL, fix->D, fix->DU, fix->XACT, ldb, 0.0f, fix->B, ldb);

    /* Save B for residual check */
    memcpy(fix->B_copy, fix->B, ldb * nrhs * sizeof(f32));

    /* Solve op(A)*X = B */
    sgttrs(trans, n, nrhs, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv,
            fix->B, ldb, &info);
    assert_info_success(info);

    /* Verify solution using sgtt02 */
    f32 resid;
    sgtt02(trans, n, nrhs, fix->DL, fix->D, fix->DU, fix->B, ldb, fix->B_copy, ldb, &resid);

    return resid;
}

/*
 * Test well-conditioned types (1-6) with no transpose.
 */
static void test_dgttrs_notrans(void **state)
{
    dgttrs_fixture_t *fix = *state;

    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f32 resid = run_dgttrs_test(fix, imat, "N");
        assert_residual_ok(resid);
    }
}

/*
 * Test well-conditioned types (1-6) with transpose.
 */
static void test_dgttrs_trans(void **state)
{
    dgttrs_fixture_t *fix = *state;

    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f32 resid = run_dgttrs_test(fix, imat, "T");
        assert_residual_ok(resid);
    }
}

/*
 * Macro to generate test entries for a given setup.
 */
#define DGTTRS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgttrs_notrans, setup_fn, dgttrs_teardown), \
    cmocka_unit_test_setup_teardown(test_dgttrs_trans, setup_fn, dgttrs_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* nrhs = 1 */
        DGTTRS_TESTS(setup_n1_nrhs1),
        DGTTRS_TESTS(setup_n2_nrhs1),
        DGTTRS_TESTS(setup_n3_nrhs1),
        DGTTRS_TESTS(setup_n5_nrhs1),
        DGTTRS_TESTS(setup_n10_nrhs1),
        DGTTRS_TESTS(setup_n50_nrhs1),

        /* nrhs = 2 */
        DGTTRS_TESTS(setup_n1_nrhs2),
        DGTTRS_TESTS(setup_n2_nrhs2),
        DGTTRS_TESTS(setup_n3_nrhs2),
        DGTTRS_TESTS(setup_n5_nrhs2),
        DGTTRS_TESTS(setup_n10_nrhs2),
        DGTTRS_TESTS(setup_n50_nrhs2),

        /* nrhs = 15 */
        DGTTRS_TESTS(setup_n1_nrhs15),
        DGTTRS_TESTS(setup_n2_nrhs15),
        DGTTRS_TESTS(setup_n3_nrhs15),
        DGTTRS_TESTS(setup_n5_nrhs15),
        DGTTRS_TESTS(setup_n10_nrhs15),
        DGTTRS_TESTS(setup_n50_nrhs15),
    };

    return cmocka_run_group_tests_name("dgttrs", tests, NULL, NULL);
}
