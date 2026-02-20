/**
 * @file test_sgtsv.c
 * @brief CMocka test suite for sgtsv (tridiagonal system solver).
 *
 * Tests the combined factor+solve routine sgtsv using LAPACK's
 * verification methodology with normalized residuals.
 *
 * Verification: sgtt02 computes ||B - A*X|| / (||A|| * ||X|| * eps)
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include "test_rng.h"
#include "verify.h"

/* Routine under test */
extern void sgtsv(const int n, const int nrhs,
                  f32 * const restrict DL, f32 * const restrict D,
                  f32 * const restrict DU, f32 * const restrict B,
                  const int ldb, int *info);

/* Utilities */
extern f32 slamch(const char *cmach);
extern void slagtm(const char *trans, const int n, const int nrhs,
                   const f32 alpha,
                   const f32 * const restrict DL,
                   const f32 * const restrict D,
                   const f32 * const restrict DU,
                   const f32 * const restrict X, const int ldx,
                   const f32 beta,
                   f32 * const restrict B, const int ldb);

/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    int n, nrhs;
    int ldb;
    f32 *DL;       /* Sub-diagonal (will be overwritten by sgtsv) */
    f32 *D;        /* Diagonal (will be overwritten by sgtsv) */
    f32 *DU;       /* Super-diagonal (will be overwritten by sgtsv) */
    f32 *DL_copy;  /* Original sub-diagonal for verification */
    f32 *D_copy;   /* Original diagonal for verification */
    f32 *DU_copy;  /* Original super-diagonal for verification */
    f32 *XACT;     /* Exact solution */
    f32 *B;        /* Right-hand side / computed solution */
    f32 *B_copy;   /* Copy of RHS for verification */
    uint64_t seed;    /* RNG seed */
    uint64_t rng_state[4]; /* RNG state */
} dgtsv_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 2718;

/**
 * Generate a tridiagonal matrix for testing.
 */
static void generate_gt_matrix(int n, int imat, f32 *DL, f32 *D, f32 *DU,
                                uint64_t state[static 4])
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    int i;

    if (n <= 0) return;

    slatb4("SGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate random tridiagonal - diagonally dominant for stability */
    for (i = 0; i < n; i++) {
        D[i] = 4.0f + rng_uniform_f32(state);
    }
    for (i = 0; i < n - 1; i++) {
        DL[i] = rng_uniform_f32(state) - 0.5f;
        DU[i] = rng_uniform_f32(state) - 0.5f;
    }

    /* Apply modifications based on type */
    if (imat == 8 && n > 0) {
        D[0] = 0.0f;
    } else if (imat == 9 && n > 0) {
        D[n - 1] = 0.0f;
    } else if (imat == 10 && n > 2) {
        int mid = n / 2;
        D[mid] = 0.0f;
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
static int dgtsv_setup(void **state, int n, int nrhs)
{
    dgtsv_fixture_t *fix = malloc(sizeof(dgtsv_fixture_t));
    assert_non_null(fix);

    int m = (n > 1) ? n - 1 : 0;
    int ldb = (n > 1) ? n : 1;

    fix->n = n;
    fix->nrhs = nrhs;
    fix->ldb = ldb;
    fix->seed = g_seed++;
    rng_seed(fix->rng_state, fix->seed);

    fix->DL = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->D = malloc(n * sizeof(f32));
    fix->DU = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->DL_copy = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->D_copy = malloc(n * sizeof(f32));
    fix->DU_copy = malloc((m > 0 ? m : 1) * sizeof(f32));
    fix->XACT = malloc(ldb * nrhs * sizeof(f32));
    fix->B = malloc(ldb * nrhs * sizeof(f32));
    fix->B_copy = malloc(ldb * nrhs * sizeof(f32));

    assert_non_null(fix->DL);
    assert_non_null(fix->D);
    assert_non_null(fix->DU);
    assert_non_null(fix->DL_copy);
    assert_non_null(fix->D_copy);
    assert_non_null(fix->DU_copy);
    assert_non_null(fix->XACT);
    assert_non_null(fix->B);
    assert_non_null(fix->B_copy);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgtsv_teardown(void **state)
{
    dgtsv_fixture_t *fix = *state;
    if (fix) {
        free(fix->DL);
        free(fix->D);
        free(fix->DU);
        free(fix->DL_copy);
        free(fix->D_copy);
        free(fix->DU_copy);
        free(fix->XACT);
        free(fix->B);
        free(fix->B_copy);
        free(fix);
    }
    return 0;
}

/* Size-specific setup wrappers: sizes {1,2,3,5,10,50} x nrhs {1,2,15} */
static int setup_n1_nrhs1(void **state) { return dgtsv_setup(state, 1, 1); }
static int setup_n2_nrhs1(void **state) { return dgtsv_setup(state, 2, 1); }
static int setup_n3_nrhs1(void **state) { return dgtsv_setup(state, 3, 1); }
static int setup_n5_nrhs1(void **state) { return dgtsv_setup(state, 5, 1); }
static int setup_n10_nrhs1(void **state) { return dgtsv_setup(state, 10, 1); }
static int setup_n50_nrhs1(void **state) { return dgtsv_setup(state, 50, 1); }
static int setup_n1_nrhs2(void **state) { return dgtsv_setup(state, 1, 2); }
static int setup_n2_nrhs2(void **state) { return dgtsv_setup(state, 2, 2); }
static int setup_n3_nrhs2(void **state) { return dgtsv_setup(state, 3, 2); }
static int setup_n5_nrhs2(void **state) { return dgtsv_setup(state, 5, 2); }
static int setup_n10_nrhs2(void **state) { return dgtsv_setup(state, 10, 2); }
static int setup_n50_nrhs2(void **state) { return dgtsv_setup(state, 50, 2); }
static int setup_n1_nrhs15(void **state) { return dgtsv_setup(state, 1, 15); }
static int setup_n2_nrhs15(void **state) { return dgtsv_setup(state, 2, 15); }
static int setup_n3_nrhs15(void **state) { return dgtsv_setup(state, 3, 15); }
static int setup_n5_nrhs15(void **state) { return dgtsv_setup(state, 5, 15); }
static int setup_n10_nrhs15(void **state) { return dgtsv_setup(state, 10, 15); }
static int setup_n50_nrhs15(void **state) { return dgtsv_setup(state, 50, 15); }

/**
 * Core test logic: generate matrix, solve, verify.
 * Returns residual for the caller to assert on.
 * Returns -1.0 if the matrix is singular (info > 0).
 */
static f32 run_dgtsv_test(dgtsv_fixture_t *fix, int imat)
{
    int info;
    int n = fix->n;
    int nrhs = fix->nrhs;
    int m = (n > 1) ? n - 1 : 0;
    int ldb = fix->ldb;
    int i, j;

    /* Generate test matrix */
    generate_gt_matrix(n, imat, fix->DL, fix->D, fix->DU, fix->rng_state);

    /* Save original matrix */
    memcpy(fix->DL_copy, fix->DL, (m > 0 ? m : 1) * sizeof(f32));
    memcpy(fix->D_copy, fix->D, n * sizeof(f32));
    memcpy(fix->DU_copy, fix->DU, (m > 0 ? m : 1) * sizeof(f32));

    /* Generate random solution XACT */
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->XACT[i + j * ldb] = rng_uniform_symmetric_f32(fix->rng_state);
        }
    }

    /* Compute B = A * XACT */
    for (j = 0; j < nrhs; j++) {
        for (i = 0; i < n; i++) {
            fix->B[i + j * ldb] = 0.0f;
        }
    }
    slagtm("N", n, nrhs, 1.0f, fix->DL_copy, fix->D_copy, fix->DU_copy,
           fix->XACT, ldb, 0.0f, fix->B, ldb);

    /* Save B for residual check */
    memcpy(fix->B_copy, fix->B, ldb * nrhs * sizeof(f32));

    /* Solve A*X = B (overwrites D, DL, DU, B) */
    sgtsv(n, nrhs, fix->DL, fix->D, fix->DU, fix->B, ldb, &info);

    /* info < 0 means illegal argument */
    assert_true(info >= 0);

    /* Singular matrix - return special value */
    if (info > 0) {
        return -1.0f;
    }

    /* Verify solution using sgtt02: ||B_copy - A*X|| / (||A|| * ||X|| * eps) */
    f32 resid;
    sgtt02("N", n, nrhs, fix->DL_copy, fix->D_copy, fix->DU_copy,
           fix->B, ldb, fix->B_copy, ldb, &resid);

    return resid;
}

/*
 * Test well-conditioned types (1-6).
 */
static void test_dgtsv_wellcond(void **state)
{
    dgtsv_fixture_t *fix = *state;

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f32 resid = run_dgtsv_test(fix, imat);
        assert_true(resid >= 0.0f); /* not singular */
        assert_residual_ok(resid);
    }
}

/*
 * Test random matrix types (7, 11, 12) - non-singular.
 * Only run for n >= 2.
 */
static void test_dgtsv_random(void **state)
{
    dgtsv_fixture_t *fix = *state;

    if (fix->n < 2) {
        skip();
    }

    int random_types[] = {7, 11, 12};
    for (int k = 0; k < 3; k++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f32 resid = run_dgtsv_test(fix, random_types[k]);
        if (resid < 0.0f) {
            /* Singular - acceptable for these types */
            continue;
        }
        assert_residual_ok(resid);
    }
}

/*
 * Macro to generate test entries for a given setup.
 */
#define DGTSV_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgtsv_wellcond, setup_fn, dgtsv_teardown), \
    cmocka_unit_test_setup_teardown(test_dgtsv_random, setup_fn, dgtsv_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* nrhs = 1 */
        DGTSV_TESTS(setup_n1_nrhs1),
        DGTSV_TESTS(setup_n2_nrhs1),
        DGTSV_TESTS(setup_n3_nrhs1),
        DGTSV_TESTS(setup_n5_nrhs1),
        DGTSV_TESTS(setup_n10_nrhs1),
        DGTSV_TESTS(setup_n50_nrhs1),

        /* nrhs = 2 */
        DGTSV_TESTS(setup_n1_nrhs2),
        DGTSV_TESTS(setup_n2_nrhs2),
        DGTSV_TESTS(setup_n3_nrhs2),
        DGTSV_TESTS(setup_n5_nrhs2),
        DGTSV_TESTS(setup_n10_nrhs2),
        DGTSV_TESTS(setup_n50_nrhs2),

        /* nrhs = 15 */
        DGTSV_TESTS(setup_n1_nrhs15),
        DGTSV_TESTS(setup_n2_nrhs15),
        DGTSV_TESTS(setup_n3_nrhs15),
        DGTSV_TESTS(setup_n5_nrhs15),
        DGTSV_TESTS(setup_n10_nrhs15),
        DGTSV_TESTS(setup_n50_nrhs15),
    };

    return cmocka_run_group_tests_name("dgtsv", tests, NULL, NULL);
}
