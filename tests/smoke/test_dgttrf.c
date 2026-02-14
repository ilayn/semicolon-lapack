/**
 * @file test_dgttrf.c
 * @brief CMocka test suite for dgttrf (tridiagonal LU factorization with partial pivoting).
 *
 * Tests the LU factorization routine dgttrf using LAPACK's
 * verification methodology with normalized residuals.
 *
 * Verification: dgtt01 computes ||L*U - A|| / (||A|| * eps)
 *
 * Matrix types tested (12 types from dlatb4 "DGT"):
 *   1. Diagonal
 *   2. Tridiagonal, well-conditioned (cond=2)
 *   3. Ill-conditioned (cond ~ 3e7)
 *   4. Very ill-conditioned (cond ~ 9e15)
 *   5. Scaled near underflow
 *   6. Scaled near overflow
 *   7-12. Random matrices with various properties
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include "testutils/test_rng.h"
#include "testutils/verify.h"

/* Routine under test */
extern void dgttrf(const int n, f64 * const restrict DL,
                   f64 * const restrict D, f64 * const restrict DU,
                   f64 * const restrict DU2, int * const restrict ipiv,
                   int *info);

/* Utilities */
extern f64 dlamch(const char *cmach);

/*
 * Test fixture: holds all allocated memory for a single test case.
 * CMocka passes this between setup -> test -> teardown.
 */
typedef struct {
    int n;
    f64 *DL;      /* Original sub-diagonal */
    f64 *D;       /* Original diagonal */
    f64 *DU;      /* Original super-diagonal */
    f64 *DLF;     /* Factored sub-diagonal */
    f64 *DF;      /* Factored diagonal */
    f64 *DUF;     /* Factored super-diagonal */
    f64 *DU2;     /* Second super-diagonal from factorization */
    int *ipiv;       /* Pivot indices */
    f64 *work;    /* Workspace for dgtt01 */
    uint64_t seed;   /* RNG seed */
    uint64_t rng_state[4]; /* RNG state */
} dgttrf_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 1988;

/**
 * Generate a tridiagonal matrix for testing.
 *
 * For types 1-6: Use dlatms with controlled singular values.
 * For types 7-12: Generate random tridiagonal directly.
 */
static void generate_gt_matrix(int n, int imat, f64 *DL, f64 *D, f64 *DU,
                                uint64_t state[static 4])
{
    char type, dist;
    int kl, ku, mode;
    f64 anorm, cndnum;
    int info;
    int i;

    if (n <= 0) return;

    dlatb4("DGT", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    if (imat >= 1 && imat <= 6) {
        /* Types 1-6: Use dlatms to generate matrix with controlled condition */
        /* Generate in band storage: 3 rows (sub, diag, super) */
        int lda = 3;
        f64 *AB = calloc(lda * n, sizeof(f64));
        f64 *d_sing = malloc(n * sizeof(f64));
        f64 *work = malloc(3 * n * sizeof(f64));

        assert_non_null(AB);
        assert_non_null(d_sing);
        assert_non_null(work);

        /* Generate band matrix with KL=1, KU=1 */
        dlatms(n, n, &dist, &type, d_sing, mode, cndnum, anorm,
               kl, ku, "N", AB, lda, work, &info, state);

        if (info != 0) {
            /* Fall back to simple generation */
            for (i = 0; i < n; i++) {
                D[i] = 2.0 * anorm;
            }
            for (i = 0; i < n - 1; i++) {
                DL[i] = -anorm * 0.5;
                DU[i] = -anorm * 0.5;
            }
        } else {
            /* Extract tridiagonal from band storage */
            /* Band storage: row 0 = super-diagonal, row 1 = diagonal, row 2 = sub-diagonal */
            for (i = 0; i < n; i++) {
                D[i] = AB[1 + i * lda];  /* Diagonal */
            }
            for (i = 0; i < n - 1; i++) {
                DU[i] = AB[0 + (i + 1) * lda];  /* Super-diagonal */
                DL[i] = AB[2 + i * lda];        /* Sub-diagonal */
            }
        }

        free(AB);
        free(d_sing);
        free(work);

    } else {
        /* Types 7-12: Random generation */
        for (i = 0; i < n; i++) {
            D[i] = rng_uniform_symmetric(state);
        }
        for (i = 0; i < n - 1; i++) {
            DL[i] = rng_uniform_symmetric(state);
            DU[i] = rng_uniform_symmetric(state);
        }

        /* Apply modifications based on type */
        if (imat == 8 && n > 0) {
            /* Zero first diagonal element */
            D[0] = 0.0;
        } else if (imat == 9 && n > 0) {
            /* Zero last diagonal element */
            D[n - 1] = 0.0;
        } else if (imat == 10 && n > 2) {
            /* Zero middle n/2 diagonal elements */
            int mid = n / 2;
            for (i = mid / 2; i < mid / 2 + mid; i++) {
                if (i < n) D[i] = 0.0;
            }
        }

        /* Scale if needed */
        if (anorm != 1.0) {
            for (i = 0; i < n; i++) {
                D[i] *= anorm;
            }
            for (i = 0; i < n - 1; i++) {
                DL[i] *= anorm;
                DU[i] *= anorm;
            }
        }
    }
}

/**
 * Setup fixture: allocate memory for given dimension.
 * Called before each test function.
 */
static int dgttrf_setup(void **state, int n)
{
    dgttrf_fixture_t *fix = malloc(sizeof(dgttrf_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;
    rng_seed(fix->rng_state, fix->seed);

    int m = (n > 1) ? n - 1 : 0;
    int m_alloc = (m > 0) ? m : 1;        /* At least 1 for malloc */
    int du2_size = (n > 2) ? n - 2 : 1;
    int work_size = (n > 0) ? n * n : 1;  /* ldwork = n for dgtt01 */

    fix->DL = malloc(m_alloc * sizeof(f64));
    fix->D = malloc((n > 0 ? n : 1) * sizeof(f64));
    fix->DU = malloc(m_alloc * sizeof(f64));
    fix->DLF = malloc(m_alloc * sizeof(f64));
    fix->DF = malloc((n > 0 ? n : 1) * sizeof(f64));
    fix->DUF = malloc(m_alloc * sizeof(f64));
    fix->DU2 = malloc(du2_size * sizeof(f64));
    fix->ipiv = malloc((n > 0 ? n : 1) * sizeof(int));
    fix->work = malloc(work_size * sizeof(f64));

    assert_non_null(fix->DL);
    assert_non_null(fix->D);
    assert_non_null(fix->DU);
    assert_non_null(fix->DLF);
    assert_non_null(fix->DF);
    assert_non_null(fix->DUF);
    assert_non_null(fix->DU2);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->work);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 * Called after each test function.
 */
static int dgttrf_teardown(void **state)
{
    dgttrf_fixture_t *fix = *state;
    if (fix) {
        free(fix->DL);
        free(fix->D);
        free(fix->DU);
        free(fix->DLF);
        free(fix->DF);
        free(fix->DUF);
        free(fix->DU2);
        free(fix->ipiv);
        free(fix->work);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_0(void **state) { return dgttrf_setup(state, 0); }
static int setup_1(void **state) { return dgttrf_setup(state, 1); }
static int setup_2(void **state) { return dgttrf_setup(state, 2); }
static int setup_3(void **state) { return dgttrf_setup(state, 3); }
static int setup_5(void **state) { return dgttrf_setup(state, 5); }
static int setup_10(void **state) { return dgttrf_setup(state, 10); }
static int setup_50(void **state) { return dgttrf_setup(state, 50); }

/**
 * Core test logic: generate matrix, factorize, verify.
 * Returns residual for the caller to assert on.
 */
static f64 run_dgttrf_test(dgttrf_fixture_t *fix, int imat)
{
    int info;
    int n = fix->n;
    int m = (n > 1) ? n - 1 : 0;

    /* Generate test matrix */
    generate_gt_matrix(n, imat, fix->DL, fix->D, fix->DU, fix->rng_state);

    /* Copy to factored arrays */
    memcpy(fix->DLF, fix->DL, m * sizeof(f64));
    memcpy(fix->DF, fix->D, n * sizeof(f64));
    memcpy(fix->DUF, fix->DU, m * sizeof(f64));

    /* Factorize */
    dgttrf(n, fix->DLF, fix->DF, fix->DUF, fix->DU2, fix->ipiv, &info);

    /* For singular matrix types (8-10), info > 0 is expected */
    assert_true(info >= 0);

    /* Verify factorization using dgtt01 */
    f64 resid;
    dgtt01(n, fix->DL, fix->D, fix->DU, fix->DLF, fix->DF, fix->DUF,
           fix->DU2, fix->ipiv, fix->work, n, &resid);

    return resid;
}

/**
 * Test dgttrf for structured matrix types (1-6).
 * These use dlatms for controlled condition numbers.
 */
static void test_dgttrf_structured(void **state)
{
    dgttrf_fixture_t *fix = *state;

    if (fix->n == 0) {
        skip_test("n=0: nothing to verify for structured types");
    }

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f64 resid = run_dgttrf_test(fix, imat);
        assert_residual_ok(resid);
    }
}

/**
 * Test dgttrf for random matrix types (7-12).
 * These use direct random generation and require n >= 2.
 */
static void test_dgttrf_random(void **state)
{
    dgttrf_fixture_t *fix = *state;

    if (fix->n < 2) {
        skip_test("random types require n >= 2");
    }

    for (int imat = 7; imat <= 12; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        f64 resid = run_dgttrf_test(fix, imat);
        assert_residual_ok(resid);
    }
}

/*
 * Macro to generate test entries for a given size.
 * Creates 2 test cases: structured (types 1-6), random (types 7-12).
 */
#define DGTTRF_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgttrf_structured, setup_fn, dgttrf_teardown), \
    cmocka_unit_test_setup_teardown(test_dgttrf_random, setup_fn, dgttrf_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DGTTRF_TESTS(setup_0),
        DGTTRF_TESTS(setup_1),
        DGTTRF_TESTS(setup_2),
        DGTTRF_TESTS(setup_3),
        DGTTRF_TESTS(setup_5),
        DGTTRF_TESTS(setup_10),
        DGTTRF_TESTS(setup_50),
    };

    return cmocka_run_group_tests_name("dgttrf", tests, NULL, NULL);
}
