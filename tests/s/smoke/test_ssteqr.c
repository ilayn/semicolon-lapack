/**
 * @file test_ssteqr.c
 * @brief CMocka test suite for ssteqr (eigenvalues/eigenvectors of symmetric tridiagonal matrix).
 *
 * Tests the QL/QR iteration routine ssteqr using LAPACK's
 * verification methodology with normalized residuals.
 *
 * Verification:
 *   sstt21 checks |T - Z*D*Z'| / (|T|*n*ulp) and |I - Z*Z'| / (n*ulp)
 *   sstech verifies eigenvalues via Sturm sequence
 *
 * Matrix types tested (symmetric tridiagonal, generated directly):
 *   1. Zero matrix
 *   2. Identity (D=1, E=0)
 *   3. 1-2-1 tridiagonal (Toeplitz: D[i]=2, E[i]=1)
 *   4. Wilkinson-like (D[i]=|i-n/2|, E[i]=1)
 *   5. Random diagonal, zero off-diagonal
 *   6. Random symmetric tridiagonal (well-conditioned)
 *   7. Graded diagonal (D[i]=2^(-i), E[i]=1)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
/* Routine under test */
/* Verification routines */
/* Utilities */
/*
 * Test fixture: holds all allocated memory for a single test case.
 * CMocka passes this between setup -> test -> teardown.
 */
typedef struct {
    INT n;
    f32* AD;         /* original diagonal (preserved) */
    f32* AE;         /* original off-diagonal (preserved) */
    f32* D;          /* diagonal for ssteqr (overwritten) */
    f32* E;          /* off-diagonal for ssteqr (overwritten) */
    f32* Z;          /* eigenvectors (n x n) */
    f32* work;       /* workspace for ssteqr: max(1, 2*(n-1)) */
    f32* work_dstt21;/* workspace for sstt21: n*(n+1) */
    f32* result;     /* sstt21 results (2 elements) */
    uint64_t seed;
    uint64_t rng_state[4];
} dsteqr_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 2024;

/**
 * Setup fixture: allocate memory for given dimension.
 * Called before each test function.
 */
static int dsteqr_setup(void** state, INT n)
{
    dsteqr_fixture_t* fix = malloc(sizeof(dsteqr_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    INT n_alloc = (n > 0) ? n : 1;
    INT e_alloc = (n > 1) ? n - 1 : 1;
    INT work_steqr = (n > 2) ? 2 * (n - 1) : 1;
    INT work_stt21 = (n > 0) ? n * (n + 1) : 1;

    fix->AD = malloc(n_alloc * sizeof(f32));
    fix->AE = malloc(e_alloc * sizeof(f32));
    fix->D = malloc(n_alloc * sizeof(f32));
    fix->E = malloc(e_alloc * sizeof(f32));
    fix->Z = malloc(n_alloc * n_alloc * sizeof(f32));
    fix->work = malloc(work_steqr * sizeof(f32));
    fix->work_dstt21 = malloc(work_stt21 * sizeof(f32));
    fix->result = malloc(2 * sizeof(f32));

    assert_non_null(fix->AD);
    assert_non_null(fix->AE);
    assert_non_null(fix->D);
    assert_non_null(fix->E);
    assert_non_null(fix->Z);
    assert_non_null(fix->work);
    assert_non_null(fix->work_dstt21);
    assert_non_null(fix->result);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 * Called after each test function.
 */
static int dsteqr_teardown(void** state)
{
    dsteqr_fixture_t* fix = *state;
    if (fix) {
        free(fix->AD);
        free(fix->AE);
        free(fix->D);
        free(fix->E);
        free(fix->Z);
        free(fix->work);
        free(fix->work_dstt21);
        free(fix->result);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_1(void** state) { return dsteqr_setup(state, 1); }
static int setup_2(void** state) { return dsteqr_setup(state, 2); }
static int setup_5(void** state) { return dsteqr_setup(state, 5); }
static int setup_10(void** state) { return dsteqr_setup(state, 10); }
static int setup_20(void** state) { return dsteqr_setup(state, 20); }
static int setup_50(void** state) { return dsteqr_setup(state, 50); }

/**
 * Generate symmetric tridiagonal test matrix.
 *
 * @param n     Matrix dimension
 * @param imat  Matrix type (1-7)
 * @param D     Diagonal array (length n)
 * @param E     Off-diagonal array (length n-1)
 * @param seed  RNG seed (used for types 5-6)
 */
static void generate_steqr_matrix(INT n, INT imat, f32* D, f32* E,
                                   uint64_t state[static 4])
{
    INT i;

    switch (imat) {
    case 1:
        /* Zero matrix */
        for (i = 0; i < n; i++) D[i] = 0.0f;
        for (i = 0; i < n - 1; i++) E[i] = 0.0f;
        break;

    case 2:
        /* Identity: D=1, E=0 */
        for (i = 0; i < n; i++) D[i] = 1.0f;
        for (i = 0; i < n - 1; i++) E[i] = 0.0f;
        break;

    case 3:
        /* 1-2-1 Toeplitz: D[i]=2, E[i]=1 */
        for (i = 0; i < n; i++) D[i] = 2.0f;
        for (i = 0; i < n - 1; i++) E[i] = 1.0f;
        break;

    case 4:
        /* Wilkinson-like: D[i]=|i - n/2|, E[i]=1 */
        for (i = 0; i < n; i++) D[i] = fabsf((f32)i - (f32)(n / 2));
        for (i = 0; i < n - 1; i++) E[i] = 1.0f;
        break;

    case 5:
        /* Random diagonal, zero off-diagonal */
        for (i = 0; i < n; i++) D[i] = rng_uniform_symmetric_f32(state);
        for (i = 0; i < n - 1; i++) E[i] = 0.0f;
        break;

    case 6:
        /* Random symmetric tridiagonal */
        for (i = 0; i < n; i++) D[i] = rng_uniform_symmetric_f32(state);
        for (i = 0; i < n - 1; i++) E[i] = rng_uniform_symmetric_f32(state);
        break;

    case 7:
        /* Graded diagonal: D[i] = 2^(-i), E[i] = 1 */
        for (i = 0; i < n; i++) D[i] = powf(2.0f, -(f32)i);
        for (i = 0; i < n - 1; i++) E[i] = 1.0f;
        break;

    default:
        /* Fallback: identity */
        for (i = 0; i < n; i++) D[i] = 1.0f;
        for (i = 0; i < n - 1; i++) E[i] = 0.0f;
        break;
    }
}

/**
 * Test COMPZ='I': compute eigenvalues and eigenvectors.
 * Verifies:
 *   result[0] = |T - Z*D*Z'| / (|T|*n*ulp)
 *   result[1] = |I - Z*Z'| / (n*ulp)
 */
static void test_compz_I(void** state)
{
    dsteqr_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;

    for (INT imat = 1; imat <= 7; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_steqr_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Save original for verification */
        memcpy(fix->AD, fix->D, n * sizeof(f32));
        if (n > 1) {
            memcpy(fix->AE, fix->E, (n - 1) * sizeof(f32));
        }

        /* Compute eigenvalues and eigenvectors */
        ssteqr("I", n, fix->D, fix->E, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Verify decomposition via sstt21:
         * kband=0 because SD contains eigenvalues (diagonal),
         * SE is not referenced when kband=0, pass NULL */
        sstt21(n, 0, fix->AD, fix->AE, fix->D, NULL,
               fix->Z, n, fix->work_dstt21, fix->result);

        assert_residual_ok(fix->result[0]);  /* decomposition residual */
        assert_residual_ok(fix->result[1]);  /* orthogonality */
    }
}

/**
 * Test COMPZ='N': eigenvalues only.
 * Compares eigenvalues with those from COMPZ='I'.
 */
static void test_compz_N(void** state)
{
    dsteqr_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;
    f32 ulp = slamch("E");

    /* Allocate temporary arrays for the COMPZ='I' computation */
    INT n_alloc = (n > 0) ? n : 1;
    INT e_alloc = (n > 1) ? n - 1 : 1;
    f32* D_I = malloc(n_alloc * sizeof(f32));
    f32* E_I = malloc(e_alloc * sizeof(f32));
    assert_non_null(D_I);
    assert_non_null(E_I);

    for (INT imat = 1; imat <= 7; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_steqr_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Copy for COMPZ='I' run */
        memcpy(D_I, fix->D, n * sizeof(f32));
        if (n > 1) {
            memcpy(E_I, fix->E, (n - 1) * sizeof(f32));
        }

        /* Compute with COMPZ='I' for reference eigenvalues */
        ssteqr("I", n, D_I, E_I, fix->Z, n, fix->work, &info);
        assert_info_success(info);

        /* Regenerate for COMPZ='N' run (E is destroyed by ssteqr) */
        rng_seed(fix->rng_state, fix->seed);
        generate_steqr_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Compute eigenvalues only */
        ssteqr("N", n, fix->D, fix->E, fix->Z, 1, fix->work, &info);
        assert_info_success(info);

        /* Compare eigenvalues: both should be sorted, so direct comparison.
         * Compute max|D_N[i] - D_I[i]| / (n * max|D_I| * ulp) */
        f32 max_eig = 0.0f;
        for (INT i = 0; i < n; i++) {
            f32 a = fabsf(D_I[i]);
            if (a > max_eig) max_eig = a;
        }

        f32 max_diff = 0.0f;
        for (INT i = 0; i < n; i++) {
            f32 d = fabsf(fix->D[i] - D_I[i]);
            if (d > max_diff) max_diff = d;
        }

        f32 denom = (f32)n * ulp;
        if (max_eig > 0.0f) denom *= max_eig;
        f32 resid = (denom > 0.0f) ? max_diff / denom : 0.0f;

        assert_residual_ok(resid);
    }

    free(D_I);
    free(E_I);
}

/**
 * Test Sturm sequence verification via sstech.
 * After computing eigenvalues, checks that they satisfy Sturm sequence counts.
 */
static void test_sturm(void** state)
{
    dsteqr_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;

    /* Allocate workspace for sstech: length n */
    INT n_alloc = (n > 0) ? n : 1;
    f32* stech_work = malloc(n_alloc * sizeof(f32));
    assert_non_null(stech_work);

    for (INT imat = 1; imat <= 7; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_steqr_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Save original diagonal and off-diagonal */
        memcpy(fix->AD, fix->D, n * sizeof(f32));
        if (n > 1) {
            memcpy(fix->AE, fix->E, (n - 1) * sizeof(f32));
        }

        /* Compute eigenvalues */
        ssteqr("N", n, fix->D, fix->E, fix->Z, 1, fix->work, &info);
        assert_info_success(info);

        /* Sturm sequence check:
         * sstech verifies eigenvalues using the original tridiagonal.
         * A = diagonal, B = off-diagonal, eig = computed eigenvalues */
        sstech(n, fix->AD, fix->AE, fix->D, THRESH, stech_work, &info);
        assert_info_success(info);
    }

    free(stech_work);
}

/*
 * Macro to generate test entries for a given size.
 * Creates 3 test cases: compz_I, compz_N, sturm.
 */
#define DSTEQR_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_compz_I, setup_fn, dsteqr_teardown), \
    cmocka_unit_test_setup_teardown(test_compz_N, setup_fn, dsteqr_teardown), \
    cmocka_unit_test_setup_teardown(test_sturm, setup_fn, dsteqr_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSTEQR_TESTS(setup_1),
        DSTEQR_TESTS(setup_2),
        DSTEQR_TESTS(setup_5),
        DSTEQR_TESTS(setup_10),
        DSTEQR_TESTS(setup_20),
        DSTEQR_TESTS(setup_50),
    };

    return cmocka_run_group_tests_name("dsteqr", tests, NULL, NULL);
}
