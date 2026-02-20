/**
 * @file test_ssterf.c
 * @brief CMocka test suite for ssterf (eigenvalues of symmetric tridiagonal matrix).
 *
 * Tests the eigenvalue computation routine ssterf using the
 * Pal-Walker-Kahan QL/QR algorithm.
 *
 * Verification:
 *   1. sstech (Sturm count) verifies eigenvalues are correct to tolerance.
 *   2. Comparison with ssteqr eigenvalues: max|D1[i]-D2[i]|/(|max_eig|*n*ulp) < THRESH
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

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include "test_rng.h"

/* Routine under test */
extern void ssterf(const int n, f32* D, f32* E, int* info);

/* Reference eigenvalue computation */
extern void ssteqr(const char* compz, const int n, f32* D, f32* E,
                   f32* Z, const int ldz, f32* work, int* info);

/* Verification routine: Sturm count check */
extern void sstech(const int n, const f32* A, const f32* B,
                   const f32* eig, const f32 tol, f32* work, int* info);

/* Machine parameters */
extern f32 slamch(const char* cmach);

/*
 * Test fixture: holds all allocated memory for a single test case.
 * CMocka passes this between setup -> test -> teardown.
 */
typedef struct {
    int n;
    f32* D;      /* diagonal (input to ssterf, overwritten with eigenvalues) */
    f32* E;      /* off-diagonal (input to ssterf, destroyed) */
    f32* D_orig; /* saved copy of diagonal */
    f32* E_orig; /* saved copy of off-diagonal */
    f32* D2;     /* copy for ssteqr comparison */
    f32* E2;     /* copy for ssteqr comparison */
    f32* Z;      /* eigenvector workspace for ssteqr */
    f32* work;   /* workspace for ssteqr and sstech */
    uint64_t seed;  /* RNG seed */
    uint64_t rng_state[4]; /* RNG state */
} dsterf_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 2024;

/**
 * Generate a symmetric tridiagonal test matrix.
 *
 * @param n     Matrix dimension
 * @param imat  Matrix type (1-7)
 * @param D     Output diagonal array, dimension n
 * @param E     Output off-diagonal array, dimension n-1
 * @param seed  RNG seed (incremented after use)
 */
static void generate_st_matrix(int n, int imat, f32* D, f32* E,
                               uint64_t state[static 4])
{
    int i;

    if (n <= 0) return;

    switch (imat) {
        case 1:
            /* Zero matrix */
            for (i = 0; i < n; i++) D[i] = 0.0f;
            for (i = 0; i < n - 1; i++) E[i] = 0.0f;
            break;

        case 2:
            /* Identity (D=1, E=0) */
            for (i = 0; i < n; i++) D[i] = 1.0f;
            for (i = 0; i < n - 1; i++) E[i] = 0.0f;
            break;

        case 3:
            /* 1-2-1 Toeplitz tridiagonal: D[i]=2, E[i]=1 */
            /* Known eigenvalues: 2 - 2*cos(k*pi/(n+1)), k=1,...,n */
            for (i = 0; i < n; i++) D[i] = 2.0f;
            for (i = 0; i < n - 1; i++) E[i] = 1.0f;
            break;

        case 4:
            /* Wilkinson-like: D[i]=|i - n/2|, E[i]=1 */
            for (i = 0; i < n; i++) {
                D[i] = fabsf((f32)(i - n / 2));
            }
            for (i = 0; i < n - 1; i++) E[i] = 1.0f;
            break;

        case 5:
            /* Random diagonal, zero off-diagonal */
            /* Eigenvalues = diagonal entries */
            for (i = 0; i < n; i++) D[i] = rng_uniform_symmetric_f32(state) * 10.0f;
            for (i = 0; i < n - 1; i++) E[i] = 0.0f;
            break;

        case 6:
            /* Random symmetric tridiagonal (well-conditioned) */
            for (i = 0; i < n; i++) {
                D[i] = rng_uniform_symmetric_f32(state) * 5.0f + (f32)n;
            }
            for (i = 0; i < n - 1; i++) {
                E[i] = rng_uniform_symmetric_f32(state);
            }
            break;

        case 7:
            /* Graded diagonal: D[i]=2^(-i), E[i]=1 */
            /* Tests scaling behavior */
            for (i = 0; i < n; i++) {
                D[i] = powf(2.0f, -(f32)i);
            }
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
 * Simple insertion sort for f64 array (ascending order).
 * Used to ensure eigenvalue arrays are sorted for comparison.
 */
static void sort_ascending(int n, f32* x)
{
    int i, j;
    f32 tmp;
    for (i = 1; i < n; i++) {
        tmp = x[i];
        j = i - 1;
        while (j >= 0 && x[j] > tmp) {
            x[j + 1] = x[j];
            j--;
        }
        x[j + 1] = tmp;
    }
}

/**
 * Setup fixture: allocate memory for given dimension.
 * Called before each test function.
 */
static int dsterf_setup(void** state, int n)
{
    dsterf_fixture_t* fix = malloc(sizeof(dsterf_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;
    rng_seed(fix->rng_state, fix->seed);

    int n_alloc = (n > 0) ? n : 1;
    int e_alloc = (n > 1) ? n - 1 : 1;
    /* ssteqr work size: max(1, 2*(n-1)) */
    int work_size = (n > 1) ? 2 * (n - 1) : 1;
    /* sstech also needs n workspace, take max */
    if (n_alloc > work_size) work_size = n_alloc;

    fix->D = malloc(n_alloc * sizeof(f32));
    fix->E = malloc(e_alloc * sizeof(f32));
    fix->D_orig = malloc(n_alloc * sizeof(f32));
    fix->E_orig = malloc(e_alloc * sizeof(f32));
    fix->D2 = malloc(n_alloc * sizeof(f32));
    fix->E2 = malloc(e_alloc * sizeof(f32));
    fix->Z = malloc(n_alloc * n_alloc * sizeof(f32));
    fix->work = malloc(work_size * sizeof(f32));

    assert_non_null(fix->D);
    assert_non_null(fix->E);
    assert_non_null(fix->D_orig);
    assert_non_null(fix->E_orig);
    assert_non_null(fix->D2);
    assert_non_null(fix->E2);
    assert_non_null(fix->Z);
    assert_non_null(fix->work);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 * Called after each test function.
 */
static int dsterf_teardown(void** state)
{
    dsterf_fixture_t* fix = *state;
    if (fix) {
        free(fix->D);
        free(fix->E);
        free(fix->D_orig);
        free(fix->E_orig);
        free(fix->D2);
        free(fix->E2);
        free(fix->Z);
        free(fix->work);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_1(void** state) { return dsterf_setup(state, 1); }
static int setup_2(void** state) { return dsterf_setup(state, 2); }
static int setup_5(void** state) { return dsterf_setup(state, 5); }
static int setup_10(void** state) { return dsterf_setup(state, 10); }
static int setup_20(void** state) { return dsterf_setup(state, 20); }
static int setup_50(void** state) { return dsterf_setup(state, 50); }
static int setup_100(void** state) { return dsterf_setup(state, 100); }

/**
 * Core test logic: generate matrix, compute eigenvalues with ssterf,
 * verify with sstech and compare with ssteqr.
 */
static void run_dsterf_test(dsterf_fixture_t* fix, int imat)
{
    int info;
    int n = fix->n;
    f32 ulp = slamch("P");  /* unit roundoff (eps * base) */

    if (n <= 0) return;

    /* Generate test matrix */
    generate_st_matrix(n, imat, fix->D, fix->E, fix->rng_state);

    /* Save originals (ssterf destroys D and E) */
    memcpy(fix->D_orig, fix->D, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->E_orig, fix->E, (n - 1) * sizeof(f32));
    }

    /* Make copies for ssteqr */
    memcpy(fix->D2, fix->D, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->E2, fix->E, (n - 1) * sizeof(f32));
    }

    /* Compute eigenvalues with ssterf */
    ssterf(n, fix->D, fix->E, &info);
    assert_info_success(info);

    /* Compute eigenvalues with ssteqr (no eigenvectors) for comparison */
    ssteqr("N", n, fix->D2, fix->E2, fix->Z, n, fix->work, &info);
    assert_info_success(info);

    /* Both should produce sorted eigenvalues, but ensure it */
    sort_ascending(n, fix->D);
    sort_ascending(n, fix->D2);

    /* Verification 1: sstech Sturm count check on ssterf eigenvalues */
    int stech_info;
    sstech(n, fix->D_orig, fix->E_orig, fix->D, THRESH,
           fix->work, &stech_info);
    assert_info_success(stech_info);

    /* Verification 2: Compare ssterf vs ssteqr eigenvalues */
    /* Compute max|D[i] - D2[i]| / (|max_eig| * n * ulp) */
    f32 max_eig = 0.0f;
    for (int i = 0; i < n; i++) {
        f32 abs_val = fabsf(fix->D[i]);
        if (abs_val > max_eig) max_eig = abs_val;
    }

    f32 max_diff = 0.0f;
    for (int i = 0; i < n; i++) {
        f32 diff = fabsf(fix->D[i] - fix->D2[i]);
        if (diff > max_diff) max_diff = diff;
    }

    f32 resid;
    if (max_eig == 0.0f) {
        /* Zero matrix: eigenvalues should be exactly zero */
        resid = max_diff / ulp;
    } else {
        resid = max_diff / (max_eig * (f32)n * ulp);
    }

    assert_residual_ok(resid);
}

/**
 * Additional verification for exact eigenvalue cases.
 */
static void run_dsterf_exact_test(dsterf_fixture_t* fix, int imat)
{
    int info;
    int n = fix->n;

    if (n <= 0) return;

    /* Generate test matrix */
    generate_st_matrix(n, imat, fix->D, fix->E, fix->rng_state);

    /* Save originals */
    memcpy(fix->D_orig, fix->D, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->E_orig, fix->E, (n - 1) * sizeof(f32));
    }

    /* Compute eigenvalues with ssterf */
    ssterf(n, fix->D, fix->E, &info);
    assert_info_success(info);

    sort_ascending(n, fix->D);

    if (imat == 1) {
        /* Zero matrix: all eigenvalues should be exactly 0 */
        for (int i = 0; i < n; i++) {
            assert_true(fix->D[i] == 0.0f);
        }
    } else if (imat == 2) {
        /* Identity: all eigenvalues should be exactly 1 */
        for (int i = 0; i < n; i++) {
            assert_true(fix->D[i] == 1.0f);
        }
    } else if (imat == 3) {
        /* 1-2-1 Toeplitz: eigenvalues = 2 - 2*cos(k*pi/(n+1)), k=1..n */
        f32 ulp = slamch("P");
        f32 pi = 3.14159265358979323846264338327950288f;
        f32* exact = malloc(n * sizeof(f32));
        assert_non_null(exact);

        for (int k = 1; k <= n; k++) {
            exact[k - 1] = 2.0f - 2.0f * cosf((f32)k * pi / (f32)(n + 1));
        }
        sort_ascending(n, exact);

        f32 max_eig = 0.0f;
        for (int i = 0; i < n; i++) {
            f32 abs_val = fabsf(exact[i]);
            if (abs_val > max_eig) max_eig = abs_val;
        }

        f32 max_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            f32 diff = fabsf(fix->D[i] - exact[i]);
            if (diff > max_diff) max_diff = diff;
        }

        f32 resid = (max_eig > 0.0f) ?
            max_diff / (max_eig * (f32)n * ulp) : max_diff / ulp;
        assert_residual_ok(resid);

        free(exact);
    }
}

/**
 * Test ssterf for all matrix types (1-7).
 * Verifies with both sstech and ssteqr comparison.
 */
static void test_dsterf_all_types(void** state)
{
    dsterf_fixture_t* fix = *state;

    if (fix->n == 0) {
        skip_test("n=0: nothing to verify");
    }

    for (int imat = 1; imat <= 7; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        run_dsterf_test(fix, imat);
    }
}

/**
 * Test ssterf for exact eigenvalue cases (zero matrix, identity, 1-2-1 Toeplitz).
 */
static void test_dsterf_exact(void** state)
{
    dsterf_fixture_t* fix = *state;

    if (fix->n == 0) {
        skip_test("n=0: nothing to verify");
    }

    /* Test types 1 (zero), 2 (identity), 3 (1-2-1 Toeplitz) */
    for (int imat = 1; imat <= 3; imat++) {
        fix->seed = g_seed++;
        rng_seed(fix->rng_state, fix->seed);
        run_dsterf_exact_test(fix, imat);
    }
}

/*
 * Macro to generate test entries for a given size.
 * Creates 2 test cases: all_types (sstech + ssteqr comparison), exact checks.
 */
#define DSTERF_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsterf_all_types, setup_fn, dsterf_teardown), \
    cmocka_unit_test_setup_teardown(test_dsterf_exact, setup_fn, dsterf_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSTERF_TESTS(setup_1),
        DSTERF_TESTS(setup_2),
        DSTERF_TESTS(setup_5),
        DSTERF_TESTS(setup_10),
        DSTERF_TESTS(setup_20),
        DSTERF_TESTS(setup_50),
        DSTERF_TESTS(setup_100),
    };

    return cmocka_run_group_tests_name("dsterf", tests, NULL, NULL);
}
