/**
 * @file test_dsterf.c
 * @brief CMocka test suite for dsterf (eigenvalues of symmetric tridiagonal matrix).
 *
 * Tests the eigenvalue computation routine dsterf using the
 * Pal-Walker-Kahan QL/QR algorithm.
 *
 * Verification:
 *   1. dstech (Sturm count) verifies eigenvalues are correct to tolerance.
 *   2. Comparison with dsteqr eigenvalues: max|D1[i]-D2[i]|/(|max_eig|*n*ulp) < THRESH
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
#define THRESH 20.0
#include "testutils/test_rng.h"

/* Routine under test */
extern void dsterf(const int n, double* D, double* E, int* info);

/* Reference eigenvalue computation */
extern void dsteqr(const char* compz, const int n, double* D, double* E,
                   double* Z, const int ldz, double* work, int* info);

/* Verification routine: Sturm count check */
extern void dstech(const int n, const double* A, const double* B,
                   const double* eig, const double tol, double* work, int* info);

/* Machine parameters */
extern double dlamch(const char* cmach);

/*
 * Test fixture: holds all allocated memory for a single test case.
 * CMocka passes this between setup -> test -> teardown.
 */
typedef struct {
    int n;
    double* D;      /* diagonal (input to dsterf, overwritten with eigenvalues) */
    double* E;      /* off-diagonal (input to dsterf, destroyed) */
    double* D_orig; /* saved copy of diagonal */
    double* E_orig; /* saved copy of off-diagonal */
    double* D2;     /* copy for dsteqr comparison */
    double* E2;     /* copy for dsteqr comparison */
    double* Z;      /* eigenvector workspace for dsteqr */
    double* work;   /* workspace for dsteqr and dstech */
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
static void generate_st_matrix(int n, int imat, double* D, double* E,
                               uint64_t state[static 4])
{
    int i;

    if (n <= 0) return;

    switch (imat) {
        case 1:
            /* Zero matrix */
            for (i = 0; i < n; i++) D[i] = 0.0;
            for (i = 0; i < n - 1; i++) E[i] = 0.0;
            break;

        case 2:
            /* Identity (D=1, E=0) */
            for (i = 0; i < n; i++) D[i] = 1.0;
            for (i = 0; i < n - 1; i++) E[i] = 0.0;
            break;

        case 3:
            /* 1-2-1 Toeplitz tridiagonal: D[i]=2, E[i]=1 */
            /* Known eigenvalues: 2 - 2*cos(k*pi/(n+1)), k=1,...,n */
            for (i = 0; i < n; i++) D[i] = 2.0;
            for (i = 0; i < n - 1; i++) E[i] = 1.0;
            break;

        case 4:
            /* Wilkinson-like: D[i]=|i - n/2|, E[i]=1 */
            for (i = 0; i < n; i++) {
                D[i] = fabs((double)(i - n / 2));
            }
            for (i = 0; i < n - 1; i++) E[i] = 1.0;
            break;

        case 5:
            /* Random diagonal, zero off-diagonal */
            /* Eigenvalues = diagonal entries */
            for (i = 0; i < n; i++) D[i] = rng_uniform_symmetric(state) * 10.0;
            for (i = 0; i < n - 1; i++) E[i] = 0.0;
            break;

        case 6:
            /* Random symmetric tridiagonal (well-conditioned) */
            for (i = 0; i < n; i++) {
                D[i] = rng_uniform_symmetric(state) * 5.0 + (double)n;
            }
            for (i = 0; i < n - 1; i++) {
                E[i] = rng_uniform_symmetric(state);
            }
            break;

        case 7:
            /* Graded diagonal: D[i]=2^(-i), E[i]=1 */
            /* Tests scaling behavior */
            for (i = 0; i < n; i++) {
                D[i] = pow(2.0, -(double)i);
            }
            for (i = 0; i < n - 1; i++) E[i] = 1.0;
            break;

        default:
            /* Fallback: identity */
            for (i = 0; i < n; i++) D[i] = 1.0;
            for (i = 0; i < n - 1; i++) E[i] = 0.0;
            break;
    }
}

/**
 * Simple insertion sort for double array (ascending order).
 * Used to ensure eigenvalue arrays are sorted for comparison.
 */
static void sort_ascending(int n, double* x)
{
    int i, j;
    double tmp;
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
    /* dsteqr work size: max(1, 2*(n-1)) */
    int work_size = (n > 1) ? 2 * (n - 1) : 1;
    /* dstech also needs n workspace, take max */
    if (n_alloc > work_size) work_size = n_alloc;

    fix->D = malloc(n_alloc * sizeof(double));
    fix->E = malloc(e_alloc * sizeof(double));
    fix->D_orig = malloc(n_alloc * sizeof(double));
    fix->E_orig = malloc(e_alloc * sizeof(double));
    fix->D2 = malloc(n_alloc * sizeof(double));
    fix->E2 = malloc(e_alloc * sizeof(double));
    fix->Z = malloc(n_alloc * n_alloc * sizeof(double));
    fix->work = malloc(work_size * sizeof(double));

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
 * Core test logic: generate matrix, compute eigenvalues with dsterf,
 * verify with dstech and compare with dsteqr.
 */
static void run_dsterf_test(dsterf_fixture_t* fix, int imat)
{
    int info;
    int n = fix->n;
    double ulp = dlamch("P");  /* unit roundoff (eps * base) */

    if (n <= 0) return;

    /* Generate test matrix */
    generate_st_matrix(n, imat, fix->D, fix->E, fix->rng_state);

    /* Save originals (dsterf destroys D and E) */
    memcpy(fix->D_orig, fix->D, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E_orig, fix->E, (n - 1) * sizeof(double));
    }

    /* Make copies for dsteqr */
    memcpy(fix->D2, fix->D, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E2, fix->E, (n - 1) * sizeof(double));
    }

    /* Compute eigenvalues with dsterf */
    dsterf(n, fix->D, fix->E, &info);
    assert_info_success(info);

    /* Compute eigenvalues with dsteqr (no eigenvectors) for comparison */
    dsteqr("N", n, fix->D2, fix->E2, fix->Z, n, fix->work, &info);
    assert_info_success(info);

    /* Both should produce sorted eigenvalues, but ensure it */
    sort_ascending(n, fix->D);
    sort_ascending(n, fix->D2);

    /* Verification 1: dstech Sturm count check on dsterf eigenvalues */
    int stech_info;
    dstech(n, fix->D_orig, fix->E_orig, fix->D, THRESH,
           fix->work, &stech_info);
    assert_info_success(stech_info);

    /* Verification 2: Compare dsterf vs dsteqr eigenvalues */
    /* Compute max|D[i] - D2[i]| / (|max_eig| * n * ulp) */
    double max_eig = 0.0;
    for (int i = 0; i < n; i++) {
        double abs_val = fabs(fix->D[i]);
        if (abs_val > max_eig) max_eig = abs_val;
    }

    double max_diff = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = fabs(fix->D[i] - fix->D2[i]);
        if (diff > max_diff) max_diff = diff;
    }

    double resid;
    if (max_eig == 0.0) {
        /* Zero matrix: eigenvalues should be exactly zero */
        resid = max_diff / ulp;
    } else {
        resid = max_diff / (max_eig * (double)n * ulp);
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
    memcpy(fix->D_orig, fix->D, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E_orig, fix->E, (n - 1) * sizeof(double));
    }

    /* Compute eigenvalues with dsterf */
    dsterf(n, fix->D, fix->E, &info);
    assert_info_success(info);

    sort_ascending(n, fix->D);

    if (imat == 1) {
        /* Zero matrix: all eigenvalues should be exactly 0 */
        for (int i = 0; i < n; i++) {
            assert_true(fix->D[i] == 0.0);
        }
    } else if (imat == 2) {
        /* Identity: all eigenvalues should be exactly 1 */
        for (int i = 0; i < n; i++) {
            assert_true(fix->D[i] == 1.0);
        }
    } else if (imat == 3) {
        /* 1-2-1 Toeplitz: eigenvalues = 2 - 2*cos(k*pi/(n+1)), k=1..n */
        double ulp = dlamch("P");
        double pi = 3.14159265358979323846264338327950288;
        double* exact = malloc(n * sizeof(double));
        assert_non_null(exact);

        for (int k = 1; k <= n; k++) {
            exact[k - 1] = 2.0 - 2.0 * cos((double)k * pi / (double)(n + 1));
        }
        sort_ascending(n, exact);

        double max_eig = 0.0;
        for (int i = 0; i < n; i++) {
            double abs_val = fabs(exact[i]);
            if (abs_val > max_eig) max_eig = abs_val;
        }

        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            double diff = fabs(fix->D[i] - exact[i]);
            if (diff > max_diff) max_diff = diff;
        }

        double resid = (max_eig > 0.0) ?
            max_diff / (max_eig * (double)n * ulp) : max_diff / ulp;
        assert_residual_ok(resid);

        free(exact);
    }
}

/**
 * Test dsterf for all matrix types (1-7).
 * Verifies with both dstech and dsteqr comparison.
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
 * Test dsterf for exact eigenvalue cases (zero matrix, identity, 1-2-1 Toeplitz).
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
 * Creates 2 test cases: all_types (dstech + dsteqr comparison), exact checks.
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
