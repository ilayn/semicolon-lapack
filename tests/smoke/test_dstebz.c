/**
 * @file test_dstebz.c
 * @brief CMocka test suite for dstebz (bisection eigenvalues of symmetric tridiagonal matrix).
 *
 * Tests the eigenvalue computation routine dstebz using three RANGE modes:
 *   RANGE='A': All eigenvalues (compared against dsterf reference)
 *   RANGE='V': Value range (Sturm sequence verification via dstech)
 *   RANGE='I': Index range (subset comparison with full eigenvalue list)
 *
 * Matrix types tested (tridiagonal, generated directly):
 *   1. Zero matrix
 *   2. Identity (D=1, E=0)
 *   3. 1-2-1 Toeplitz (D=2, E=1)
 *   4. Wilkinson (D[i]=|i-n/2|, E[i]=1)
 *   5. Random symmetric tridiagonal
 *   6. Graded diagonal (D[i]=2^(-i), E[i]=1)
 *
 * Sizes tested: 2, 5, 10, 20, 50
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dstebz(const char* range, const char* order, const int n,
                   const double vl, const double vu,
                   const int il, const int iu, const double abstol,
                   const double* D, const double* E,
                   int* m, int* nsplit, double* W,
                   int* iblock, int* isplit,
                   double* work, int* iwork, int* info);

/* Reference eigenvalue computation */
extern void dsterf(const int n, double* D, double* E, int* info);

/* Eigenvalue verification via Sturm sequence */
extern void dstech(const int n, const double* A, const double* B,
                   const double* eig, const double tol, double* work, int* info);

/* Utility */
extern double dlamch(const char* cmach);

/* -----------------------------------------------------------------------
 * RNG: xoshiro256+ (Blackman & Vigna, 2018)
 * ----------------------------------------------------------------------- */
static uint64_t rng_state[4];

static void rng_seed(uint64_t s)
{
    for (int i = 0; i < 4; i++) {
        s = s * 6364136223846793005ULL + 1442695040888963407ULL;
        rng_state[i] = s;
    }
}

static double rng_uniform(void)
{
    uint64_t s = rng_state[1] * 5;
    uint64_t r = ((s << 7) | (s >> 57)) * 9;
    uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = (rng_state[3] << 45) | (rng_state[3] >> 19);
    return (double)(r >> 11) * 0x1.0p-53;
}

/* -----------------------------------------------------------------------
 * Test fixture
 * ----------------------------------------------------------------------- */
typedef struct {
    int n;
    double* D;       /* diagonal (preserved - dstebz doesn't modify) */
    double* E;       /* off-diagonal (preserved - dstebz doesn't modify) */
    double* D2;      /* copy of D for dsterf */
    double* E2;      /* copy of E for dsterf */
    double* W;       /* eigenvalues from dstebz */
    double* W_ref;   /* reference eigenvalues from dsterf */
    int* iblock;     /* block info */
    int* isplit;     /* split info */
    double* work;    /* workspace (4*n) */
    int* iwork;      /* int workspace (3*n) */
    uint64_t seed;
} dstebz_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 2024;

/* -----------------------------------------------------------------------
 * Matrix generation helpers
 * ----------------------------------------------------------------------- */

/** Matrix type 1: Zero matrix */
static void gen_zero(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) D[i] = 0.0;
    for (int i = 0; i < n - 1; i++) E[i] = 0.0;
}

/** Matrix type 2: Identity (D=1, E=0) */
static void gen_identity(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) D[i] = 1.0;
    for (int i = 0; i < n - 1; i++) E[i] = 0.0;
}

/** Matrix type 3: 1-2-1 Toeplitz (D=2, E=1) */
static void gen_toeplitz_121(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) D[i] = 2.0;
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/** Matrix type 4: Wilkinson (D[i]=|i - n/2|, E[i]=1) */
static void gen_wilkinson(int n, double* D, double* E)
{
    int half = n / 2;
    for (int i = 0; i < n; i++) {
        D[i] = (double)(i >= half ? i - half : half - i);
    }
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/** Matrix type 5: Random symmetric tridiagonal */
static void gen_random(int n, double* D, double* E, uint64_t seed)
{
    rng_seed(seed);
    for (int i = 0; i < n; i++) {
        D[i] = 2.0 * rng_uniform() - 1.0;
    }
    for (int i = 0; i < n - 1; i++) {
        E[i] = 2.0 * rng_uniform() - 1.0;
    }
}

/** Matrix type 6: Graded diagonal (D[i]=2^(-i), E[i]=1) */
static void gen_graded(int n, double* D, double* E)
{
    for (int i = 0; i < n; i++) {
        D[i] = pow(2.0, -(double)i);
    }
    for (int i = 0; i < n - 1; i++) E[i] = 1.0;
}

/** Generate matrix of given type */
static void generate_matrix(int n, int imat, double* D, double* E, uint64_t seed)
{
    switch (imat) {
        case 1: gen_zero(n, D, E); break;
        case 2: gen_identity(n, D, E); break;
        case 3: gen_toeplitz_121(n, D, E); break;
        case 4: gen_wilkinson(n, D, E); break;
        case 5: gen_random(n, D, E, seed); break;
        case 6: gen_graded(n, D, E); break;
    }
}

/* -----------------------------------------------------------------------
 * Comparison helper for qsort (ascending doubles)
 * ----------------------------------------------------------------------- */
static int cmp_double(const void* a, const void* b)
{
    double da = *(const double*)a;
    double db = *(const double*)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

/* -----------------------------------------------------------------------
 * Setup / Teardown
 * ----------------------------------------------------------------------- */
static int dstebz_setup(void** state, int n)
{
    dstebz_fixture_t* fix = malloc(sizeof(dstebz_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    int n_alloc = (n > 0) ? n : 1;
    int e_alloc = (n > 1) ? n - 1 : 1;
    int work_sz = (n > 0) ? 4 * n : 4;
    int iwork_sz = (n > 0) ? 3 * n : 3;

    fix->D = malloc(n_alloc * sizeof(double));
    fix->E = malloc(e_alloc * sizeof(double));
    fix->D2 = malloc(n_alloc * sizeof(double));
    fix->E2 = malloc(e_alloc * sizeof(double));
    fix->W = malloc(n_alloc * sizeof(double));
    fix->W_ref = malloc(n_alloc * sizeof(double));
    fix->iblock = malloc(n_alloc * sizeof(int));
    fix->isplit = malloc(n_alloc * sizeof(int));
    fix->work = malloc(work_sz * sizeof(double));
    fix->iwork = malloc(iwork_sz * sizeof(int));

    assert_non_null(fix->D);
    assert_non_null(fix->E);
    assert_non_null(fix->D2);
    assert_non_null(fix->E2);
    assert_non_null(fix->W);
    assert_non_null(fix->W_ref);
    assert_non_null(fix->iblock);
    assert_non_null(fix->isplit);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

static int dstebz_teardown(void** state)
{
    dstebz_fixture_t* fix = *state;
    if (fix) {
        free(fix->D);
        free(fix->E);
        free(fix->D2);
        free(fix->E2);
        free(fix->W);
        free(fix->W_ref);
        free(fix->iblock);
        free(fix->isplit);
        free(fix->work);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_2(void** state) { return dstebz_setup(state, 2); }
static int setup_5(void** state) { return dstebz_setup(state, 5); }
static int setup_10(void** state) { return dstebz_setup(state, 10); }
static int setup_20(void** state) { return dstebz_setup(state, 20); }
static int setup_50(void** state) { return dstebz_setup(state, 50); }

/* -----------------------------------------------------------------------
 * test_range_all: RANGE='A', all eigenvalues
 *
 * For each matrix type:
 *   - Call dstebz("A", "E", ...) to compute all eigenvalues
 *   - Verify m == n
 *   - Verify eigenvalues via dstech (Sturm sequence check)
 *   - Compare with dsterf reference: max|W[i]-W_ref[i]|/(n*ulp*max|W_ref|)
 * ----------------------------------------------------------------------- */
static void test_range_all(void** state)
{
    dstebz_fixture_t* fix = *state;
    int n = fix->n;
    int info, info2;
    int m, nsplit;
    double ulp = dlamch("E");

    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        generate_matrix(n, imat, fix->D, fix->E, fix->seed);

        /* Compute all eigenvalues via dstebz */
        dstebz("A", "E", n, 0.0, 0.0, 0, 0, 2.0 * dlamch("S"),
               fix->D, fix->E, &m, &nsplit, fix->W,
               fix->iblock, fix->isplit, fix->work, fix->iwork, &info);
        assert_info_success(info);
        assert_int_equal(m, n);

        /* Verify eigenvalues via dstech (Sturm sequence) */
        /* dstech needs workspace of size n */
        double* work2 = malloc(n * sizeof(double));
        assert_non_null(work2);
        dstech(n, fix->D, fix->E, fix->W, THRESH, work2, &info2);
        assert_info_success(info2);
        free(work2);

        /* Compute reference eigenvalues via dsterf */
        memcpy(fix->D2, fix->D, n * sizeof(double));
        if (n > 1) {
            memcpy(fix->E2, fix->E, (n - 1) * sizeof(double));
        }
        dsterf(n, fix->D2, fix->E2, &info2);
        assert_info_success(info2);

        /* dsterf returns eigenvalues in ascending order in D2 */
        memcpy(fix->W_ref, fix->D2, n * sizeof(double));

        /* Sort dstebz eigenvalues for comparison */
        qsort(fix->W, n, sizeof(double), cmp_double);

        /* Compute max|W[i] - W_ref[i]| / (n * ulp * max|W_ref|) */
        double maxref = 0.0;
        for (int i = 0; i < n; i++) {
            double absref = fabs(fix->W_ref[i]);
            if (absref > maxref) maxref = absref;
        }
        /* For zero matrix, all eigenvalues are zero */
        if (maxref == 0.0) maxref = 1.0;

        double maxerr = 0.0;
        for (int i = 0; i < n; i++) {
            double err = fabs(fix->W[i] - fix->W_ref[i]);
            if (err > maxerr) maxerr = err;
        }
        double resid = maxerr / (n * ulp * maxref);
        assert_residual_ok(resid);
    }
}

/* -----------------------------------------------------------------------
 * test_range_value: RANGE='V', value range
 *
 * Uses 1-2-1 Toeplitz matrix:
 *   - Compute all eigenvalues with dsterf to get reference
 *   - Pick vl = W_ref[n/4], vu = W_ref[3*n/4]
 *   - Call dstebz("V", "E", ..., vl, vu, ...)
 *   - Verify each W[i] is in (vl, vu]
 *   - Verify m matches expected count from reference
 * ----------------------------------------------------------------------- */
static void test_range_value(void** state)
{
    dstebz_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m, nsplit;

    if (n < 8) {
        skip_test("RANGE='V' requires n >= 8 for meaningful subsets");
    }

    /* Generate 1-2-1 Toeplitz matrix */
    gen_toeplitz_121(n, fix->D, fix->E);

    /* Get reference eigenvalues via dsterf */
    memcpy(fix->D2, fix->D, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E2, fix->E, (n - 1) * sizeof(double));
    }
    dsterf(n, fix->D2, fix->E2, &info);
    assert_info_success(info);
    memcpy(fix->W_ref, fix->D2, n * sizeof(double));

    /* Pick vl and vu as midpoints between consecutive eigenvalues to avoid
     * boundary sensitivity (eigenvalues on the boundary may be included or
     * excluded depending on rounding between dsterf and dstebz). */
    double vl = 0.5 * (fix->W_ref[n / 4 - 1] + fix->W_ref[n / 4]);
    double vu = 0.5 * (fix->W_ref[3 * n / 4] + fix->W_ref[3 * n / 4 + 1]);

    /* Call dstebz with RANGE='V' */
    dstebz("V", "E", n, vl, vu, 0, 0, 2.0 * dlamch("S"),
           fix->D, fix->E, &m, &nsplit, fix->W,
           fix->iblock, fix->isplit, fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Verify each eigenvalue is in (vl, vu] */
    for (int i = 0; i < m; i++) {
        assert_true(fix->W[i] > vl);
        assert_true(fix->W[i] <= vu);
    }

    /* Count expected eigenvalues in (vl, vu] from reference */
    int expected_m = 0;
    for (int i = 0; i < n; i++) {
        if (fix->W_ref[i] > vl && fix->W_ref[i] <= vu) {
            expected_m++;
        }
    }
    assert_int_equal(m, expected_m);

    /* Verify subset eigenvalues by comparison with reference:
     * max|W[i] - W_ref[k]| / (n * ulp * max|W_ref|) < THRESH
     * where k ranges over the eigenvalues in (vl, vu] */
    if (m > 0) {
        double ulp = dlamch("E") * dlamch("B");
        double max_eig = fabs(fix->W_ref[n - 1]);  /* largest in magnitude */
        for (int i = 0; i < n; i++) {
            double a = fabs(fix->W_ref[i]);
            if (a > max_eig) max_eig = a;
        }

        /* k should equal m (already verified by assert_int_equal above) */

        /* Compare sorted subset eigenvalues */
        double max_diff = 0.0;
        int ref_idx = 0;
        for (int i = 0; i < n && ref_idx < m; i++) {
            if (fix->W_ref[i] > vl && fix->W_ref[i] <= vu) {
                double diff = fabs(fix->W[ref_idx] - fix->W_ref[i]);
                if (diff > max_diff) max_diff = diff;
                ref_idx++;
            }
        }
        double denom = (double)n * ulp;
        if (max_eig > 0.0) denom *= max_eig;
        double resid = (denom > 0.0) ? max_diff / denom : 0.0;
        assert_residual_ok(resid);
    }
}

/* -----------------------------------------------------------------------
 * test_range_index: RANGE='I', index range
 *
 * Uses 1-2-1 Toeplitz matrix:
 *   - Call dstebz("I", "E", ..., il=n/4+1, iu=3*n/4, ...)
 *   - il and iu are 1-based
 *   - Compare with corresponding subset of dsterf eigenvalues
 * ----------------------------------------------------------------------- */
static void test_range_index(void** state)
{
    dstebz_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m, nsplit;
    double ulp = dlamch("E");

    if (n < 8) {
        skip_test("RANGE='I' requires n >= 8 for meaningful subsets");
    }

    /* Generate 1-2-1 Toeplitz matrix */
    gen_toeplitz_121(n, fix->D, fix->E);

    /* il and iu are 1-based indices */
    int il = n / 4 + 1;
    int iu = 3 * n / 4;

    /* Call dstebz with RANGE='I' */
    dstebz("I", "E", n, 0.0, 0.0, il, iu, 2.0 * dlamch("S"),
           fix->D, fix->E, &m, &nsplit, fix->W,
           fix->iblock, fix->isplit, fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Expected count */
    int expected_m = iu - il + 1;
    assert_int_equal(m, expected_m);

    /* Get reference eigenvalues via dsterf */
    memcpy(fix->D2, fix->D, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E2, fix->E, (n - 1) * sizeof(double));
    }
    dsterf(n, fix->D2, fix->E2, &info);
    assert_info_success(info);
    memcpy(fix->W_ref, fix->D2, n * sizeof(double));

    /* Sort dstebz eigenvalues for comparison */
    qsort(fix->W, m, sizeof(double), cmp_double);

    /* Compare with the corresponding subset of sorted reference eigenvalues */
    /* Reference eigenvalues are sorted; indices il..iu correspond to positions [il-1..iu-1] */
    double maxref = 0.0;
    for (int i = 0; i < n; i++) {
        double absref = fabs(fix->W_ref[i]);
        if (absref > maxref) maxref = absref;
    }
    if (maxref == 0.0) maxref = 1.0;

    double maxerr = 0.0;
    for (int i = 0; i < m; i++) {
        double err = fabs(fix->W[i] - fix->W_ref[il - 1 + i]);
        if (err > maxerr) maxerr = err;
    }
    double resid = maxerr / (n * ulp * maxref);
    assert_residual_ok(resid);
}

/* -----------------------------------------------------------------------
 * Macro to generate test entries for a given size
 * ----------------------------------------------------------------------- */
#define DSTEBZ_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_range_all, setup_fn, dstebz_teardown), \
    cmocka_unit_test_setup_teardown(test_range_value, setup_fn, dstebz_teardown), \
    cmocka_unit_test_setup_teardown(test_range_index, setup_fn, dstebz_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSTEBZ_TESTS(setup_2),
        DSTEBZ_TESTS(setup_5),
        DSTEBZ_TESTS(setup_10),
        DSTEBZ_TESTS(setup_20),
        DSTEBZ_TESTS(setup_50),
    };

    return cmocka_run_group_tests_name("dstebz", tests, NULL, NULL);
}
