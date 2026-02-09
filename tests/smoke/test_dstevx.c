/**
 * @file test_dstevx.c
 * @brief CMocka test suite for dstevx (expert eigenvalue driver for symmetric tridiagonal).
 *
 * Tests the selective eigenvalue/eigenvector computation for symmetric
 * tridiagonal matrices. dstevx supports:
 *   - JOBZ='V' or 'N' (eigenvectors or eigenvalues only)
 *   - RANGE='A' (all), 'V' (value range), 'I' (index range)
 *
 * Verification:
 *   dstt21 checks |T - Z*D*Z'| / (|T|*n*ulp) and |I - Z*Z'| / (n*ulp)
 *   dstech verifies eigenvalues via Sturm sequence
 *   Manual orthogonality check for partial eigenvector sets
 *
 * Matrix types tested (symmetric tridiagonal):
 *   1. Identity (D=1, E=0)
 *   2. 1-2-1 Toeplitz (D[i]=2, E[i]=1)
 *   3. Wilkinson-like (D[i]=|i-n/2|, E[i]=1)
 *   4. Random symmetric tridiagonal
 *   5. Graded diagonal (D[i]=2^(-i), E[i]=1)
 */

#include "test_harness.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dstevx(const char* jobz, const char* range, const int n,
                   double* const restrict D, double* const restrict E,
                   const double vl, const double vu,
                   const int il, const int iu, const double abstol,
                   int* m, double* const restrict W,
                   double* const restrict Z, const int ldz,
                   double* const restrict work, int* const restrict iwork,
                   int* const restrict ifail, int* info);

/* Reference routine for eigenvalue comparison */
extern void dstev(const char* jobz, const int n,
                  double* const restrict D, double* const restrict E,
                  double* const restrict Z, const int ldz,
                  double* const restrict work, int* info);

/* Verification routines */
extern void dstt21(const int n, const int kband,
                   const double* AD, const double* AE,
                   const double* SD, const double* SE,
                   const double* U, const int ldu,
                   double* work, double* result);
extern void dstech(const int n, const double* A, const double* B,
                   const double* eig, const double tol, double* work, int* info);

/* Utilities */
extern double dlamch(const char* cmach);

/*
 * Test fixture: holds all allocated memory for a single test case.
 * CMocka passes this between setup -> test -> teardown.
 */
typedef struct {
    int n;
    double* AD;         /* original diagonal (preserved) */
    double* AE;         /* original off-diagonal (preserved) */
    double* D;          /* diagonal for dstevx (overwritten) */
    double* E;          /* off-diagonal for dstevx (overwritten) */
    double* W;          /* eigenvalues output */
    double* W_ref;      /* reference eigenvalues from dstev */
    double* Z;          /* eigenvectors (n x n) */
    double* work;       /* workspace: max(5*n, n*(n+1)) for dstevx + dstt21 */
    int* iwork;         /* int workspace (5*n) */
    int* ifail;         /* convergence info (n) */
    double* result;     /* dstt21 results (2 elements) */
    uint64_t seed;
    uint64_t rng_state[4];
} dstevx_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 3000;

/**
 * Setup fixture: allocate memory for given dimension.
 * Called before each test function.
 */
static int dstevx_setup(void** state, int n)
{
    dstevx_fixture_t* fix = malloc(sizeof(dstevx_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    int n_alloc = (n > 0) ? n : 1;
    int e_alloc = (n > 1) ? n - 1 : 1;
    int work_stevx = 5 * n_alloc;
    int work_stt21 = n_alloc * (n_alloc + 1);
    int work_total = (work_stevx > work_stt21) ? work_stevx : work_stt21;

    fix->AD = malloc(n_alloc * sizeof(double));
    fix->AE = malloc(e_alloc * sizeof(double));
    fix->D = malloc(n_alloc * sizeof(double));
    fix->E = malloc(e_alloc * sizeof(double));
    fix->W = malloc(n_alloc * sizeof(double));
    fix->W_ref = malloc(n_alloc * sizeof(double));
    fix->Z = malloc(n_alloc * n_alloc * sizeof(double));
    fix->work = malloc(work_total * sizeof(double));
    fix->iwork = malloc(5 * n_alloc * sizeof(int));
    fix->ifail = malloc(n_alloc * sizeof(int));
    fix->result = malloc(2 * sizeof(double));

    assert_non_null(fix->AD);
    assert_non_null(fix->AE);
    assert_non_null(fix->D);
    assert_non_null(fix->E);
    assert_non_null(fix->W);
    assert_non_null(fix->W_ref);
    assert_non_null(fix->Z);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);
    assert_non_null(fix->ifail);
    assert_non_null(fix->result);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 * Called after each test function.
 */
static int dstevx_teardown(void** state)
{
    dstevx_fixture_t* fix = *state;
    if (fix) {
        free(fix->AD);
        free(fix->AE);
        free(fix->D);
        free(fix->E);
        free(fix->W);
        free(fix->W_ref);
        free(fix->Z);
        free(fix->work);
        free(fix->iwork);
        free(fix->ifail);
        free(fix->result);
        free(fix);
    }
    return 0;
}

/* Size-specific setup functions */
static int setup_2(void** state) { return dstevx_setup(state, 2); }
static int setup_5(void** state) { return dstevx_setup(state, 5); }
static int setup_10(void** state) { return dstevx_setup(state, 10); }
static int setup_20(void** state) { return dstevx_setup(state, 20); }
static int setup_50(void** state) { return dstevx_setup(state, 50); }

/**
 * Generate symmetric tridiagonal test matrix.
 *
 * @param n     Matrix dimension
 * @param imat  Matrix type (1-5)
 * @param D     Diagonal array (length n)
 * @param E     Off-diagonal array (length n-1)
 * @param seed  RNG seed (used for type 4)
 */
static void generate_stevx_matrix(int n, int imat, double* D, double* E,
                                   uint64_t state[static 4])
{
    int i;

    switch (imat) {
    case 1:
        /* Identity: D=1, E=0 */
        for (i = 0; i < n; i++) D[i] = 1.0;
        for (i = 0; i < n - 1; i++) E[i] = 0.0;
        break;

    case 2:
        /* 1-2-1 Toeplitz: D[i]=2, E[i]=1 */
        for (i = 0; i < n; i++) D[i] = 2.0;
        for (i = 0; i < n - 1; i++) E[i] = 1.0;
        break;

    case 3:
        /* Wilkinson-like: D[i]=|i - n/2|, E[i]=1 */
        for (i = 0; i < n; i++) D[i] = fabs((double)i - (double)(n / 2));
        for (i = 0; i < n - 1; i++) E[i] = 1.0;
        break;

    case 4:
        /* Random symmetric tridiagonal */
        for (i = 0; i < n; i++) D[i] = rng_uniform_symmetric(state);
        for (i = 0; i < n - 1; i++) E[i] = rng_uniform_symmetric(state);
        break;

    case 5:
        /* Graded diagonal: D[i]=2^(-i), E[i]=1 */
        for (i = 0; i < n; i++) D[i] = pow(2.0, -(double)i);
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
 * Test JOBZ='V', RANGE='A': all eigenvalues and eigenvectors.
 * Verifies via dstt21:
 *   result[0] = |T - Z*D*Z'| / (|T|*n*ulp)
 *   result[1] = |I - Z*Z'| / (n*ulp)
 * Also checks that all ifail[i] == 0.
 */
static void test_range_all_V(void** state)
{
    dstevx_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m;

    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Save original for verification */
        memcpy(fix->AD, fix->D, n * sizeof(double));
        if (n > 1) {
            memcpy(fix->AE, fix->E, (n - 1) * sizeof(double));
        }

        /* Compute all eigenvalues and eigenvectors (abstol=0.0: fast path) */
        dstevx("V", "A", n, fix->D, fix->E,
               0.0, 0.0, 0, 0, 0.0,
               &m, fix->W, fix->Z, n,
               fix->work, fix->iwork, fix->ifail, &info);
        assert_info_success(info);
        assert_int_equal(m, n);

        /* Verify all ifail entries are zero */
        for (int i = 0; i < n; i++) {
            assert_int_equal(fix->ifail[i], 0);
        }

        /* Verify decomposition via dstt21:
         * kband=0 because W contains eigenvalues (diagonal),
         * SE is not referenced when kband=0, pass NULL */
        dstt21(n, 0, fix->AD, fix->AE, fix->W, NULL,
               fix->Z, n, fix->work, fix->result);

        assert_residual_ok(fix->result[0]);  /* decomposition residual */
        assert_residual_ok(fix->result[1]);  /* orthogonality */
    }
}

/**
 * Test JOBZ='N', RANGE='A': eigenvalues only.
 * Compares eigenvalues with reference from dstev.
 */
static void test_range_all_N(void** state)
{
    dstevx_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m;
    double ulp = dlamch("E");

    /* Allocate temporaries for dstev reference computation */
    int n_alloc = (n > 0) ? n : 1;
    int e_alloc = (n > 1) ? n - 1 : 1;
    double* D_ref = malloc(n_alloc * sizeof(double));
    double* E_ref = malloc(e_alloc * sizeof(double));
    double* work_ref = malloc((2 * n_alloc) * sizeof(double));
    assert_non_null(D_ref);
    assert_non_null(E_ref);
    assert_non_null(work_ref);

    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Copy for dstev reference */
        memcpy(D_ref, fix->D, n * sizeof(double));
        if (n > 1) {
            memcpy(E_ref, fix->E, (n - 1) * sizeof(double));
        }

        /* Compute reference eigenvalues with dstev */
        dstev("N", n, D_ref, E_ref, NULL, 1, work_ref, &info);
        assert_info_success(info);

        /* Regenerate for dstevx (D, E may be overwritten) */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Compute eigenvalues only with dstevx */
        dstevx("N", "A", n, fix->D, fix->E,
               0.0, 0.0, 0, 0, 0.0,
               &m, fix->W, NULL, 1,
               fix->work, fix->iwork, fix->ifail, &info);
        assert_info_success(info);
        assert_int_equal(m, n);

        /* Compare eigenvalues: both should be sorted ascending.
         * Compute max|W[i] - D_ref[i]| / (n * max|D_ref| * ulp) */
        double max_eig = 0.0;
        for (int i = 0; i < n; i++) {
            double a = fabs(D_ref[i]);
            if (a > max_eig) max_eig = a;
        }

        double max_diff = 0.0;
        for (int i = 0; i < n; i++) {
            double d = fabs(fix->W[i] - D_ref[i]);
            if (d > max_diff) max_diff = d;
        }

        double denom = (double)n * ulp;
        if (max_eig > 0.0) denom *= max_eig;
        double resid = (denom > 0.0) ? max_diff / denom : 0.0;

        assert_residual_ok(resid);
    }

    free(D_ref);
    free(E_ref);
    free(work_ref);
}

/**
 * Test JOBZ='V', RANGE='V': value range eigenvalue selection.
 * Uses the 1-2-1 Toeplitz matrix with n=20.
 * Selects a middle subset of eigenvalues by value and verifies:
 *   - Each W[i] is in (vl, vu]
 *   - Eigenvector orthogonality: |I - Z'*Z| / (m*ulp) < THRESH
 */
static void test_range_value(void** state)
{
    dstevx_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m;
    double ulp = dlamch("E");
    double safmin = dlamch("S");
    double abstol = 2.0 * safmin;

    if (n < 10) {
        skip_test("RANGE='V' test requires n >= 10");
    }

    /* Generate 1-2-1 Toeplitz matrix */
    rng_seed(fix->rng_state, fix->seed);
    generate_stevx_matrix(n, 2, fix->D, fix->E, fix->rng_state);

    /* Save original */
    memcpy(fix->AD, fix->D, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->AE, fix->E, (n - 1) * sizeof(double));
    }

    /* First, compute all eigenvalues with dstev to get reference */
    int n_alloc = (n > 0) ? n : 1;
    int e_alloc = (n > 1) ? n - 1 : 1;
    double* D_tmp = malloc(n_alloc * sizeof(double));
    double* E_tmp = malloc(e_alloc * sizeof(double));
    double* work_tmp = malloc((2 * n_alloc) * sizeof(double));
    assert_non_null(D_tmp);
    assert_non_null(E_tmp);
    assert_non_null(work_tmp);

    memcpy(D_tmp, fix->AD, n * sizeof(double));
    if (n > 1) {
        memcpy(E_tmp, fix->AE, (n - 1) * sizeof(double));
    }

    dstev("N", n, D_tmp, E_tmp, NULL, 1, work_tmp, &info);
    assert_info_success(info);

    /* Copy reference eigenvalues */
    memcpy(fix->W_ref, D_tmp, n * sizeof(double));

    /* Pick value range: middle subset */
    int idx_lo = n / 4;       /* roughly 25% from bottom */
    int idx_hi = 3 * n / 4;   /* roughly 75% from bottom */
    double vl = fix->W_ref[idx_lo];
    double vu = fix->W_ref[idx_hi];

    /* Reset D, E from originals */
    memcpy(fix->D, fix->AD, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E, fix->AE, (n - 1) * sizeof(double));
    }

    /* Compute eigenvalues in (vl, vu] with eigenvectors */
    dstevx("V", "V", n, fix->D, fix->E,
           vl, vu, 0, 0, abstol,
           &m, fix->W, fix->Z, n,
           fix->work, fix->iwork, fix->ifail, &info);
    assert_info_success(info);

    /* Verify each eigenvalue is in (vl, vu] */
    for (int i = 0; i < m; i++) {
        assert_true(fix->W[i] > vl);
        assert_true(fix->W[i] <= vu);
    }

    /* Verify eigenvector orthogonality: compute Z(:,1:m)' * Z(:,1:m) into
     * an m x m matrix, then check |I - Z'Z| / (m * ulp) */
    if (m > 0) {
        double* ZtZ = malloc(m * m * sizeof(double));
        assert_non_null(ZtZ);

        /* ZtZ = Z(:,1:m)' * Z(:,1:m)  (m x n) * (n x m) = m x m */
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n,
                    1.0, fix->Z, n,
                    fix->Z, n,
                    0.0, ZtZ, m);

        /* Compute |I - ZtZ| / (m * ulp) */
        /* Subtract identity */
        for (int i = 0; i < m; i++) {
            ZtZ[i + i * m] -= 1.0;
        }

        /* Find max absolute entry */
        double max_val = 0.0;
        for (int i = 0; i < m * m; i++) {
            double a = fabs(ZtZ[i]);
            if (a > max_val) max_val = a;
        }

        double resid = max_val / ((double)m * ulp);
        assert_residual_ok(resid);

        free(ZtZ);
    }

    /* Verify all ifail entries are zero */
    for (int i = 0; i < m; i++) {
        assert_int_equal(fix->ifail[i], 0);
    }

    free(D_tmp);
    free(E_tmp);
    free(work_tmp);
}

/**
 * Test JOBZ='V', RANGE='I': index range eigenvalue selection.
 * Uses the 1-2-1 Toeplitz matrix.
 * Selects eigenvalues il through iu and verifies:
 *   - m == iu - il + 1
 *   - Eigenvalues match reference
 *   - Eigenvector orthogonality
 */
static void test_range_index(void** state)
{
    dstevx_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m;
    double ulp = dlamch("E");
    double safmin = dlamch("S");
    double abstol = 2.0 * safmin;

    if (n < 10) {
        skip_test("RANGE='I' test requires n >= 10");
    }

    /* Use il, iu as 1-based indices into the sorted eigenvalues */
    int il = n / 4 + 1;      /* 1-based lower index */
    int iu = 3 * n / 4;      /* 1-based upper index */
    int m_expected = iu - il + 1;

    /* Generate 1-2-1 Toeplitz matrix */
    rng_seed(fix->rng_state, fix->seed);
    generate_stevx_matrix(n, 2, fix->D, fix->E, fix->rng_state);

    /* Save original */
    memcpy(fix->AD, fix->D, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->AE, fix->E, (n - 1) * sizeof(double));
    }

    /* Compute reference eigenvalues with dstev */
    int n_alloc = (n > 0) ? n : 1;
    int e_alloc = (n > 1) ? n - 1 : 1;
    double* D_tmp = malloc(n_alloc * sizeof(double));
    double* E_tmp = malloc(e_alloc * sizeof(double));
    double* work_tmp = malloc((2 * n_alloc) * sizeof(double));
    assert_non_null(D_tmp);
    assert_non_null(E_tmp);
    assert_non_null(work_tmp);

    memcpy(D_tmp, fix->AD, n * sizeof(double));
    if (n > 1) {
        memcpy(E_tmp, fix->AE, (n - 1) * sizeof(double));
    }

    dstev("N", n, D_tmp, E_tmp, NULL, 1, work_tmp, &info);
    assert_info_success(info);

    /* Copy reference eigenvalues */
    memcpy(fix->W_ref, D_tmp, n * sizeof(double));

    /* Reset D, E from originals */
    memcpy(fix->D, fix->AD, n * sizeof(double));
    if (n > 1) {
        memcpy(fix->E, fix->AE, (n - 1) * sizeof(double));
    }

    /* Compute eigenvalues il..iu with eigenvectors */
    dstevx("V", "I", n, fix->D, fix->E,
           0.0, 0.0, il, iu, abstol,
           &m, fix->W, fix->Z, n,
           fix->work, fix->iwork, fix->ifail, &info);
    assert_info_success(info);
    assert_int_equal(m, m_expected);

    /* Compare eigenvalues with reference (il..iu are 1-based) */
    double max_eig = 0.0;
    for (int i = 0; i < m; i++) {
        double a = fabs(fix->W_ref[il - 1 + i]);
        if (a > max_eig) max_eig = a;
    }

    double max_diff = 0.0;
    for (int i = 0; i < m; i++) {
        double d = fabs(fix->W[i] - fix->W_ref[il - 1 + i]);
        if (d > max_diff) max_diff = d;
    }

    double denom = (double)n * ulp;
    if (max_eig > 0.0) denom *= max_eig;
    double resid = (denom > 0.0) ? max_diff / denom : 0.0;
    assert_residual_ok(resid);

    /* Verify eigenvector orthogonality: |I - Z(:,1:m)'*Z(:,1:m)| / (m*ulp) */
    if (m > 0) {
        double* ZtZ = malloc(m * m * sizeof(double));
        assert_non_null(ZtZ);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n,
                    1.0, fix->Z, n,
                    fix->Z, n,
                    0.0, ZtZ, m);

        /* Subtract identity and compute max norm */
        for (int i = 0; i < m; i++) {
            ZtZ[i + i * m] -= 1.0;
        }

        double max_val = 0.0;
        for (int i = 0; i < m * m; i++) {
            double a = fabs(ZtZ[i]);
            if (a > max_val) max_val = a;
        }

        double orth_resid = max_val / ((double)m * ulp);
        assert_residual_ok(orth_resid);

        free(ZtZ);
    }

    /* Verify all ifail entries are zero */
    for (int i = 0; i < m; i++) {
        assert_int_equal(fix->ifail[i], 0);
    }

    free(D_tmp);
    free(E_tmp);
    free(work_tmp);
}

/**
 * Test Sturm sequence verification via dstech.
 * After computing eigenvalues with dstevx, checks that they satisfy
 * Sturm sequence counts using the original tridiagonal.
 */
static void test_sturm(void** state)
{
    dstevx_fixture_t* fix = *state;
    int n = fix->n;
    int info;
    int m;

    /* Allocate workspace for dstech: length n */
    int n_alloc = (n > 0) ? n : 1;
    double* stech_work = malloc(n_alloc * sizeof(double));
    assert_non_null(stech_work);

    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Save original diagonal and off-diagonal */
        memcpy(fix->AD, fix->D, n * sizeof(double));
        if (n > 1) {
            memcpy(fix->AE, fix->E, (n - 1) * sizeof(double));
        }

        /* Compute all eigenvalues with dstevx (eigenvalues only) */
        dstevx("N", "A", n, fix->D, fix->E,
               0.0, 0.0, 0, 0, 0.0,
               &m, fix->W, NULL, 1,
               fix->work, fix->iwork, fix->ifail, &info);
        assert_info_success(info);
        assert_int_equal(m, n);

        /* Sturm sequence check:
         * dstech verifies eigenvalues using the original tridiagonal.
         * A = diagonal, B = off-diagonal, eig = computed eigenvalues */
        dstech(n, fix->AD, fix->AE, fix->W, THRESH, stech_work, &info);
        assert_info_success(info);
    }

    free(stech_work);
}

/*
 * Macro to generate test entries for a given size.
 * Creates 5 test cases: range_all_V, range_all_N, range_value, range_index, sturm.
 */
#define DSTEVX_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_range_all_V, setup_fn, dstevx_teardown), \
    cmocka_unit_test_setup_teardown(test_range_all_N, setup_fn, dstevx_teardown), \
    cmocka_unit_test_setup_teardown(test_range_value, setup_fn, dstevx_teardown), \
    cmocka_unit_test_setup_teardown(test_range_index, setup_fn, dstevx_teardown), \
    cmocka_unit_test_setup_teardown(test_sturm, setup_fn, dstevx_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSTEVX_TESTS(setup_2),
        DSTEVX_TESTS(setup_5),
        DSTEVX_TESTS(setup_10),
        DSTEVX_TESTS(setup_20),
        DSTEVX_TESTS(setup_50),
    };

    return cmocka_run_group_tests_name("dstevx", tests, NULL, NULL);
}
