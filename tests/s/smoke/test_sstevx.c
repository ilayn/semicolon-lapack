/**
 * @file test_sstevx.c
 * @brief CMocka test suite for sstevx (expert eigenvalue driver for symmetric tridiagonal).
 *
 * Tests the selective eigenvalue/eigenvector computation for symmetric
 * tridiagonal matrices. sstevx supports:
 *   - JOBZ='V' or 'N' (eigenvectors or eigenvalues only)
 *   - RANGE='A' (all), 'V' (value range), 'I' (index range)
 *
 * Verification:
 *   sstt21 checks |T - Z*D*Z'| / (|T|*n*ulp) and |I - Z*Z'| / (n*ulp)
 *   sstech verifies eigenvalues via Sturm sequence
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
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include "semicolon_cblas.h"

/* Routine under test */
/* Reference routine for eigenvalue comparison */
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
    f32* D;          /* diagonal for sstevx (overwritten) */
    f32* E;          /* off-diagonal for sstevx (overwritten) */
    f32* W;          /* eigenvalues output */
    f32* W_ref;      /* reference eigenvalues from sstev */
    f32* Z;          /* eigenvectors (n x n) */
    f32* work;       /* workspace: max(5*n, n*(n+1)) for sstevx + sstt21 */
    INT* iwork;         /* INT workspace (5*n) */
    INT* ifail;         /* convergence info (n) */
    f32* result;     /* sstt21 results (2 elements) */
    uint64_t seed;
    uint64_t rng_state[4];
} dstevx_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 3000;

/**
 * Setup fixture: allocate memory for given dimension.
 * Called before each test function.
 */
static int dstevx_setup(void** state, INT n)
{
    dstevx_fixture_t* fix = malloc(sizeof(dstevx_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->seed = g_seed++;

    INT n_alloc = (n > 0) ? n : 1;
    INT e_alloc = (n > 1) ? n - 1 : 1;
    INT work_stevx = 5 * n_alloc;
    INT work_stt21 = n_alloc * (n_alloc + 1);
    INT work_total = (work_stevx > work_stt21) ? work_stevx : work_stt21;

    fix->AD = malloc(n_alloc * sizeof(f32));
    fix->AE = malloc(e_alloc * sizeof(f32));
    fix->D = malloc(n_alloc * sizeof(f32));
    fix->E = malloc(e_alloc * sizeof(f32));
    fix->W = malloc(n_alloc * sizeof(f32));
    fix->W_ref = malloc(n_alloc * sizeof(f32));
    fix->Z = malloc(n_alloc * n_alloc * sizeof(f32));
    fix->work = malloc(work_total * sizeof(f32));
    fix->iwork = malloc(5 * n_alloc * sizeof(INT));
    fix->ifail = malloc(n_alloc * sizeof(INT));
    fix->result = malloc(2 * sizeof(f32));

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
static void generate_stevx_matrix(INT n, INT imat, f32* D, f32* E,
                                   uint64_t state[static 4])
{
    INT i;

    switch (imat) {
    case 1:
        /* Identity: D=1, E=0 */
        for (i = 0; i < n; i++) D[i] = 1.0f;
        for (i = 0; i < n - 1; i++) E[i] = 0.0f;
        break;

    case 2:
        /* 1-2-1 Toeplitz: D[i]=2, E[i]=1 */
        for (i = 0; i < n; i++) D[i] = 2.0f;
        for (i = 0; i < n - 1; i++) E[i] = 1.0f;
        break;

    case 3:
        /* Wilkinson-like: D[i]=|i - n/2|, E[i]=1 */
        for (i = 0; i < n; i++) D[i] = fabsf((f32)i - (f32)(n / 2));
        for (i = 0; i < n - 1; i++) E[i] = 1.0f;
        break;

    case 4:
        /* Random symmetric tridiagonal */
        for (i = 0; i < n; i++) D[i] = rng_uniform_symmetric_f32(state);
        for (i = 0; i < n - 1; i++) E[i] = rng_uniform_symmetric_f32(state);
        break;

    case 5:
        /* Graded diagonal: D[i]=2^(-i), E[i]=1 */
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
 * Test JOBZ='V', RANGE='A': all eigenvalues and eigenvectors.
 * Verifies via sstt21:
 *   result[0] = |T - Z*D*Z'| / (|T|*n*ulp)
 *   result[1] = |I - Z*Z'| / (n*ulp)
 * Also checks that all ifail[i] == 0.
 */
static void test_range_all_V(void** state)
{
    dstevx_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;
    INT m;

    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Save original for verification */
        memcpy(fix->AD, fix->D, n * sizeof(f32));
        if (n > 1) {
            memcpy(fix->AE, fix->E, (n - 1) * sizeof(f32));
        }

        /* Compute all eigenvalues and eigenvectors (abstol=0.0: fast path) */
        sstevx("V", "A", n, fix->D, fix->E,
               0.0f, 0.0f, 0, 0, 0.0f,
               &m, fix->W, fix->Z, n,
               fix->work, fix->iwork, fix->ifail, &info);
        assert_info_success(info);
        assert_int_equal(m, n);

        /* Verify all ifail entries are zero */
        for (INT i = 0; i < n; i++) {
            assert_int_equal(fix->ifail[i], 0);
        }

        /* Verify decomposition via sstt21:
         * kband=0 because W contains eigenvalues (diagonal),
         * SE is not referenced when kband=0, pass NULL */
        sstt21(n, 0, fix->AD, fix->AE, fix->W, NULL,
               fix->Z, n, fix->work, fix->result);

        assert_residual_ok(fix->result[0]);  /* decomposition residual */
        assert_residual_ok(fix->result[1]);  /* orthogonality */
    }
}

/**
 * Test JOBZ='N', RANGE='A': eigenvalues only.
 * Compares eigenvalues with reference from sstev.
 */
static void test_range_all_N(void** state)
{
    dstevx_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;
    INT m;
    f32 ulp = slamch("E");

    /* Allocate temporaries for sstev reference computation */
    INT n_alloc = (n > 0) ? n : 1;
    INT e_alloc = (n > 1) ? n - 1 : 1;
    f32* D_ref = malloc(n_alloc * sizeof(f32));
    f32* E_ref = malloc(e_alloc * sizeof(f32));
    f32* work_ref = malloc((2 * n_alloc) * sizeof(f32));
    assert_non_null(D_ref);
    assert_non_null(E_ref);
    assert_non_null(work_ref);

    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Copy for sstev reference */
        memcpy(D_ref, fix->D, n * sizeof(f32));
        if (n > 1) {
            memcpy(E_ref, fix->E, (n - 1) * sizeof(f32));
        }

        /* Compute reference eigenvalues with sstev */
        sstev("N", n, D_ref, E_ref, NULL, 1, work_ref, &info);
        assert_info_success(info);

        /* Regenerate for sstevx (D, E may be overwritten) */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Compute eigenvalues only with sstevx */
        sstevx("N", "A", n, fix->D, fix->E,
               0.0f, 0.0f, 0, 0, 0.0f,
               &m, fix->W, NULL, 1,
               fix->work, fix->iwork, fix->ifail, &info);
        assert_info_success(info);
        assert_int_equal(m, n);

        /* Compare eigenvalues: both should be sorted ascending.
         * Compute max|W[i] - D_ref[i]| / (n * max|D_ref| * ulp) */
        f32 max_eig = 0.0f;
        for (INT i = 0; i < n; i++) {
            f32 a = fabsf(D_ref[i]);
            if (a > max_eig) max_eig = a;
        }

        f32 max_diff = 0.0f;
        for (INT i = 0; i < n; i++) {
            f32 d = fabsf(fix->W[i] - D_ref[i]);
            if (d > max_diff) max_diff = d;
        }

        f32 denom = (f32)n * ulp;
        if (max_eig > 0.0f) denom *= max_eig;
        f32 resid = (denom > 0.0f) ? max_diff / denom : 0.0f;

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
    INT n = fix->n;
    INT info;
    INT m;
    f32 ulp = slamch("E");
    f32 safmin = slamch("S");
    f32 abstol = 2.0f * safmin;

    if (n < 10) {
        skip_test("RANGE='V' test requires n >= 10");
    }

    /* Generate 1-2-1 Toeplitz matrix */
    rng_seed(fix->rng_state, fix->seed);
    generate_stevx_matrix(n, 2, fix->D, fix->E, fix->rng_state);

    /* Save original */
    memcpy(fix->AD, fix->D, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->AE, fix->E, (n - 1) * sizeof(f32));
    }

    /* First, compute all eigenvalues with sstev to get reference */
    INT n_alloc = (n > 0) ? n : 1;
    INT e_alloc = (n > 1) ? n - 1 : 1;
    f32* D_tmp = malloc(n_alloc * sizeof(f32));
    f32* E_tmp = malloc(e_alloc * sizeof(f32));
    f32* work_tmp = malloc((2 * n_alloc) * sizeof(f32));
    assert_non_null(D_tmp);
    assert_non_null(E_tmp);
    assert_non_null(work_tmp);

    memcpy(D_tmp, fix->AD, n * sizeof(f32));
    if (n > 1) {
        memcpy(E_tmp, fix->AE, (n - 1) * sizeof(f32));
    }

    sstev("N", n, D_tmp, E_tmp, NULL, 1, work_tmp, &info);
    assert_info_success(info);

    /* Copy reference eigenvalues */
    memcpy(fix->W_ref, D_tmp, n * sizeof(f32));

    /* Pick value range: middle subset */
    INT idx_lo = n / 4;       /* roughly 25% from bottom */
    INT idx_hi = 3 * n / 4;   /* roughly 75% from bottom */
    f32 vl = fix->W_ref[idx_lo];
    f32 vu = fix->W_ref[idx_hi];

    /* Reset D, E from originals */
    memcpy(fix->D, fix->AD, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->E, fix->AE, (n - 1) * sizeof(f32));
    }

    /* Compute eigenvalues in (vl, vu] with eigenvectors */
    sstevx("V", "V", n, fix->D, fix->E,
           vl, vu, 0, 0, abstol,
           &m, fix->W, fix->Z, n,
           fix->work, fix->iwork, fix->ifail, &info);
    assert_info_success(info);

    /* Verify each eigenvalue is in (vl, vu] */
    for (INT i = 0; i < m; i++) {
        assert_true(fix->W[i] > vl);
        assert_true(fix->W[i] <= vu);
    }

    /* Verify eigenvector orthogonality: compute Z(:,1:m)' * Z(:,1:m) into
     * an m x m matrix, then check |I - Z'Z| / (m * ulp) */
    if (m > 0) {
        f32* ZtZ = malloc(m * m * sizeof(f32));
        assert_non_null(ZtZ);

        /* ZtZ = Z(:,1:m)' * Z(:,1:m)  (m x n) * (n x m) = m x m */
        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n,
                    1.0f, fix->Z, n,
                    fix->Z, n,
                    0.0f, ZtZ, m);

        /* Compute |I - ZtZ| / (m * ulp) */
        /* Subtract identity */
        for (INT i = 0; i < m; i++) {
            ZtZ[i + i * m] -= 1.0f;
        }

        /* Find max absolute entry */
        f32 max_val = 0.0f;
        for (INT i = 0; i < m * m; i++) {
            f32 a = fabsf(ZtZ[i]);
            if (a > max_val) max_val = a;
        }

        f32 resid = max_val / ((f32)m * ulp);
        assert_residual_ok(resid);

        free(ZtZ);
    }

    /* Verify all ifail entries are zero */
    for (INT i = 0; i < m; i++) {
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
    INT n = fix->n;
    INT info;
    INT m;
    f32 ulp = slamch("E");
    f32 safmin = slamch("S");
    f32 abstol = 2.0f * safmin;

    if (n < 10) {
        skip_test("RANGE='I' test requires n >= 10");
    }

    /* Use il, iu as 0-based indices into the sorted eigenvalues */
    INT il = n / 4;           /* 0-based lower index */
    INT iu = 3 * n / 4 - 1;  /* 0-based upper index */
    INT m_expected = iu - il + 1;

    /* Generate 1-2-1 Toeplitz matrix */
    rng_seed(fix->rng_state, fix->seed);
    generate_stevx_matrix(n, 2, fix->D, fix->E, fix->rng_state);

    /* Save original */
    memcpy(fix->AD, fix->D, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->AE, fix->E, (n - 1) * sizeof(f32));
    }

    /* Compute reference eigenvalues with sstev */
    INT n_alloc = (n > 0) ? n : 1;
    INT e_alloc = (n > 1) ? n - 1 : 1;
    f32* D_tmp = malloc(n_alloc * sizeof(f32));
    f32* E_tmp = malloc(e_alloc * sizeof(f32));
    f32* work_tmp = malloc((2 * n_alloc) * sizeof(f32));
    assert_non_null(D_tmp);
    assert_non_null(E_tmp);
    assert_non_null(work_tmp);

    memcpy(D_tmp, fix->AD, n * sizeof(f32));
    if (n > 1) {
        memcpy(E_tmp, fix->AE, (n - 1) * sizeof(f32));
    }

    sstev("N", n, D_tmp, E_tmp, NULL, 1, work_tmp, &info);
    assert_info_success(info);

    /* Copy reference eigenvalues */
    memcpy(fix->W_ref, D_tmp, n * sizeof(f32));

    /* Reset D, E from originals */
    memcpy(fix->D, fix->AD, n * sizeof(f32));
    if (n > 1) {
        memcpy(fix->E, fix->AE, (n - 1) * sizeof(f32));
    }

    /* Compute eigenvalues il..iu with eigenvectors */
    sstevx("V", "I", n, fix->D, fix->E,
           0.0f, 0.0f, il, iu, abstol,
           &m, fix->W, fix->Z, n,
           fix->work, fix->iwork, fix->ifail, &info);
    assert_info_success(info);
    assert_int_equal(m, m_expected);

    /* Compare eigenvalues with reference (il..iu are 0-based) */
    f32 max_eig = 0.0f;
    for (INT i = 0; i < m; i++) {
        f32 a = fabsf(fix->W_ref[il + i]);
        if (a > max_eig) max_eig = a;
    }

    f32 max_diff = 0.0f;
    for (INT i = 0; i < m; i++) {
        f32 d = fabsf(fix->W[i] - fix->W_ref[il + i]);
        if (d > max_diff) max_diff = d;
    }

    f32 denom = (f32)n * ulp;
    if (max_eig > 0.0f) denom *= max_eig;
    f32 resid = (denom > 0.0f) ? max_diff / denom : 0.0f;
    assert_residual_ok(resid);

    /* Verify eigenvector orthogonality: |I - Z(:,1:m)'*Z(:,1:m)| / (m*ulp) */
    if (m > 0) {
        f32* ZtZ = malloc(m * m * sizeof(f32));
        assert_non_null(ZtZ);

        cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                    m, m, n,
                    1.0f, fix->Z, n,
                    fix->Z, n,
                    0.0f, ZtZ, m);

        /* Subtract identity and compute max norm */
        for (INT i = 0; i < m; i++) {
            ZtZ[i + i * m] -= 1.0f;
        }

        f32 max_val = 0.0f;
        for (INT i = 0; i < m * m; i++) {
            f32 a = fabsf(ZtZ[i]);
            if (a > max_val) max_val = a;
        }

        f32 orth_resid = max_val / ((f32)m * ulp);
        assert_residual_ok(orth_resid);

        free(ZtZ);
    }

    /* Verify all ifail entries are zero */
    for (INT i = 0; i < m; i++) {
        assert_int_equal(fix->ifail[i], 0);
    }

    free(D_tmp);
    free(E_tmp);
    free(work_tmp);
}

/**
 * Test Sturm sequence verification via sstech.
 * After computing eigenvalues with sstevx, checks that they satisfy
 * Sturm sequence counts using the original tridiagonal.
 */
static void test_sturm(void** state)
{
    dstevx_fixture_t* fix = *state;
    INT n = fix->n;
    INT info;
    INT m;

    /* Allocate workspace for sstech: length n */
    INT n_alloc = (n > 0) ? n : 1;
    f32* stech_work = malloc(n_alloc * sizeof(f32));
    assert_non_null(stech_work);

    for (INT imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;

        /* Generate test matrix */
        rng_seed(fix->rng_state, fix->seed);
        generate_stevx_matrix(n, imat, fix->D, fix->E, fix->rng_state);

        /* Save original diagonal and off-diagonal */
        memcpy(fix->AD, fix->D, n * sizeof(f32));
        if (n > 1) {
            memcpy(fix->AE, fix->E, (n - 1) * sizeof(f32));
        }

        /* Compute all eigenvalues with sstevx (eigenvalues only) */
        sstevx("N", "A", n, fix->D, fix->E,
               0.0f, 0.0f, 0, 0, 0.0f,
               &m, fix->W, NULL, 1,
               fix->work, fix->iwork, fix->ifail, &info);
        assert_info_success(info);
        assert_int_equal(m, n);

        /* Sturm sequence check:
         * sstech verifies eigenvalues using the original tridiagonal.
         * A = diagonal, B = off-diagonal, eig = computed eigenvalues */
        sstech(n, fix->AD, fix->AE, fix->W, THRESH, stech_work, &info);
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
