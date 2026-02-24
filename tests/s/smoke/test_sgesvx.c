/**
 * @file test_sgesvx.c
 * @brief CMocka test suite for sgesvx (expert driver for general linear systems).
 *
 * Tests the expert driver which provides equilibration, condition estimation,
 * iterative refinement, and error bounds.
 *
 * Verification:
 * - sget02: ||B - A*X|| / (||A|| * ||X|| * eps) for solution quality
 * - sget04: ||X_computed - X_exact|| scaled by condition
 * - sget07: Tests error bounds (FERR) and backward error (BERR)
 *
 * Configurations:
 *   Sizes: {2, 3, 5, 10, 20}
 *   NRHS:  {1, 2, 5}
 *   FACT:  {'N', 'E'}
 *   TRANS: {'N', 'T'}
 *   Types: {4 (well-conditioned), 8 (ill-conditioned, N>=5 only)}
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include "semicolon_cblas.h"

/* Routine under test */
/* For test_factored: pre-factor using sgetrf */
/* Utilities */
/*
 * Test fixture: holds all allocated memory for a single test case.
 */
typedef struct {
    INT n;
    INT nrhs;
    INT lda, ldaf, ldb, ldx;
    f32 *A;        /* Original matrix */
    f32 *A_orig;   /* Pristine copy of A */
    f32 *AF;       /* Factored matrix */
    f32 *B;        /* Right-hand side (modified by sgesvx) */
    f32 *B_orig;   /* Pristine copy of B */
    f32 *X;        /* Solution */
    f32 *XACT;     /* Exact solution */
    f32 *d;        /* Singular values for slatms */
    f32 *R;        /* Row scale factors */
    f32 *C;        /* Column scale factors */
    f32 *work;     /* Workspace (max of slatms and sgesvx needs) */
    f32 *rwork;    /* Workspace for sget02 */
    f32 *ferr;     /* Forward error bounds */
    f32 *berr;     /* Backward error bounds */
    f32 *reslts;   /* Results from sget07 */
    INT* ipiv;        /* Pivot indices */
    INT* iwork;       /* Integer workspace for sgesvx */
    uint64_t seed;    /* RNG seed */
} dgesvx_fixture_t;

/* Global seed for test sequence reproducibility */
static uint64_t g_seed = 1729;

/**
 * Setup fixture: allocate memory for given dimensions.
 */
static int dgesvx_setup(void **state, INT n, INT nrhs)
{
    dgesvx_fixture_t *fix = malloc(sizeof(dgesvx_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldaf = n;
    fix->ldb = n;
    fix->ldx = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->A_orig = malloc(fix->lda * n * sizeof(f32));
    fix->AF = malloc(fix->ldaf * n * sizeof(f32));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->B_orig = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->X = malloc(fix->ldx * nrhs * sizeof(f32));
    fix->XACT = malloc(fix->ldx * nrhs * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->R = malloc(n * sizeof(f32));
    fix->C = malloc(n * sizeof(f32));
    fix->work = malloc(4 * n * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));
    fix->ferr = malloc(nrhs * sizeof(f32));
    fix->berr = malloc(nrhs * sizeof(f32));
    fix->reslts = malloc(2 * sizeof(f32));
    fix->ipiv = malloc(n * sizeof(INT));
    fix->iwork = malloc(n * sizeof(INT));

    assert_non_null(fix->A);
    assert_non_null(fix->A_orig);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->B_orig);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->d);
    assert_non_null(fix->R);
    assert_non_null(fix->C);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->reslts);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

/**
 * Teardown fixture: free all allocated memory.
 */
static int dgesvx_teardown(void **state)
{
    dgesvx_fixture_t *fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->A_orig);
        free(fix->AF);
        free(fix->B);
        free(fix->B_orig);
        free(fix->X);
        free(fix->XACT);
        free(fix->d);
        free(fix->R);
        free(fix->C);
        free(fix->work);
        free(fix->rwork);
        free(fix->ferr);
        free(fix->berr);
        free(fix->reslts);
        free(fix->ipiv);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

/* Size-specific setup wrappers: N x NRHS */
static int setup_2_1(void **state) { return dgesvx_setup(state, 2, 1); }
static int setup_2_2(void **state) { return dgesvx_setup(state, 2, 2); }
static int setup_3_1(void **state) { return dgesvx_setup(state, 3, 1); }
static int setup_3_2(void **state) { return dgesvx_setup(state, 3, 2); }
static int setup_5_1(void **state) { return dgesvx_setup(state, 5, 1); }
static int setup_5_2(void **state) { return dgesvx_setup(state, 5, 2); }
static int setup_5_5(void **state) { return dgesvx_setup(state, 5, 5); }
static int setup_10_1(void **state) { return dgesvx_setup(state, 10, 1); }
static int setup_10_2(void **state) { return dgesvx_setup(state, 10, 2); }
static int setup_10_5(void **state) { return dgesvx_setup(state, 10, 5); }
static int setup_20_1(void **state) { return dgesvx_setup(state, 20, 1); }
static int setup_20_2(void **state) { return dgesvx_setup(state, 20, 2); }
static int setup_20_5(void **state) { return dgesvx_setup(state, 20, 5); }

/* Sanity test setups */
static int setup_3_1_sanity(void **state) { return dgesvx_setup(state, 3, 1); }
static int setup_4_1_equil(void **state) { return dgesvx_setup(state, 4, 1); }
static int setup_3_2_factored(void **state) { return dgesvx_setup(state, 3, 2); }
static int setup_3_1_singular(void **state) { return dgesvx_setup(state, 3, 1); }
static int setup_3_1_transpose(void **state) { return dgesvx_setup(state, 3, 1); }

/**
 * Core test logic: generate matrix, solve with sgesvx, compute residuals.
 * Populates resid_02, resid_04, reslts[0] (FERR), reslts[1] (BERR).
 * Returns 0 on success, nonzero if the matrix was singular (skip residual checks).
 */
static INT run_dgesvx_test(dgesvx_fixture_t *fix, INT imat, const char* fact, const char* trans,
                           f32 *resid_02, f32 *resid_04, f32 *reslts)
{
    char type, dist;
    INT kl, ku, mode;
    f32 anorm_param, cndnum;
    INT info;
    INT n = fix->n;
    INT nrhs = fix->nrhs;

    /* Get matrix parameters */
    slatb4("SGE", imat, n, n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    /* Generate test matrix A */
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(n, n, &dist, &type, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);
    memcpy(fix->A_orig, fix->A, fix->lda * n * sizeof(f32));

    /* Generate known exact solution XACT */
    for (INT j = 0; j < nrhs; j++) {
        for (INT i = 0; i < n; i++) {
            fix->XACT[i + j * fix->ldx] = 1.0f + (f32)i / n + (f32)j / nrhs;
        }
    }

    /* Compute B = op(A) * XACT */
    CBLAS_TRANSPOSE cblas_trans = (trans[0] == 'N') ? CblasNoTrans : CblasTrans;
    cblas_sgemm(CblasColMajor, cblas_trans, CblasNoTrans,
                n, nrhs, n, 1.0f, fix->A, fix->lda, fix->XACT, fix->ldx,
                0.0f, fix->B, fix->ldb);
    memcpy(fix->B_orig, fix->B, fix->ldb * nrhs * sizeof(f32));

    /* Solve using sgesvx */
    char equed;
    f32 rcond;
    sgesvx(fact, trans, n, nrhs, fix->A, fix->lda, fix->AF, fix->ldaf,
           fix->ipiv, &equed, fix->R, fix->C, fix->B, fix->ldb,
           fix->X, fix->ldx, &rcond, fix->ferr, fix->berr,
           fix->work, fix->iwork, &info);

    if (info > 0 && info <= n) {
        /* Singular matrix - skip residual tests */
        return 1;
    }
    assert_true(info >= 0);

    /* Test 1: Solution residual using sget02 */
    f32 *B_copy = malloc(fix->ldb * nrhs * sizeof(f32));
    assert_non_null(B_copy);
    memcpy(B_copy, fix->B_orig, fix->ldb * nrhs * sizeof(f32));
    sget02(trans, n, n, nrhs, fix->A_orig, fix->lda, fix->X, fix->ldx,
           B_copy, fix->ldb, fix->rwork, resid_02);
    free(B_copy);

    /* Test 2: Solution accuracy using sget04 */
    f32 rcond_use = (rcond > 0.0f) ? rcond : 1.0f / cndnum;
    sget04(n, nrhs, fix->X, fix->ldx, fix->XACT, fix->ldx, rcond_use, resid_04);

    /* Test 3: Error bounds using sget07 */
    INT chkferr = (imat <= 4);
    sget07(trans, n, nrhs, fix->A_orig, fix->lda, fix->B_orig, fix->ldb,
           fix->X, fix->ldx, fix->XACT, fix->ldx,
           fix->ferr, chkferr, fix->berr, reslts);

    return 0;
}

/*
 * Sanity test: simple known system
 * A * x = b where solution is x = [1, 1, 1]'
 */
static void test_dgesvx_simple(void **state)
{
    dgesvx_fixture_t *fix = *state;
    INT n = 3;
    INT nrhs = 1;
    INT info;
    char equed;
    f32 rcond;

    /* System: A * x = b where solution is x = [1, 1, 1]' */
    f32 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};  /* Column-major */
    f32 B[3] = {4, 10, 24};

    memcpy(fix->A, A, 9 * sizeof(f32));
    memcpy(fix->B, B, 3 * sizeof(f32));

    sgesvx("N", "N", n, nrhs, fix->A, n, fix->AF, n, fix->ipiv, &equed,
           fix->R, fix->C, fix->B, n, fix->X, n, &rcond,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    assert_info_success(info);

    f32 tol = 1e-5f;
    assert_true(fabsf(fix->X[0] - 1.0f) < tol);
    assert_true(fabsf(fix->X[1] - 1.0f) < tol);
    assert_true(fabsf(fix->X[2] - 1.0f) < tol);
}

/*
 * Sanity test: equilibration with poorly scaled matrix
 */
static void test_dgesvx_equilibration(void **state)
{
    dgesvx_fixture_t *fix = *state;
    INT n = 4;
    INT nrhs = 1;
    INT info;
    char equed;
    f32 rcond;

    /* Poorly scaled diagonal matrix */
    f32 A[16] = {
        1e10, 0, 0, 0,
        0, 1e-10, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    f32 B[4] = {1e10, 1e-10, 1, 1};

    memcpy(fix->A, A, 16 * sizeof(f32));
    memcpy(fix->B, B, 4 * sizeof(f32));

    sgesvx("E", "N", n, nrhs, fix->A, n, fix->AF, n, fix->ipiv, &equed,
           fix->R, fix->C, fix->B, n, fix->X, n, &rcond,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    /* info = 0 or info = n+1 (ill-conditioned warning) are both acceptable */
    assert_true(info == 0 || info == n + 1);

    f32 tol = 1e-8;
    for (INT i = 0; i < n; i++) {
        assert_true(fabsf(fix->X[i] - 1.0f) < tol);
    }
}

/*
 * Sanity test: pre-factored matrix (FACT='F')
 */
static void test_dgesvx_factored(void **state)
{
    dgesvx_fixture_t *fix = *state;
    INT n = 3;
    INT nrhs = 2;
    INT info;
    char equed = 'N';
    f32 rcond;

    f32 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    f32 A_orig[9];
    memcpy(A_orig, A, 9 * sizeof(f32));

    /* Pre-factor using sgetrf */
    memcpy(fix->AF, A, 9 * sizeof(f32));
    sgetrf(n, n, fix->AF, n, fix->ipiv, &info);
    assert_info_success(info);

    /* Two RHS: first has solution [1,1,1], second has solution ~[0,1,1] */
    f32 B[6] = {4, 10, 24, 3, 9, 23};
    memcpy(fix->A, A_orig, 9 * sizeof(f32));
    memcpy(fix->B, B, 6 * sizeof(f32));

    sgesvx("F", "N", n, nrhs, fix->A, n, fix->AF, n, fix->ipiv, &equed,
           fix->R, fix->C, fix->B, n, fix->X, n, &rcond,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    assert_info_success(info);

    /* First RHS: solution should be [1, 1, 1] */
    f32 tol = 1e-5f;
    assert_true(fabsf(fix->X[0] - 1.0f) < tol);
    assert_true(fabsf(fix->X[1] - 1.0f) < tol);
    assert_true(fabsf(fix->X[2] - 1.0f) < tol);
}

/*
 * Sanity test: singular matrix detection
 */
static void test_dgesvx_singular(void **state)
{
    dgesvx_fixture_t *fix = *state;
    INT n = 3;
    INT nrhs = 1;
    INT info;
    char equed;
    f32 rcond;

    /* Singular matrix (all columns identical) */
    f32 A[9] = {1, 2, 3, 1, 2, 3, 1, 2, 3};
    f32 B[3] = {1, 2, 3};

    memcpy(fix->A, A, 9 * sizeof(f32));
    memcpy(fix->B, B, 3 * sizeof(f32));

    sgesvx("N", "N", n, nrhs, fix->A, n, fix->AF, n, fix->ipiv, &equed,
           fix->R, fix->C, fix->B, n, fix->X, n, &rcond,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    assert_info_singular(info);
}

/*
 * Sanity test: transpose solve
 */
static void test_dgesvx_transpose(void **state)
{
    dgesvx_fixture_t *fix = *state;
    INT n = 3;
    INT nrhs = 1;
    INT info;
    char equed;
    f32 rcond;

    f32 A[9] = {2, 4, 8, 1, 3, 7, 1, 3, 9};
    f32 xact[3] = {1.0f, 1.0f, 1.0f};

    memcpy(fix->A, A, 9 * sizeof(f32));

    /* Compute B = A' * xact */
    cblas_sgemv(CblasColMajor, CblasTrans, n, n, 1.0f, fix->A, n, xact, 1, 0.0f, fix->B, 1);

    sgesvx("N", "T", n, nrhs, fix->A, n, fix->AF, n, fix->ipiv, &equed,
           fix->R, fix->C, fix->B, n, fix->X, n, &rcond,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    assert_info_success(info);

    f32 tol = 1e-5f;
    assert_true(fabsf(fix->X[0] - 1.0f) < tol);
    assert_true(fabsf(fix->X[1] - 1.0f) < tol);
    assert_true(fabsf(fix->X[2] - 1.0f) < tol);
}

/*
 * Comprehensive test: FACT='N', TRANS='N', well-conditioned (type 4)
 */
static void test_dgesvx_fact_N_trans_N_type4(void **state)
{
    dgesvx_fixture_t *fix = *state;
    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 4, "N", "N", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    /* Type 4 is well-conditioned, check FERR */
    assert_residual_ok(reslts[0]);
    assert_residual_ok(reslts[1]);
}

/*
 * Comprehensive test: FACT='N', TRANS='N', ill-conditioned (type 8)
 */
static void test_dgesvx_fact_N_trans_N_type8(void **state)
{
    dgesvx_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip_test("type 8 requires N >= 5");
    }

    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 8, "N", "N", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    /* Type 8 is ill-conditioned, skip FERR check */
    assert_residual_ok(reslts[1]);
}

/*
 * Comprehensive test: FACT='N', TRANS='T', well-conditioned (type 4)
 */
static void test_dgesvx_fact_N_trans_T_type4(void **state)
{
    dgesvx_fixture_t *fix = *state;
    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 4, "N", "T", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    assert_residual_ok(reslts[0]);
    assert_residual_ok(reslts[1]);
}

/*
 * Comprehensive test: FACT='N', TRANS='T', ill-conditioned (type 8)
 */
static void test_dgesvx_fact_N_trans_T_type8(void **state)
{
    dgesvx_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip_test("type 8 requires N >= 5");
    }

    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 8, "N", "T", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    assert_residual_ok(reslts[1]);
}

/*
 * Comprehensive test: FACT='E', TRANS='N', well-conditioned (type 4)
 */
static void test_dgesvx_fact_E_trans_N_type4(void **state)
{
    dgesvx_fixture_t *fix = *state;
    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 4, "E", "N", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    assert_residual_ok(reslts[0]);
    assert_residual_ok(reslts[1]);
}

/*
 * Comprehensive test: FACT='E', TRANS='N', ill-conditioned (type 8)
 */
static void test_dgesvx_fact_E_trans_N_type8(void **state)
{
    dgesvx_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip_test("type 8 requires N >= 5");
    }

    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 8, "E", "N", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    assert_residual_ok(reslts[1]);
}

/*
 * Comprehensive test: FACT='E', TRANS='T', well-conditioned (type 4)
 */
static void test_dgesvx_fact_E_trans_T_type4(void **state)
{
    dgesvx_fixture_t *fix = *state;
    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 4, "E", "T", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    assert_residual_ok(reslts[0]);
    assert_residual_ok(reslts[1]);
}

/*
 * Comprehensive test: FACT='E', TRANS='T', ill-conditioned (type 8)
 */
static void test_dgesvx_fact_E_trans_T_type8(void **state)
{
    dgesvx_fixture_t *fix = *state;

    if (fix->n < 5) {
        skip_test("type 8 requires N >= 5");
    }

    f32 resid_02, resid_04;
    f32 reslts[2];

    fix->seed = g_seed++;
    INT rc = run_dgesvx_test(fix, 8, "E", "T", &resid_02, &resid_04, reslts);
    if (rc != 0) {
        skip_test("singular matrix");
    }

    assert_residual_ok(resid_02);
    assert_residual_ok(resid_04);
    assert_residual_ok(reslts[1]);
}

/*
 * Macro to generate all 8 test entries for a given size setup.
 * Creates tests for all FACT x TRANS x TYPE combinations.
 */
#define DGESVX_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_N_trans_N_type4, setup_fn, dgesvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_N_trans_N_type8, setup_fn, dgesvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_N_trans_T_type4, setup_fn, dgesvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_N_trans_T_type8, setup_fn, dgesvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_E_trans_N_type4, setup_fn, dgesvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_E_trans_N_type8, setup_fn, dgesvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_E_trans_T_type4, setup_fn, dgesvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dgesvx_fact_E_trans_T_type8, setup_fn, dgesvx_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Sanity checks */
        cmocka_unit_test_setup_teardown(test_dgesvx_simple, setup_3_1_sanity, dgesvx_teardown),
        cmocka_unit_test_setup_teardown(test_dgesvx_equilibration, setup_4_1_equil, dgesvx_teardown),
        cmocka_unit_test_setup_teardown(test_dgesvx_factored, setup_3_2_factored, dgesvx_teardown),
        cmocka_unit_test_setup_teardown(test_dgesvx_singular, setup_3_1_singular, dgesvx_teardown),
        cmocka_unit_test_setup_teardown(test_dgesvx_transpose, setup_3_1_transpose, dgesvx_teardown),

        /* Comprehensive: N=2, NRHS=1 */
        DGESVX_TESTS(setup_2_1),
        /* Comprehensive: N=2, NRHS=2 */
        DGESVX_TESTS(setup_2_2),

        /* Comprehensive: N=3, NRHS=1 */
        DGESVX_TESTS(setup_3_1),
        /* Comprehensive: N=3, NRHS=2 */
        DGESVX_TESTS(setup_3_2),

        /* Comprehensive: N=5, NRHS=1 */
        DGESVX_TESTS(setup_5_1),
        /* Comprehensive: N=5, NRHS=2 */
        DGESVX_TESTS(setup_5_2),
        /* Comprehensive: N=5, NRHS=5 */
        DGESVX_TESTS(setup_5_5),

        /* Comprehensive: N=10, NRHS=1 */
        DGESVX_TESTS(setup_10_1),
        /* Comprehensive: N=10, NRHS=2 */
        DGESVX_TESTS(setup_10_2),
        /* Comprehensive: N=10, NRHS=5 */
        DGESVX_TESTS(setup_10_5),

        /* Comprehensive: N=20, NRHS=1 */
        DGESVX_TESTS(setup_20_1),
        /* Comprehensive: N=20, NRHS=2 */
        DGESVX_TESTS(setup_20_2),
        /* Comprehensive: N=20, NRHS=5 */
        DGESVX_TESTS(setup_20_5),
    };

    return cmocka_run_group_tests_name("dgesvx", tests, NULL, NULL);
}
