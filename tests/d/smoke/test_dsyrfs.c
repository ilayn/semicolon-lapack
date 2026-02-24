/**
 * @file test_dsyrfs.c
 * @brief CMocka test suite for dsyrfs (iterative refinement for symmetric indefinite).
 *
 * Tests the iterative refinement routine dsyrfs which improves the computed
 * solution and provides error bounds for a symmetric indefinite system.
 *
 * Verification: check BERR is small for well-conditioned matrices, and compute
 * manual residual ||B - A*X||/(||A||*||X||*n*eps) after refinement.
 *
 * Tests both UPLO='U' and UPLO='L', with NRHS=1,2,5.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include "semicolon_cblas.h"

/* Routines under test */
/*
 * Test fixture
 */
typedef struct {
    INT n, nrhs;
    INT lda;
    f64* A;       /* Original matrix */
    f64* AF;      /* Factored matrix */
    f64* B;       /* RHS */
    f64* X;       /* Computed solution */
    f64* XACT;    /* Exact solution */
    f64* ferr;
    f64* berr;
    INT* ipiv;
    f64* d;
    f64* work;
    INT* iwork;
    f64* rwork;
    uint64_t seed;
} dsyrfs_fixture_t;

static uint64_t g_seed = 7500;

static int dsyrfs_setup(void** state, INT n, INT nrhs)
{
    dsyrfs_fixture_t* fix = malloc(sizeof(dsyrfs_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->seed = g_seed++;

    INT worksize = n * 64;
    if (worksize < 3 * n) worksize = 3 * n;

    fix->A = malloc(fix->lda * n * sizeof(f64));
    fix->AF = malloc(fix->lda * n * sizeof(f64));
    fix->B = malloc(fix->lda * nrhs * sizeof(f64));
    fix->X = malloc(fix->lda * nrhs * sizeof(f64));
    fix->XACT = malloc(fix->lda * nrhs * sizeof(f64));
    fix->ferr = malloc(nrhs * sizeof(f64));
    fix->berr = malloc(nrhs * sizeof(f64));
    fix->ipiv = malloc(n * sizeof(INT));
    fix->d = malloc(n * sizeof(f64));
    fix->work = malloc(worksize * sizeof(f64));
    fix->iwork = malloc(n * sizeof(INT));
    fix->rwork = malloc(n * sizeof(f64));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);
    assert_non_null(fix->rwork);

    *state = fix;
    return 0;
}

static int dsyrfs_teardown(void** state)
{
    dsyrfs_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->B);
        free(fix->X);
        free(fix->XACT);
        free(fix->ferr);
        free(fix->berr);
        free(fix->ipiv);
        free(fix->d);
        free(fix->work);
        free(fix->iwork);
        free(fix->rwork);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dsyrfs_setup(state, 5, 1); }
static int setup_10(void** state) { return dsyrfs_setup(state, 10, 1); }
static int setup_20(void** state) { return dsyrfs_setup(state, 20, 1); }
static int setup_5_nrhs2(void** state) { return dsyrfs_setup(state, 5, 2); }
static int setup_10_nrhs2(void** state) { return dsyrfs_setup(state, 10, 2); }
static int setup_20_nrhs2(void** state) { return dsyrfs_setup(state, 20, 2); }
static int setup_5_nrhs5(void** state) { return dsyrfs_setup(state, 5, 5); }
static int setup_10_nrhs5(void** state) { return dsyrfs_setup(state, 10, 5); }
static int setup_20_nrhs5(void** state) { return dsyrfs_setup(state, 20, 5); }

/**
 * Core test logic: generate symmetric indefinite matrix, solve, refine, verify.
 */
static void run_dsyrfs_test(dsyrfs_fixture_t* fix, INT imat, const char* uplo)
{
    char type, dist;
    INT kl, ku, mode;
    f64 anorm, cndnum;
    INT info;
    const f64 eps = DBL_EPSILON;

    dlatb4("DSY", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate exact solution */
    for (INT j = 0; j < fix->nrhs; j++) {
        for (INT i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->lda] = 1.0 + (f64)i / fix->n;
        }
    }

    /* Compute B = A * XACT using symmetric matrix-vector product */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->XACT, fix->lda, 0.0, fix->B, fix->lda);

    /* Factor A via Bunch-Kaufman: copy to AF first */
    memcpy(fix->AF, fix->A, fix->lda * fix->n * sizeof(f64));
    INT lwork = fix->n * 64;
    dsytrf(uplo, fix->n, fix->AF, fix->lda, fix->ipiv, fix->work, lwork, &info);
    assert_info_success(info);

    /* Solve AF*X = B via dsytrs */
    memcpy(fix->X, fix->B, fix->lda * fix->nrhs * sizeof(f64));
    dsytrs(uplo, fix->n, fix->nrhs, fix->AF, fix->lda, fix->ipiv,
           fix->X, fix->lda, &info);
    assert_info_success(info);

    /* Iterative refinement */
    dsyrfs(uplo, fix->n, fix->nrhs, fix->A, fix->lda, fix->AF, fix->lda,
           fix->ipiv, fix->B, fix->lda, fix->X, fix->lda,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Verify: BERR should be small for well-conditioned matrices */
    for (INT j = 0; j < fix->nrhs; j++) {
        assert_true(fix->berr[j] < 1.0);
    }

    /* Compute manual residual: ||B - A*X|| / (||A|| * ||X|| * n * eps) */
    /* Use rwork as temporary for residual R = B - A*X */
    for (INT j = 0; j < fix->nrhs; j++) {
        /* Copy B column to rwork */
        for (INT i = 0; i < fix->n; i++) {
            fix->rwork[i] = fix->B[i + j * fix->lda];
        }

        /* rwork = B - A*X: rwork -= A * X(:,j) */
        cblas_dsymv(CblasColMajor, cblas_uplo, fix->n,
                    -1.0, fix->A, fix->lda,
                    &fix->X[j * fix->lda], 1,
                    1.0, fix->rwork, 1);

        /* ||R|| */
        f64 rnorm = 0.0;
        for (INT i = 0; i < fix->n; i++) {
            f64 val = fabs(fix->rwork[i]);
            if (val > rnorm) rnorm = val;
        }

        /* ||A|| (infinity norm via row sums of full symmetric matrix) */
        f64 anorm_comp = 0.0;
        for (INT i = 0; i < fix->n; i++) {
            f64 rowsum = 0.0;
            for (INT k = 0; k < fix->n; k++) {
                rowsum += fabs(fix->A[i + k * fix->lda]);
            }
            if (rowsum > anorm_comp) anorm_comp = rowsum;
        }

        /* ||X(:,j)|| */
        f64 xnorm = 0.0;
        for (INT i = 0; i < fix->n; i++) {
            f64 val = fabs(fix->X[i + j * fix->lda]);
            if (val > xnorm) xnorm = val;
        }

        f64 denom = anorm_comp * xnorm * fix->n * eps;
        f64 resid = (denom > 0.0) ? rnorm / denom : rnorm;
        assert_residual_ok(resid);
    }
}

static void test_dsyrfs_wellcond_upper(void** state)
{
    dsyrfs_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        run_dsyrfs_test(fix, imat, "U");
    }
}

static void test_dsyrfs_wellcond_lower(void** state)
{
    dsyrfs_fixture_t* fix = *state;
    for (INT imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        run_dsyrfs_test(fix, imat, "L");
    }
}

#define DSYRFS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsyrfs_wellcond_upper, setup_fn, dsyrfs_teardown), \
    cmocka_unit_test_setup_teardown(test_dsyrfs_wellcond_lower, setup_fn, dsyrfs_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DSYRFS_TESTS(setup_5),
        DSYRFS_TESTS(setup_10),
        DSYRFS_TESTS(setup_20),
        DSYRFS_TESTS(setup_5_nrhs2),
        DSYRFS_TESTS(setup_10_nrhs2),
        DSYRFS_TESTS(setup_20_nrhs2),
        DSYRFS_TESTS(setup_5_nrhs5),
        DSYRFS_TESTS(setup_10_nrhs5),
        DSYRFS_TESTS(setup_20_nrhs5),
    };

    return cmocka_run_group_tests_name("dsyrfs", tests, NULL, NULL);
}
