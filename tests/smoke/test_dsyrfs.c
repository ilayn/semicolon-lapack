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

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routines under test */
extern void dsytrf(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* const restrict ipiv,
                   double* const restrict work, const int lwork, int* info);
extern void dsytrs(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   const int* const restrict ipiv,
                   double* const restrict B, const int ldb, int* info);
extern void dsyrfs(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   const double* const restrict AF, const int ldaf,
                   const int* const restrict ipiv,
                   const double* const restrict B, const int ldb,
                   double* const restrict X, const int ldx,
                   double* const restrict ferr, double* const restrict berr,
                   double* const restrict work, int* const restrict iwork,
                   int* info);

/* Matrix generation */
extern void dlatb4(const char* path, const int imat, const int m, const int n,
                   char* type, int* kl, int* ku, double* anorm, int* mode,
                   double* cndnum, char* dist);
extern void dlatms(const int m, const int n, const char* dist,
                   uint64_t seed, const char* sym, double* d,
                   const int mode, const double cond, const double dmax,
                   const int kl, const int ku, const char* pack,
                   double* A, const int lda, double* work, int* info);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda;
    double* A;       /* Original matrix */
    double* AF;      /* Factored matrix */
    double* B;       /* RHS */
    double* X;       /* Computed solution */
    double* XACT;    /* Exact solution */
    double* ferr;
    double* berr;
    int* ipiv;
    double* d;
    double* work;
    int* iwork;
    double* rwork;
    uint64_t seed;
} dsyrfs_fixture_t;

static uint64_t g_seed = 7500;

static int dsyrfs_setup(void** state, int n, int nrhs)
{
    dsyrfs_fixture_t* fix = malloc(sizeof(dsyrfs_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->seed = g_seed++;

    int worksize = n * 64;
    if (worksize < 3 * n) worksize = 3 * n;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AF = malloc(fix->lda * n * sizeof(double));
    fix->B = malloc(fix->lda * nrhs * sizeof(double));
    fix->X = malloc(fix->lda * nrhs * sizeof(double));
    fix->XACT = malloc(fix->lda * nrhs * sizeof(double));
    fix->ferr = malloc(nrhs * sizeof(double));
    fix->berr = malloc(nrhs * sizeof(double));
    fix->ipiv = malloc(n * sizeof(int));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(worksize * sizeof(double));
    fix->iwork = malloc(n * sizeof(int));
    fix->rwork = malloc(n * sizeof(double));

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
static void run_dsyrfs_test(dsyrfs_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    double anorm, cndnum;
    int info;
    const double eps = DBL_EPSILON;

    dlatb4("DSY", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    dlatms(fix->n, fix->n, &dist, fix->seed, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    /* Generate exact solution */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->lda] = 1.0 + (double)i / fix->n;
        }
    }

    /* Compute B = A * XACT using symmetric matrix-vector product */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->XACT, fix->lda, 0.0, fix->B, fix->lda);

    /* Factor A via Bunch-Kaufman: copy to AF first */
    memcpy(fix->AF, fix->A, fix->lda * fix->n * sizeof(double));
    int lwork = fix->n * 64;
    dsytrf(uplo, fix->n, fix->AF, fix->lda, fix->ipiv, fix->work, lwork, &info);
    assert_info_success(info);

    /* Solve AF*X = B via dsytrs */
    memcpy(fix->X, fix->B, fix->lda * fix->nrhs * sizeof(double));
    dsytrs(uplo, fix->n, fix->nrhs, fix->AF, fix->lda, fix->ipiv,
           fix->X, fix->lda, &info);
    assert_info_success(info);

    /* Iterative refinement */
    dsyrfs(uplo, fix->n, fix->nrhs, fix->A, fix->lda, fix->AF, fix->lda,
           fix->ipiv, fix->B, fix->lda, fix->X, fix->lda,
           fix->ferr, fix->berr, fix->work, fix->iwork, &info);
    assert_info_success(info);

    /* Verify: BERR should be small for well-conditioned matrices */
    for (int j = 0; j < fix->nrhs; j++) {
        assert_true(fix->berr[j] < 1.0);
    }

    /* Compute manual residual: ||B - A*X|| / (||A|| * ||X|| * n * eps) */
    /* Use rwork as temporary for residual R = B - A*X */
    for (int j = 0; j < fix->nrhs; j++) {
        /* Copy B column to rwork */
        for (int i = 0; i < fix->n; i++) {
            fix->rwork[i] = fix->B[i + j * fix->lda];
        }

        /* rwork = B - A*X: rwork -= A * X(:,j) */
        cblas_dsymv(CblasColMajor, cblas_uplo, fix->n,
                    -1.0, fix->A, fix->lda,
                    &fix->X[j * fix->lda], 1,
                    1.0, fix->rwork, 1);

        /* ||R|| */
        double rnorm = 0.0;
        for (int i = 0; i < fix->n; i++) {
            double val = fabs(fix->rwork[i]);
            if (val > rnorm) rnorm = val;
        }

        /* ||A|| (infinity norm via row sums of full symmetric matrix) */
        double anorm_comp = 0.0;
        for (int i = 0; i < fix->n; i++) {
            double rowsum = 0.0;
            for (int k = 0; k < fix->n; k++) {
                rowsum += fabs(fix->A[i + k * fix->lda]);
            }
            if (rowsum > anorm_comp) anorm_comp = rowsum;
        }

        /* ||X(:,j)|| */
        double xnorm = 0.0;
        for (int i = 0; i < fix->n; i++) {
            double val = fabs(fix->X[i + j * fix->lda]);
            if (val > xnorm) xnorm = val;
        }

        double denom = anorm_comp * xnorm * fix->n * eps;
        double resid = (denom > 0.0) ? rnorm / denom : rnorm;
        assert_residual_ok(resid);
    }
}

static void test_dsyrfs_wellcond_upper(void** state)
{
    dsyrfs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        run_dsyrfs_test(fix, imat, "U");
    }
}

static void test_dsyrfs_wellcond_lower(void** state)
{
    dsyrfs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
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
