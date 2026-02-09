/**
 * @file test_dsysvx.c
 * @brief CMocka test suite for dsysvx (expert symmetric indefinite driver).
 *
 * Tests the expert driver which provides condition estimation, iterative
 * refinement, and error bounds for symmetric indefinite systems A*X = B
 * using Bunch-Kaufman diagonal pivoting.
 *
 * Verification:
 * - Solution residual: ||B - A*X|| / (||A|| * ||X|| * N * EPS) < THRESH
 * - BERR should be small for well-conditioned matrices
 * - rcond should be reasonable (positive for non-singular)
 *
 * Configurations:
 *   Sizes: {5, 10, 20, 50}
 *   NRHS:  {1, 5}
 *   UPLO:  {'U', 'L'}
 *   FACT:  {'N', 'F'}
 *   Types: {1-6 (well-conditioned), 7-8 (ill-conditioned)}
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dsysvx(const char* fact, const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   double* const restrict AF, const int ldaf,
                   int* const restrict ipiv,
                   const double* const restrict B, const int ldb,
                   double* const restrict X, const int ldx,
                   double* rcond,
                   double* const restrict ferr, double* const restrict berr,
                   double* const restrict work, const int lwork,
                   int* const restrict iwork, int* info);

/* Norm computation */
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);
extern double dlange(const char* norm, const int m, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda;
    double* A;       /* Original matrix (const input to dsysvx) */
    double* AF;      /* Factored matrix output */
    double* B;       /* Right-hand side (const input to dsysvx) */
    double* X;       /* Solution output */
    double* XACT;    /* Known exact solution */
    int* ipiv;       /* Pivot indices */
    double* d;       /* Singular values for dlatms */
    double* ferr;    /* Forward error bounds */
    double* berr;    /* Backward error bounds */
    double* work;    /* Workspace */
    int* iwork;      /* Integer workspace */
    double* rwork;   /* Workspace for norm computation */
    double rcond;    /* Reciprocal condition number */
    uint64_t seed;
} dsysvx_fixture_t;

static uint64_t g_seed = 7600;

static int dsysvx_setup(void** state, int n, int nrhs)
{
    dsysvx_fixture_t* fix = malloc(sizeof(dsysvx_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->seed = g_seed++;

    int lwork = (n * 64 > 3 * n) ? n * 64 : 3 * n;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->AF = malloc(fix->lda * n * sizeof(double));
    fix->B = malloc(fix->lda * nrhs * sizeof(double));
    fix->X = malloc(fix->lda * nrhs * sizeof(double));
    fix->XACT = malloc(fix->lda * nrhs * sizeof(double));
    fix->ipiv = malloc(n * sizeof(int));
    fix->d = malloc(n * sizeof(double));
    fix->ferr = malloc(nrhs * sizeof(double));
    fix->berr = malloc(nrhs * sizeof(double));
    fix->work = malloc(lwork * sizeof(double));
    fix->iwork = malloc(n * sizeof(int));
    fix->rwork = malloc(n * sizeof(double));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->d);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->work);
    assert_non_null(fix->iwork);
    assert_non_null(fix->rwork);

    *state = fix;
    return 0;
}

static int dsysvx_teardown(void** state)
{
    dsysvx_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AF);
        free(fix->B);
        free(fix->X);
        free(fix->XACT);
        free(fix->ipiv);
        free(fix->d);
        free(fix->ferr);
        free(fix->berr);
        free(fix->work);
        free(fix->iwork);
        free(fix->rwork);
        free(fix);
    }
    return 0;
}

/* Size-specific setups: N x NRHS */
static int setup_5_1(void** state) { return dsysvx_setup(state, 5, 1); }
static int setup_5_5(void** state) { return dsysvx_setup(state, 5, 5); }
static int setup_10_1(void** state) { return dsysvx_setup(state, 10, 1); }
static int setup_10_5(void** state) { return dsysvx_setup(state, 10, 5); }
static int setup_20_1(void** state) { return dsysvx_setup(state, 20, 1); }
static int setup_20_5(void** state) { return dsysvx_setup(state, 20, 5); }
static int setup_50_1(void** state) { return dsysvx_setup(state, 50, 1); }
static int setup_50_5(void** state) { return dsysvx_setup(state, 50, 5); }

/**
 * Core test logic for fact="N": generate symmetric matrix, solve with dsysvx,
 * compute normalized residual.
 *
 * Returns the normalized residual ||B - A*X|| / (||A|| * ||X|| * N * EPS).
 * Sets *out_info to the info value returned by dsysvx.
 */
static double run_dsysvx_test(dsysvx_fixture_t* fix, int imat, const char* uplo,
                              int* out_info)
{
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;
    int info;

    dlatb4("DSY", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate known exact solution XACT = 1 + i/n */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->lda] = 1.0 + (double)i / fix->n;
        }
    }

    /* Compute B = A * XACT (A is symmetric) */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->XACT, fix->lda, 0.0, fix->B, fix->lda);

    /* Clear AF and ipiv so dsysvx computes fresh factorization */
    memset(fix->AF, 0, fix->lda * fix->n * sizeof(double));
    memset(fix->ipiv, 0, fix->n * sizeof(int));

    /* Call dsysvx with fact="N" */
    int lwork = fix->n * 64;
    dsysvx("N", uplo, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->AF, fix->lda, fix->ipiv,
           fix->B, fix->lda, fix->X, fix->lda,
           &fix->rcond, fix->ferr, fix->berr,
           fix->work, lwork, fix->iwork, &info);

    *out_info = info;

    if (info > 0 && info <= fix->n) {
        /* Singular matrix - cannot compute meaningful residual */
        return 0.0;
    }

    /* Compute residual: ||B - A*X|| / (||A|| * ||X|| * N * EPS) */
    double eps = DBL_EPSILON;

    double anorm_val = dlansy("1", uplo, fix->n, fix->A, fix->lda, fix->rwork);
    double xnorm = dlange("1", fix->n, fix->nrhs, fix->X, fix->lda, fix->rwork);

    /* Compute B - A*X in a temporary buffer */
    double* resid_vec = malloc(fix->lda * fix->nrhs * sizeof(double));
    assert_non_null(resid_vec);
    memcpy(resid_vec, fix->B, fix->lda * fix->nrhs * sizeof(double));

    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, -1.0, fix->A, fix->lda,
                fix->X, fix->lda, 1.0, resid_vec, fix->lda);

    double rnorm = dlange("1", fix->n, fix->nrhs, resid_vec, fix->lda, fix->rwork);
    free(resid_vec);

    double resid;
    if (anorm_val <= 0.0 || xnorm <= 0.0) {
        resid = (rnorm > 0.0) ? 1.0 / eps : 0.0;
    } else {
        resid = rnorm / (anorm_val * xnorm * fix->n * eps);
    }

    return resid;
}

/**
 * Test fact="F": use pre-computed factorization from a prior fact="N" call.
 *
 * Returns the normalized residual for the fact="F" solve.
 */
static double run_dsysvx_factored_test(dsysvx_fixture_t* fix, int imat, const char* uplo,
                                       int* out_info)
{
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;
    int info;

    dlatb4("DSY", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    dlatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate known exact solution XACT = 1 + i/n */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->lda] = 1.0 + (double)i / fix->n;
        }
    }

    /* Compute B = A * XACT */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->XACT, fix->lda, 0.0, fix->B, fix->lda);

    /* First: call dsysvx with fact="N" to obtain AF and ipiv */
    memset(fix->AF, 0, fix->lda * fix->n * sizeof(double));
    memset(fix->ipiv, 0, fix->n * sizeof(int));

    int lwork = fix->n * 64;
    double rcond_first;
    dsysvx("N", uplo, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->AF, fix->lda, fix->ipiv,
           fix->B, fix->lda, fix->X, fix->lda,
           &rcond_first, fix->ferr, fix->berr,
           fix->work, lwork, fix->iwork, &info);

    if (info > 0 && info <= fix->n) {
        *out_info = info;
        return 0.0;
    }
    assert_true(info == 0 || info == fix->n + 1);

    /* Second: call dsysvx with fact="F" using previously computed AF/ipiv */
    dsysvx("F", uplo, fix->n, fix->nrhs, fix->A, fix->lda,
           fix->AF, fix->lda, fix->ipiv,
           fix->B, fix->lda, fix->X, fix->lda,
           &fix->rcond, fix->ferr, fix->berr,
           fix->work, lwork, fix->iwork, &info);

    *out_info = info;

    if (info > 0 && info <= fix->n) {
        return 0.0;
    }

    /* Compute residual */
    double eps = DBL_EPSILON;
    double anorm_val = dlansy("1", uplo, fix->n, fix->A, fix->lda, fix->rwork);
    double xnorm = dlange("1", fix->n, fix->nrhs, fix->X, fix->lda, fix->rwork);

    double* resid_vec = malloc(fix->lda * fix->nrhs * sizeof(double));
    assert_non_null(resid_vec);
    memcpy(resid_vec, fix->B, fix->lda * fix->nrhs * sizeof(double));

    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, -1.0, fix->A, fix->lda,
                fix->X, fix->lda, 1.0, resid_vec, fix->lda);

    double rnorm = dlange("1", fix->n, fix->nrhs, resid_vec, fix->lda, fix->rwork);
    free(resid_vec);

    double resid;
    if (anorm_val <= 0.0 || xnorm <= 0.0) {
        resid = (rnorm > 0.0) ? 1.0 / eps : 0.0;
    } else {
        resid = rnorm / (anorm_val * xnorm * fix->n * eps);
    }

    return resid;
}

/*
 * Well-conditioned tests (types 1-6): UPLO='U', fact='N'
 */
static void test_dsysvx_upper_wellcond(void** state)
{
    dsysvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        int info;
        double resid = run_dsysvx_test(fix, imat, "U", &info);
        assert_true(info == 0 || info == fix->n + 1);
        if (info == 0) {
            assert_true(fix->rcond > 0.0);
        }
        assert_residual_ok(resid);
        /* BERR should be small for well-conditioned */
        for (int j = 0; j < fix->nrhs; j++) {
            assert_residual_ok(fix->berr[j] / DBL_EPSILON);
        }
    }
}

/*
 * Well-conditioned tests (types 1-6): UPLO='L', fact='N'
 */
static void test_dsysvx_lower_wellcond(void** state)
{
    dsysvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        int info;
        double resid = run_dsysvx_test(fix, imat, "L", &info);
        assert_true(info == 0 || info == fix->n + 1);
        if (info == 0) {
            assert_true(fix->rcond > 0.0);
        }
        assert_residual_ok(resid);
        for (int j = 0; j < fix->nrhs; j++) {
            assert_residual_ok(fix->berr[j] / DBL_EPSILON);
        }
    }
}

/*
 * Ill-conditioned tests (types 7-8): UPLO='U', fact='N'
 * Expect info = n+1 (ill-conditioned warning) but solution still computed.
 */
static void test_dsysvx_upper_illcond(void** state)
{
    dsysvx_fixture_t* fix = *state;
    for (int imat = 7; imat <= 8; imat++) {
        fix->seed = g_seed++;
        int info;
        double resid = run_dsysvx_test(fix, imat, "U", &info);
        /* info can be 0, n+1 (ill-conditioned), or 1..n (singular) */
        assert_true(info >= 0);
        if (info > 0 && info <= fix->n) {
            /* Singular - skip residual check */
            continue;
        }
        /* Solution was computed; residual should still be reasonable */
        assert_residual_ok(resid);
    }
}

/*
 * Ill-conditioned tests (types 7-8): UPLO='L', fact='N'
 */
static void test_dsysvx_lower_illcond(void** state)
{
    dsysvx_fixture_t* fix = *state;
    for (int imat = 7; imat <= 8; imat++) {
        fix->seed = g_seed++;
        int info;
        double resid = run_dsysvx_test(fix, imat, "L", &info);
        assert_true(info >= 0);
        if (info > 0 && info <= fix->n) {
            continue;
        }
        assert_residual_ok(resid);
    }
}

/*
 * Factored tests (fact='F'): UPLO='U', well-conditioned (types 1-6)
 */
static void test_dsysvx_upper_factored(void** state)
{
    dsysvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        int info;
        double resid = run_dsysvx_factored_test(fix, imat, "U", &info);
        if (info > 0 && info <= fix->n) {
            continue;
        }
        assert_true(info == 0 || info == fix->n + 1);
        assert_residual_ok(resid);
        for (int j = 0; j < fix->nrhs; j++) {
            assert_residual_ok(fix->berr[j] / DBL_EPSILON);
        }
    }
}

/*
 * Factored tests (fact='F'): UPLO='L', well-conditioned (types 1-6)
 */
static void test_dsysvx_lower_factored(void** state)
{
    dsysvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        int info;
        double resid = run_dsysvx_factored_test(fix, imat, "L", &info);
        if (info > 0 && info <= fix->n) {
            continue;
        }
        assert_true(info == 0 || info == fix->n + 1);
        assert_residual_ok(resid);
        for (int j = 0; j < fix->nrhs; j++) {
            assert_residual_ok(fix->berr[j] / DBL_EPSILON);
        }
    }
}

#define DSYSVX_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsysvx_upper_wellcond, setup_fn, dsysvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dsysvx_lower_wellcond, setup_fn, dsysvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dsysvx_upper_illcond, setup_fn, dsysvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dsysvx_lower_illcond, setup_fn, dsysvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dsysvx_upper_factored, setup_fn, dsysvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dsysvx_lower_factored, setup_fn, dsysvx_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* N=5, NRHS=1 */
        DSYSVX_TESTS(setup_5_1),
        /* N=5, NRHS=5 */
        DSYSVX_TESTS(setup_5_5),

        /* N=10, NRHS=1 */
        DSYSVX_TESTS(setup_10_1),
        /* N=10, NRHS=5 */
        DSYSVX_TESTS(setup_10_5),

        /* N=20, NRHS=1 */
        DSYSVX_TESTS(setup_20_1),
        /* N=20, NRHS=5 */
        DSYSVX_TESTS(setup_20_5),

        /* N=50, NRHS=1 */
        DSYSVX_TESTS(setup_50_1),
        /* N=50, NRHS=5 */
        DSYSVX_TESTS(setup_50_5),
    };

    return cmocka_run_group_tests_name("dsysvx", tests, NULL, NULL);
}
