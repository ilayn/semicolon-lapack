/**
 * @file test_dposvx.c
 * @brief CMocka test suite for dposvx (expert driver).
 *
 * Tests the expert driver dposvx which provides equilibration, factorization,
 * solve, condition estimation, iterative refinement, and error bounds.
 *
 * Verification: dpot02 (solve residual), dpot05 (error bounds), dget06 (condition).
 *
 * Tests FACT='N' and FACT='E', both UPLO='U' and UPLO='L', NRHS=1,2.
 */

#include "test_harness.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Routine under test */
extern void dposvx(const char* fact, const char* uplo, const int n, const int nrhs,
                   double* const restrict A, const int lda,
                   double* const restrict AF, const int ldaf,
                   char* equed, double* const restrict S,
                   double* const restrict B, const int ldb,
                   double* const restrict X, const int ldx,
                   double* rcond,
                   double* const restrict ferr, double* const restrict berr,
                   double* const restrict work, int* const restrict iwork,
                   int* info);

/* Verification routines */
extern void dpot02(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   const double* const restrict X, const int ldx,
                   double* const restrict B, const int ldb,
                   double* const restrict rwork, double* resid);
extern void dpot05(const char* uplo, const int n, const int nrhs,
                   const double* const restrict A, const int lda,
                   const double* const restrict B, const int ldb,
                   const double* const restrict X, const int ldx,
                   const double* const restrict XACT, const int ldxact,
                   const double* const restrict ferr,
                   const double* const restrict berr,
                   double* const restrict reslts);
extern double dget06(const double rcond, const double rcondc);

/* Utilities */
extern void dpotrf(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* info);
extern void dpotri(const char* uplo, const int n, double* const restrict A,
                   const int lda, int* info);
extern double dlamch(const char* cmach);
extern double dlansy(const char* norm, const char* uplo, const int n,
                     const double* const restrict A, const int lda,
                     double* const restrict work);

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
    int lda, ldb, ldx;
    double* A;       /* Original matrix */
    double* A_save;  /* Saved copy (dposvx modifies A when FACT='E') */
    double* AF;      /* Factored matrix */
    double* B;       /* RHS */
    double* B_save;  /* Saved RHS */
    double* X;       /* Computed solution */
    double* XACT;    /* Exact solution */
    double* S;       /* Scale factors */
    double* ferr;
    double* berr;
    double* d;
    double* work;
    double* rwork;
    int* iwork;
    uint64_t seed;
} dposvx_fixture_t;

static uint64_t g_seed = 5700;

static int dposvx_setup(void** state, int n, int nrhs)
{
    dposvx_fixture_t* fix = malloc(sizeof(dposvx_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->ldb = n;
    fix->ldx = n;
    fix->seed = g_seed++;

    fix->A = malloc(fix->lda * n * sizeof(double));
    fix->A_save = malloc(fix->lda * n * sizeof(double));
    fix->AF = malloc(fix->lda * n * sizeof(double));
    fix->B = malloc(fix->ldb * nrhs * sizeof(double));
    fix->B_save = malloc(fix->ldb * nrhs * sizeof(double));
    fix->X = malloc(fix->ldx * nrhs * sizeof(double));
    fix->XACT = malloc(fix->ldb * nrhs * sizeof(double));
    fix->S = malloc(n * sizeof(double));
    fix->ferr = malloc(nrhs * sizeof(double));
    fix->berr = malloc(nrhs * sizeof(double));
    fix->d = malloc(n * sizeof(double));
    fix->work = malloc(3 * n * sizeof(double));
    fix->rwork = malloc(n * sizeof(double));
    fix->iwork = malloc(n * sizeof(int));

    assert_non_null(fix->A);
    assert_non_null(fix->A_save);
    assert_non_null(fix->AF);
    assert_non_null(fix->B);
    assert_non_null(fix->B_save);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->S);
    assert_non_null(fix->ferr);
    assert_non_null(fix->berr);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->iwork);

    *state = fix;
    return 0;
}

static int dposvx_teardown(void** state)
{
    dposvx_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->A_save);
        free(fix->AF);
        free(fix->B);
        free(fix->B_save);
        free(fix->X);
        free(fix->XACT);
        free(fix->S);
        free(fix->ferr);
        free(fix->berr);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix->iwork);
        free(fix);
    }
    return 0;
}

static int setup_5(void** state) { return dposvx_setup(state, 5, 1); }
static int setup_10(void** state) { return dposvx_setup(state, 10, 1); }
static int setup_20(void** state) { return dposvx_setup(state, 20, 1); }
static int setup_5_nrhs2(void** state) { return dposvx_setup(state, 5, 2); }
static int setup_10_nrhs2(void** state) { return dposvx_setup(state, 10, 2); }
static int setup_20_nrhs2(void** state) { return dposvx_setup(state, 20, 2); }

/**
 * Core test logic: generate matrix, call dposvx with FACT='N', verify.
 */
static void run_dposvx_test_N(dposvx_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;
    int info;

    dlatb4("DPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    dlatms(fix->n, fix->n, &dist, fix->seed, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    /* Save original A */
    memcpy(fix->A_save, fix->A, fix->lda * fix->n * sizeof(double));

    /* Generate exact solution */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0 + (double)i / fix->n;
        }
    }

    /* Compute B = A * XACT */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_save, fix->B, fix->ldb * fix->nrhs * sizeof(double));

    /* Call dposvx with FACT='N' */
    char equed = 'N';
    double rcond;
    dposvx("N", uplo, fix->n, fix->nrhs, fix->A, fix->lda,
            fix->AF, fix->lda, &equed, fix->S,
            fix->B, fix->ldb, fix->X, fix->ldx, &rcond,
            fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    if (info > 0 && info <= fix->n) {
        /* Not positive definite - skip */
        return;
    }
    assert_true(info == 0 || info == fix->n + 1);

    /* Test 1: Verify solution residual */
    /* Restore B for dpot02 */
    memcpy(fix->B, fix->B_save, fix->ldb * fix->nrhs * sizeof(double));
    double resid;
    dpot02(uplo, fix->n, fix->nrhs, fix->A_save, fix->lda,
           fix->X, fix->ldx, fix->B, fix->ldb, fix->rwork, &resid);
    assert_residual_ok(resid);

    /* Test 2: Verify error bounds (only for well-conditioned) */
    if (imat <= 5) {
        double reslts[2];
        dpot05(uplo, fix->n, fix->nrhs, fix->A_save, fix->lda,
               fix->B_save, fix->ldb, fix->X, fix->ldx, fix->XACT, fix->ldb,
               fix->ferr, fix->berr, reslts);
        assert_residual_ok(reslts[0]);
        assert_residual_ok(reslts[1]);
    }

    /* Test 3: Verify condition number estimate */
    if (info == 0) {
        /* Compute true condition number via inverse */
        double* AINV = malloc(fix->lda * fix->n * sizeof(double));
        assert_non_null(AINV);
        memcpy(AINV, fix->AF, fix->lda * fix->n * sizeof(double));
        int info2;
        dpotri(uplo, fix->n, AINV, fix->lda, &info2);
        if (info2 == 0) {
            double anorm_1 = dlansy("1", uplo, fix->n, fix->A_save, fix->lda, fix->rwork);
            double ainvnm_1 = dlansy("1", uplo, fix->n, AINV, fix->lda, fix->rwork);
            double rcondc = (anorm_1 > 0.0 && ainvnm_1 > 0.0) ?
                            (1.0 / anorm_1) / ainvnm_1 : 0.0;
            if (rcondc > 0.0) {
                double ratio = dget06(rcond, rcondc);
                assert_residual_ok(ratio);
            }
        }
        free(AINV);
    }
}

/**
 * Test with FACT='E' (equilibrate then factor).
 */
static void run_dposvx_test_E(dposvx_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    double anorm_param, cndnum;
    int info;

    dlatb4("DPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    dlatms(fix->n, fix->n, &dist, fix->seed, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info);
    assert_int_equal(info, 0);

    /* Save original A */
    memcpy(fix->A_save, fix->A, fix->lda * fix->n * sizeof(double));

    /* Generate exact solution */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0 + (double)i / fix->n;
        }
    }

    /* Compute B = A * XACT */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_dsymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0, fix->B, fix->ldb);
    memcpy(fix->B_save, fix->B, fix->ldb * fix->nrhs * sizeof(double));

    /* Call dposvx with FACT='E' */
    char equed = 'N';
    double rcond;
    dposvx("E", uplo, fix->n, fix->nrhs, fix->A, fix->lda,
            fix->AF, fix->lda, &equed, fix->S,
            fix->B, fix->ldb, fix->X, fix->ldx, &rcond,
            fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    if (info > 0 && info <= fix->n) {
        return;
    }
    assert_true(info == 0 || info == fix->n + 1);

    /* Verify solution residual using original A and B */
    memcpy(fix->B, fix->B_save, fix->ldb * fix->nrhs * sizeof(double));
    double resid;
    dpot02(uplo, fix->n, fix->nrhs, fix->A_save, fix->lda,
           fix->X, fix->ldx, fix->B, fix->ldb, fix->rwork, &resid);
    assert_residual_ok(resid);
}

static void test_dposvx_N_wellcond_upper(void** state)
{
    dposvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dposvx_test_N(fix, imat, "U");
    }
}

static void test_dposvx_N_wellcond_lower(void** state)
{
    dposvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dposvx_test_N(fix, imat, "L");
    }
}

static void test_dposvx_N_illcond_upper(void** state)
{
    dposvx_fixture_t* fix = *state;
    for (int imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        run_dposvx_test_N(fix, imat, "U");
    }
}

static void test_dposvx_N_illcond_lower(void** state)
{
    dposvx_fixture_t* fix = *state;
    for (int imat = 6; imat <= 7; imat++) {
        fix->seed = g_seed++;
        run_dposvx_test_N(fix, imat, "L");
    }
}

static void test_dposvx_E_upper(void** state)
{
    dposvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dposvx_test_E(fix, imat, "U");
    }
}

static void test_dposvx_E_lower(void** state)
{
    dposvx_fixture_t* fix = *state;
    for (int imat = 1; imat <= 5; imat++) {
        fix->seed = g_seed++;
        run_dposvx_test_E(fix, imat, "L");
    }
}

#define DPOSVX_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dposvx_N_wellcond_upper, setup_fn, dposvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dposvx_N_wellcond_lower, setup_fn, dposvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dposvx_N_illcond_upper, setup_fn, dposvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dposvx_N_illcond_lower, setup_fn, dposvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dposvx_E_upper, setup_fn, dposvx_teardown), \
    cmocka_unit_test_setup_teardown(test_dposvx_E_lower, setup_fn, dposvx_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        DPOSVX_TESTS(setup_5),
        DPOSVX_TESTS(setup_10),
        DPOSVX_TESTS(setup_20),
        DPOSVX_TESTS(setup_5_nrhs2),
        DPOSVX_TESTS(setup_10_nrhs2),
        DPOSVX_TESTS(setup_20_nrhs2),
    };

    return cmocka_run_group_tests_name("dposvx", tests, NULL, NULL);
}
