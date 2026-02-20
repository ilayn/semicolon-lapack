/**
 * @file test_sposvx.c
 * @brief CMocka test suite for sposvx (expert driver).
 *
 * Tests the expert driver sposvx which provides equilibration, factorization,
 * solve, condition estimation, iterative refinement, and error bounds.
 *
 * Verification: spot02 (solve residual), spot05 (error bounds), sget06 (condition).
 *
 * Tests FACT='N' and FACT='E', both UPLO='U' and UPLO='L', NRHS=1,2.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

/* Routine under test */
extern void sposvx(const char* fact, const char* uplo, const int n, const int nrhs,
                   f32* const restrict A, const int lda,
                   f32* const restrict AF, const int ldaf,
                   char* equed, f32* const restrict S,
                   f32* const restrict B, const int ldb,
                   f32* const restrict X, const int ldx,
                   f32* rcond,
                   f32* const restrict ferr, f32* const restrict berr,
                   f32* const restrict work, int* const restrict iwork,
                   int* info);

/* Utilities */
extern void spotrf(const char* uplo, const int n, f32* const restrict A,
                   const int lda, int* info);
extern void spotri(const char* uplo, const int n, f32* const restrict A,
                   const int lda, int* info);
extern f32 slamch(const char* cmach);
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda, ldb, ldx;
    f32* A;       /* Original matrix */
    f32* A_save;  /* Saved copy (sposvx modifies A when FACT='E') */
    f32* AF;      /* Factored matrix */
    f32* B;       /* RHS */
    f32* B_save;  /* Saved RHS */
    f32* X;       /* Computed solution */
    f32* XACT;    /* Exact solution */
    f32* S;       /* Scale factors */
    f32* ferr;
    f32* berr;
    f32* d;
    f32* work;
    f32* rwork;
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

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->A_save = malloc(fix->lda * n * sizeof(f32));
    fix->AF = malloc(fix->lda * n * sizeof(f32));
    fix->B = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->B_save = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->X = malloc(fix->ldx * nrhs * sizeof(f32));
    fix->XACT = malloc(fix->ldb * nrhs * sizeof(f32));
    fix->S = malloc(n * sizeof(f32));
    fix->ferr = malloc(nrhs * sizeof(f32));
    fix->berr = malloc(nrhs * sizeof(f32));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(3 * n * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));
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
 * Core test logic: generate matrix, call sposvx with FACT='N', verify.
 */
static void run_dposvx_test_N(dposvx_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm_param, cndnum;
    int info;

    slatb4("SPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Save original A */
    memcpy(fix->A_save, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Generate exact solution */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0f + (f32)i / fix->n;
        }
    }

    /* Compute B = A * XACT */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0f, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0f, fix->B, fix->ldb);
    memcpy(fix->B_save, fix->B, fix->ldb * fix->nrhs * sizeof(f32));

    /* Call sposvx with FACT='N' */
    char equed = 'N';
    f32 rcond;
    sposvx("N", uplo, fix->n, fix->nrhs, fix->A, fix->lda,
            fix->AF, fix->lda, &equed, fix->S,
            fix->B, fix->ldb, fix->X, fix->ldx, &rcond,
            fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    if (info > 0 && info <= fix->n) {
        /* Not positive definite - skip */
        return;
    }
    assert_true(info == 0 || info == fix->n + 1);

    /* Test 1: Verify solution residual */
    /* Restore B for spot02 */
    memcpy(fix->B, fix->B_save, fix->ldb * fix->nrhs * sizeof(f32));
    f32 resid;
    spot02(uplo, fix->n, fix->nrhs, fix->A_save, fix->lda,
           fix->X, fix->ldx, fix->B, fix->ldb, fix->rwork, &resid);
    assert_residual_ok(resid);

    /* Test 2: Verify error bounds (only for well-conditioned) */
    if (imat <= 5) {
        f32 reslts[2];
        spot05(uplo, fix->n, fix->nrhs, fix->A_save, fix->lda,
               fix->B_save, fix->ldb, fix->X, fix->ldx, fix->XACT, fix->ldb,
               fix->ferr, fix->berr, reslts);
        assert_residual_ok(reslts[0]);
        assert_residual_ok(reslts[1]);
    }

    /* Test 3: Verify condition number estimate */
    if (info == 0) {
        /* Compute true condition number via inverse */
        f32* AINV = malloc(fix->lda * fix->n * sizeof(f32));
        assert_non_null(AINV);
        memcpy(AINV, fix->AF, fix->lda * fix->n * sizeof(f32));
        int info2;
        spotri(uplo, fix->n, AINV, fix->lda, &info2);
        if (info2 == 0) {
            f32 anorm_1 = slansy("1", uplo, fix->n, fix->A_save, fix->lda, fix->rwork);
            f32 ainvnm_1 = slansy("1", uplo, fix->n, AINV, fix->lda, fix->rwork);
            f32 rcondc = (anorm_1 > 0.0f && ainvnm_1 > 0.0f) ?
                            (1.0f / anorm_1) / ainvnm_1 : 0.0f;
            if (rcondc > 0.0f) {
                f32 ratio = sget06(rcond, rcondc);
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
    f32 anorm_param, cndnum;
    int info;

    slatb4("SPO", imat, fix->n, fix->n, &type, &kl, &ku, &anorm_param, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm_param,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Save original A */
    memcpy(fix->A_save, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Generate exact solution */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->ldb] = 1.0f + (f32)i / fix->n;
        }
    }

    /* Compute B = A * XACT */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0f, fix->A, fix->lda,
                fix->XACT, fix->ldb, 0.0f, fix->B, fix->ldb);
    memcpy(fix->B_save, fix->B, fix->ldb * fix->nrhs * sizeof(f32));

    /* Call sposvx with FACT='E' */
    char equed = 'N';
    f32 rcond;
    sposvx("E", uplo, fix->n, fix->nrhs, fix->A, fix->lda,
            fix->AF, fix->lda, &equed, fix->S,
            fix->B, fix->ldb, fix->X, fix->ldx, &rcond,
            fix->ferr, fix->berr, fix->work, fix->iwork, &info);

    if (info > 0 && info <= fix->n) {
        return;
    }
    assert_true(info == 0 || info == fix->n + 1);

    /* Verify solution residual using original A and B */
    memcpy(fix->B, fix->B_save, fix->ldb * fix->nrhs * sizeof(f32));
    f32 resid;
    spot02(uplo, fix->n, fix->nrhs, fix->A_save, fix->lda,
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
