/**
 * @file test_ssysv.c
 * @brief CMocka test suite for ssysv (symmetric indefinite combined solve).
 *
 * Tests the combined driver ssysv which factors A using Bunch-Kaufman
 * pivoting (ssytrf) and solves A*X = B (ssytrs) in one call.
 *
 * Verification: ||B - A*X|| / (||A|| * ||X|| * N * EPS)
 *
 * Tests both UPLO='U' and UPLO='L', sizes 5, 10, 20, 50, matrix types 1-6.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

/* Routine under test */
extern void ssysv(const char* uplo, const int n, const int nrhs,
                  f32* const restrict A, const int lda,
                  int* const restrict ipiv,
                  f32* const restrict B, const int ldb,
                  f32* const restrict work, const int lwork,
                  int* info);

/* Norm computation */
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* const restrict A, const int lda,
                     f32* const restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda;
    f32* A;       /* Original matrix */
    f32* AFAC;    /* Matrix for factorization (overwritten by ssysv) */
    f32* B;       /* Original RHS */
    f32* X;       /* RHS copy (overwritten with solution by ssysv) */
    f32* XACT;    /* Known exact solution */
    int* ipiv;       /* Pivot indices */
    f32* d;       /* Singular values for slatms */
    f32* work;    /* Workspace */
    f32* rwork;   /* Workspace for norm computation */
    uint64_t seed;
} dsysv_fixture_t;

static uint64_t g_seed = 7200;

static int dsysv_setup(void** state, int n, int nrhs)
{
    dsysv_fixture_t* fix = malloc(sizeof(dsysv_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->seed = g_seed++;

    int lwork = n * 64;

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->AFAC = malloc(fix->lda * n * sizeof(f32));
    fix->B = malloc(fix->lda * nrhs * sizeof(f32));
    fix->X = malloc(fix->lda * nrhs * sizeof(f32));
    fix->XACT = malloc(fix->lda * nrhs * sizeof(f32));
    fix->ipiv = malloc(n * sizeof(int));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc(lwork * sizeof(f32));
    fix->rwork = malloc(n * sizeof(f32));

    assert_non_null(fix->A);
    assert_non_null(fix->AFAC);
    assert_non_null(fix->B);
    assert_non_null(fix->X);
    assert_non_null(fix->XACT);
    assert_non_null(fix->ipiv);
    assert_non_null(fix->d);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);

    *state = fix;
    return 0;
}

static int dsysv_teardown(void** state)
{
    dsysv_fixture_t* fix = *state;
    if (fix) {
        free(fix->A);
        free(fix->AFAC);
        free(fix->B);
        free(fix->X);
        free(fix->XACT);
        free(fix->ipiv);
        free(fix->d);
        free(fix->work);
        free(fix->rwork);
        free(fix);
    }
    return 0;
}

/* Size-specific setups */
static int setup_5(void** state) { return dsysv_setup(state, 5, 1); }
static int setup_10(void** state) { return dsysv_setup(state, 10, 1); }
static int setup_20(void** state) { return dsysv_setup(state, 20, 1); }
static int setup_50(void** state) { return dsysv_setup(state, 50, 1); }
static int setup_5_nrhs2(void** state) { return dsysv_setup(state, 5, 2); }
static int setup_10_nrhs2(void** state) { return dsysv_setup(state, 10, 2); }
static int setup_20_nrhs2(void** state) { return dsysv_setup(state, 20, 2); }
static int setup_50_nrhs2(void** state) { return dsysv_setup(state, 50, 2); }
static int setup_5_nrhs5(void** state) { return dsysv_setup(state, 5, 5); }
static int setup_10_nrhs5(void** state) { return dsysv_setup(state, 10, 5); }
static int setup_20_nrhs5(void** state) { return dsysv_setup(state, 20, 5); }
static int setup_50_nrhs5(void** state) { return dsysv_setup(state, 50, 5); }

/**
 * Core test logic: generate matrix, call ssysv, verify residual.
 *
 * Residual = ||B - A*X|| / (||A|| * ||X|| * N * EPS)
 */
static f32 run_dsysv_test(dsysv_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info;

    slatb4("SSY", imat, fix->n, fix->n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(fix->n, fix->n, &dist, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, fix->lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate known exact solution XACT */
    for (int j = 0; j < fix->nrhs; j++) {
        for (int i = 0; i < fix->n; i++) {
            fix->XACT[i + j * fix->lda] = 1.0f + (f32)i / fix->n;
        }
    }

    /* Compute B = A * XACT (A is symmetric) */
    CBLAS_UPLO cblas_uplo = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, 1.0f, fix->A, fix->lda,
                fix->XACT, fix->lda, 0.0f, fix->B, fix->lda);

    /* Copy A to AFAC (ssysv overwrites the first argument) */
    memcpy(fix->AFAC, fix->A, fix->lda * fix->n * sizeof(f32));

    /* Copy B to X (ssysv overwrites B argument with solution) */
    memcpy(fix->X, fix->B, fix->lda * fix->nrhs * sizeof(f32));

    /* Call ssysv */
    int lwork = fix->n * 64;
    ssysv(uplo, fix->n, fix->nrhs, fix->AFAC, fix->lda, fix->ipiv,
          fix->X, fix->lda, fix->work, lwork, &info);
    assert_info_success(info);

    /* Verify: compute residual = ||B - A*X|| / (||A|| * ||X|| * N * EPS) */
    f32 eps = FLT_EPSILON;

    /* Compute ||A|| */
    f32 anorm_val = slansy("1", uplo, fix->n, fix->A, fix->lda, fix->rwork);

    /* Compute ||X|| */
    f32 xnorm = slange("1", fix->n, fix->nrhs, fix->X, fix->lda, fix->rwork);

    /* Compute B - A*X: use B as workspace (already have original in fix->B) */
    f32* resid_vec = malloc(fix->lda * fix->nrhs * sizeof(f32));
    assert_non_null(resid_vec);
    memcpy(resid_vec, fix->B, fix->lda * fix->nrhs * sizeof(f32));

    /* resid_vec = B - A*X = B - 1.0*A*X + 1.0*resid_vec => alpha=-1, beta=1 */
    cblas_ssymm(CblasColMajor, CblasLeft, cblas_uplo,
                fix->n, fix->nrhs, -1.0f, fix->A, fix->lda,
                fix->X, fix->lda, 1.0f, resid_vec, fix->lda);

    /* Compute ||B - A*X|| */
    f32 rnorm = slange("1", fix->n, fix->nrhs, resid_vec, fix->lda, fix->rwork);
    free(resid_vec);

    /* Normalized residual */
    f32 resid;
    if (anorm_val <= 0.0f || xnorm <= 0.0f) {
        resid = (rnorm > 0.0f) ? 1.0f / eps : 0.0f;
    } else {
        resid = rnorm / (anorm_val * xnorm * fix->n * eps);
    }

    return resid;
}

/*
 * Test well-conditioned matrices (types 1-6) with UPLO='U'
 */
static void test_dsysv_upper(void** state)
{
    dsysv_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dsysv_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

/*
 * Test well-conditioned matrices (types 1-6) with UPLO='L'
 */
static void test_dsysv_lower(void** state)
{
    dsysv_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dsysv_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

#define DSYSV_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsysv_upper, setup_fn, dsysv_teardown), \
    cmocka_unit_test_setup_teardown(test_dsysv_lower, setup_fn, dsysv_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Single RHS */
        DSYSV_TESTS(setup_5),
        DSYSV_TESTS(setup_10),
        DSYSV_TESTS(setup_20),
        DSYSV_TESTS(setup_50),

        /* Multiple RHS (nrhs=2) */
        DSYSV_TESTS(setup_5_nrhs2),
        DSYSV_TESTS(setup_10_nrhs2),
        DSYSV_TESTS(setup_20_nrhs2),
        DSYSV_TESTS(setup_50_nrhs2),

        /* Multiple RHS (nrhs=5) */
        DSYSV_TESTS(setup_5_nrhs5),
        DSYSV_TESTS(setup_10_nrhs5),
        DSYSV_TESTS(setup_20_nrhs5),
        DSYSV_TESTS(setup_50_nrhs5),
    };

    return cmocka_run_group_tests_name("dsysv", tests, NULL, NULL);
}
