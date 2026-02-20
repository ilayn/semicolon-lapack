/**
 * @file test_ssytrs.c
 * @brief CMocka test suite for ssytrs (symmetric indefinite solve).
 *
 * Tests the solve routine ssytrs which solves A*X = B using the Bunch-Kaufman
 * factorization computed by ssytrf.
 *
 * Verification: compute ||B - A*X|| / (||A|| * ||X|| * N * EPS)
 * where A*X is computed via cblas_dsymm since A is symmetric.
 *
 * Tests both UPLO='U' and UPLO='L', with NRHS=1 and NRHS=5.
 * Sizes: 5, 10, 20, 50.
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0f
#include <cblas.h>

/* Routines under test */
extern void ssytrf(const char* uplo, const int n, f32* restrict A,
                   const int lda, int* restrict ipiv, f32* restrict work,
                   const int lwork, int* info);
extern void ssytrs(const char* uplo, const int n, const int nrhs,
                   const f32* const restrict A, const int lda,
                   const int* const restrict ipiv,
                   f32* const restrict B, const int ldb, int* info);

/* Norm routines */
extern f32 slansy(const char* norm, const char* uplo, const int n,
                     const f32* restrict A, const int lda,
                     f32* restrict work);
extern f32 slange(const char* norm, const int m, const int n,
                     const f32* restrict A, const int lda,
                     f32* restrict work);

/*
 * Test fixture
 */
typedef struct {
    int n, nrhs;
    int lda;
    f32* A;       /* Original matrix */
    f32* AFAC;    /* Factored matrix */
    f32* B;       /* RHS (overwritten with solution) */
    f32* X;       /* Computed solution (copy of B after solve) */
    f32* XACT;    /* Known exact solution */
    int* ipiv;       /* Pivot indices from ssytrf */
    f32* d;       /* Singular values for slatms */
    f32* work;    /* Workspace for slatms and ssytrf */
    f32* rwork;   /* Workspace for norm computations */
    uint64_t seed;
} dsytrs_fixture_t;

static uint64_t g_seed = 7100;

static int dsytrs_setup(void** state, int n, int nrhs)
{
    dsytrs_fixture_t* fix = malloc(sizeof(dsytrs_fixture_t));
    assert_non_null(fix);

    fix->n = n;
    fix->nrhs = nrhs;
    fix->lda = n;
    fix->seed = g_seed++;

    int lwork = n * 64;  /* NB=64 for SYTRF from lapack_tuning.h */

    fix->A = malloc(fix->lda * n * sizeof(f32));
    fix->AFAC = malloc(fix->lda * n * sizeof(f32));
    fix->B = malloc(n * nrhs * sizeof(f32));
    fix->X = malloc(n * nrhs * sizeof(f32));
    fix->XACT = malloc(n * nrhs * sizeof(f32));
    fix->ipiv = malloc(n * sizeof(int));
    fix->d = malloc(n * sizeof(f32));
    fix->work = malloc((3 * n > lwork ? 3 * n : lwork) * sizeof(f32));
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

static int dsytrs_teardown(void** state)
{
    dsytrs_fixture_t* fix = *state;
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

/* Size-specific setup functions (nrhs=1) */
static int setup_5(void** state) { return dsytrs_setup(state, 5, 1); }
static int setup_10(void** state) { return dsytrs_setup(state, 10, 1); }
static int setup_20(void** state) { return dsytrs_setup(state, 20, 1); }
static int setup_50(void** state) { return dsytrs_setup(state, 50, 1); }

/* Multi-RHS setups (nrhs=5) */
static int setup_5_nrhs5(void** state) { return dsytrs_setup(state, 5, 5); }
static int setup_10_nrhs5(void** state) { return dsytrs_setup(state, 10, 5); }
static int setup_20_nrhs5(void** state) { return dsytrs_setup(state, 20, 5); }
static int setup_50_nrhs5(void** state) { return dsytrs_setup(state, 50, 5); }

/**
 * Core test logic: generate symmetric matrix, factorize with ssytrf,
 * solve with ssytrs, verify residual.
 */
static f32 run_dsytrs_test(dsytrs_fixture_t* fix, int imat, const char* uplo)
{
    char type, dist;
    int kl, ku, mode;
    f32 anorm, cndnum;
    int info;
    int n = fix->n;
    int nrhs = fix->nrhs;
    int lda = fix->lda;
    int lwork = n * 64;

    /* Get matrix parameters for symmetric indefinite */
    slatb4("SSY", imat, n, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);

    /* Generate symmetric test matrix A */
    char sym_str[2] = {type, '\0'};
    uint64_t rng_state[4];
    rng_seed(rng_state, fix->seed);
    slatms(n, n, &dist, sym_str, fix->d, mode, cndnum, anorm,
           kl, ku, "N", fix->A, lda, fix->work, &info, rng_state);
    assert_int_equal(info, 0);

    /* Generate known exact solution XACT */
    for (int j = 0; j < nrhs; j++) {
        for (int i = 0; i < n; i++) {
            fix->XACT[i + j * n] = 1.0f + (f32)i / n;
        }
    }

    /* Compute B = A * XACT via cblas_dsymm (A is symmetric) */
    CBLAS_UPLO uplo_enum = (uplo[0] == 'U') ? CblasUpper : CblasLower;
    cblas_ssymm(CblasColMajor, CblasLeft, uplo_enum,
                n, nrhs, 1.0f, fix->A, lda,
                fix->XACT, n, 0.0f, fix->B, n);

    /* Copy B into X (ssytrs will overwrite X with the solution) */
    memcpy(fix->X, fix->B, n * nrhs * sizeof(f32));

    /* Factor A with ssytrf */
    memcpy(fix->AFAC, fix->A, lda * n * sizeof(f32));
    ssytrf(uplo, n, fix->AFAC, lda, fix->ipiv, fix->work, lwork, &info);
    assert_info_success(info);

    /* Solve A*X = B using ssytrs */
    ssytrs(uplo, n, nrhs, fix->AFAC, lda, fix->ipiv,
           fix->X, n, &info);
    assert_info_success(info);

    /* Compute residual: ||B - A*X|| / (||A|| * ||X|| * N * EPS) */

    /* R = B (copy original B into work area, reuse fix->B as residual) */
    f32* residvec = malloc(n * nrhs * sizeof(f32));
    assert_non_null(residvec);
    cblas_scopy(n * nrhs, fix->B, 1, residvec, 1);

    /* R = B - A * X */
    cblas_ssymm(CblasColMajor, CblasLeft, uplo_enum,
                n, nrhs, -1.0f, fix->A, lda,
                fix->X, n, 1.0f, residvec, n);

    /* ||A|| in 1-norm */
    f32 anorm_val = slansy("1", uplo, n, fix->A, lda, fix->rwork);

    /* ||X|| in 1-norm */
    f32 xnorm = slange("1", n, nrhs, fix->X, n, fix->rwork);

    /* ||R|| in 1-norm */
    f32 rnorm = slange("1", n, nrhs, residvec, n, fix->rwork);

    free(residvec);

    /* Compute normalized residual */
    f32 eps = FLT_EPSILON;
    f32 resid;
    if (anorm_val <= 0.0f || xnorm <= 0.0f) {
        resid = (rnorm > 0.0f) ? 1.0f / eps : 0.0f;
    } else {
        resid = rnorm / (anorm_val * xnorm * n * eps);
    }

    return resid;
}

/*
 * Test well-conditioned matrices (imat 1-6) with UPLO='U'
 */
static void test_dsytrs_wellcond_upper(void** state)
{
    dsytrs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dsytrs_test(fix, imat, "U");
        assert_residual_ok(resid);
    }
}

/*
 * Test well-conditioned matrices (imat 1-6) with UPLO='L'
 */
static void test_dsytrs_wellcond_lower(void** state)
{
    dsytrs_fixture_t* fix = *state;
    for (int imat = 1; imat <= 6; imat++) {
        fix->seed = g_seed++;
        f32 resid = run_dsytrs_test(fix, imat, "L");
        assert_residual_ok(resid);
    }
}

#define DSYTRS_TESTS(setup_fn) \
    cmocka_unit_test_setup_teardown(test_dsytrs_wellcond_upper, setup_fn, dsytrs_teardown), \
    cmocka_unit_test_setup_teardown(test_dsytrs_wellcond_lower, setup_fn, dsytrs_teardown)

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* Single RHS (nrhs=1) */
        DSYTRS_TESTS(setup_5),
        DSYTRS_TESTS(setup_10),
        DSYTRS_TESTS(setup_20),
        DSYTRS_TESTS(setup_50),

        /* Multiple RHS (nrhs=5) */
        DSYTRS_TESTS(setup_5_nrhs5),
        DSYTRS_TESTS(setup_10_nrhs5),
        DSYTRS_TESTS(setup_20_nrhs5),
        DSYTRS_TESTS(setup_50_nrhs5),
    };

    return cmocka_run_group_tests_name("dsytrs", tests, NULL, NULL);
}
