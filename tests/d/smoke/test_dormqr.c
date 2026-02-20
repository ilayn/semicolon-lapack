/**
 * @file test_dormqr.c
 * @brief CMocka test suite for dormqr/dormlq (Q application).
 *
 * Tests both DORMQR (QR Q-application) and DORMLQ (LQ Q-application)
 * by comparing implicit Q application via DORM* routines against
 * explicit Q*C or C*Q via DGEMM.
 *
 * Verification:
 *   dqrt03: 4 residuals for DORMQR (Left/Right × NoTrans/Trans)
 *   dlqt03: 4 residuals for DORMLQ (Left/Right × NoTrans/Trans)
 */

#include "test_harness.h"
#include "verify.h"
#include "test_rng.h"

/* Test threshold - see LAPACK dtest.in */
#define THRESH 20.0
#include <cblas.h>

/* Factorization routines */
extern void dgeqrf(const int m, const int n,
                   f64 * const restrict A, const int lda,
                   f64 * const restrict tau,
                   f64 * const restrict work, const int lwork, int *info);
extern void dgelqf(const int m, const int n,
                   f64 * const restrict A, const int lda,
                   f64 * const restrict tau,
                   f64 * const restrict work, const int lwork, int *info);
extern void dlacpy(const char *uplo, const int m, const int n,
                   const f64 * const restrict A, const int lda,
                   f64 * const restrict B, const int ldb);

typedef struct {
    int m, n;    /* dimensions of the matrix to factor */
    int nrhs;    /* "other" dimension for C in the multiplication test */
    int lda;
    f64 *A, *AF;
    f64 *C, *CC, *Q;
    f64 *tau, *work, *rwork;
    f64 *d, *genwork;
    int lwork;
    uint64_t seed;
} orm_fixture_t;

static uint64_t g_seed = 7001;

static int orm_setup(void **state, int m, int n, int nrhs)
{
    orm_fixture_t *fix = malloc(sizeof(orm_fixture_t));
    assert_non_null(fix);

    fix->m = m;
    fix->n = n;
    fix->nrhs = nrhs;
    int maxmn = m > n ? m : n;
    int maxall = maxmn > nrhs ? maxmn : nrhs;
    fix->lda = maxall;
    fix->seed = g_seed++;

    int minmn = m < n ? m : n;
    fix->lwork = maxall * 64;

    fix->A = calloc(fix->lda * maxall, sizeof(f64));
    fix->AF = calloc(fix->lda * maxall, sizeof(f64));
    fix->C = calloc(fix->lda * maxall, sizeof(f64));
    fix->CC = calloc(fix->lda * maxall, sizeof(f64));
    fix->Q = calloc(fix->lda * maxall, sizeof(f64));
    fix->tau = calloc(minmn, sizeof(f64));
    fix->work = calloc(fix->lwork, sizeof(f64));
    fix->rwork = calloc(maxall, sizeof(f64));
    fix->d = calloc(minmn, sizeof(f64));
    fix->genwork = calloc(3 * maxall, sizeof(f64));

    assert_non_null(fix->A);
    assert_non_null(fix->AF);
    assert_non_null(fix->C);
    assert_non_null(fix->CC);
    assert_non_null(fix->Q);
    assert_non_null(fix->tau);
    assert_non_null(fix->work);
    assert_non_null(fix->rwork);
    assert_non_null(fix->d);
    assert_non_null(fix->genwork);

    *state = fix;
    return 0;
}

static int orm_teardown(void **state)
{
    orm_fixture_t *fix = *state;
    if (fix) {
        free(fix->A); free(fix->AF);
        free(fix->C); free(fix->CC); free(fix->Q);
        free(fix->tau); free(fix->work); free(fix->rwork);
        free(fix->d); free(fix->genwork);
        free(fix);
    }
    return 0;
}

/* Setup with m=n (square) and nrhs for the C matrix dimension */
static int setup_5x5(void **state) { return orm_setup(state, 5, 5, 3); }
static int setup_10x10(void **state) { return orm_setup(state, 10, 10, 5); }
static int setup_20x20(void **state) { return orm_setup(state, 20, 20, 7); }
static int setup_50x50(void **state) { return orm_setup(state, 50, 50, 10); }
static int setup_20x10(void **state) { return orm_setup(state, 20, 10, 5); }
static int setup_10x20(void **state) { return orm_setup(state, 10, 20, 5); }

/**
 * Test DORMQR: Apply Q from QR factorization.
 * Q is m-by-m, C is m-by-nrhs (Left) or nrhs-by-m (Right).
 */
static void test_dormqr(void **state)
{
    orm_fixture_t *fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = m < n ? m : n;
    f64 result[4];
    int info;

    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        int kl, ku, mode;
        f64 anorm, cndnum;

        dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        uint64_t rng_state[4];
        rng_seed(rng_state, fix->seed);
        dlatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
        assert_int_equal(info, 0);

        /* QR factorize into AF */
        dlacpy("F", m, n, fix->A, fix->lda, fix->AF, fix->lda);
        dgeqrf(m, n, fix->AF, fix->lda, fix->tau, fix->work, fix->lwork, &info);
        assert_int_equal(info, 0);

        /* Test all 4 combinations of side/trans.
         * dqrt03(m, n, k, AF, C, CC, Q, lda, tau, work, lwork, rwork, result)
         * Here m = order of Q, n = other dimension of C, k = number of reflectors */
        dqrt03(m, fix->nrhs, minmn, fix->AF, fix->C, fix->CC, fix->Q,
               fix->lda, fix->tau, fix->work, fix->lwork, fix->rwork, result);

        assert_residual_ok(result[0]);
        assert_residual_ok(result[1]);
        assert_residual_ok(result[2]);
        assert_residual_ok(result[3]);
    }
}

/**
 * Test DORMLQ: Apply Q from LQ factorization.
 * Q is n-by-n, C is n-by-nrhs (Left) or nrhs-by-n (Right).
 */
static void test_dormlq(void **state)
{
    orm_fixture_t *fix = *state;
    int m = fix->m, n = fix->n;
    int minmn = m < n ? m : n;
    f64 result[4];
    int info;

    for (int imat = 1; imat <= 4; imat++) {
        fix->seed = g_seed++;
        char type, dist;
        int kl, ku, mode;
        f64 anorm, cndnum;

        dlatb4("DGE", imat, m, n, &type, &kl, &ku, &anorm, &mode, &cndnum, &dist);
        uint64_t rng_state[4];
        rng_seed(rng_state, fix->seed);
        dlatms(m, n, &dist, &type, fix->d, mode, cndnum, anorm,
               kl, ku, "N", fix->A, fix->lda, fix->genwork, &info, rng_state);
        assert_int_equal(info, 0);

        /* LQ factorize into AF */
        dlacpy("F", m, n, fix->A, fix->lda, fix->AF, fix->lda);
        dgelqf(m, n, fix->AF, fix->lda, fix->tau, fix->work, fix->lwork, &info);
        assert_int_equal(info, 0);

        /* Test all 4 combinations.
         * dlqt03(m, n, k, ...) where m=other dim, n=order of Q, k=reflectors */
        dlqt03(fix->nrhs, n, minmn, fix->AF, fix->C, fix->CC, fix->Q,
               fix->lda, fix->tau, fix->work, fix->lwork, fix->rwork, result);

        assert_residual_ok(result[0]);
        assert_residual_ok(result[1]);
        assert_residual_ok(result[2]);
        assert_residual_ok(result[3]);
    }
}

int main(void)
{
    const struct CMUnitTest tests[] = {
        /* DORMQR tests */
        cmocka_unit_test_setup_teardown(test_dormqr, setup_5x5, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_10x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_20x20, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_50x50, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_20x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormqr, setup_10x20, orm_teardown),
        /* DORMLQ tests */
        cmocka_unit_test_setup_teardown(test_dormlq, setup_5x5, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_10x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_20x20, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_50x50, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_20x10, orm_teardown),
        cmocka_unit_test_setup_teardown(test_dormlq, setup_10x20, orm_teardown),
    };
    return cmocka_run_group_tests_name("dormqr", tests, NULL, NULL);
}
